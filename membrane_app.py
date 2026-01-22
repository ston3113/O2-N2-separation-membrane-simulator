import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import time
import streamlit as st
import pandas as pd

# --- 기본 상수 및 파라미터 ---
STP_MOLAR_VOLUME = 22414.0  # cm³/mol
STAGE_CUT_THRESHOLD = 0.7    # [NEW] 스테이지 컷 경고 임계값 (70%)

# 단위 환산 계수
BAR_TO_ATM = 0.986923
ATM_TO_BAR = 1.01325
M3H_TO_CM3S = 1_000_000.0 / 3600.0
CM3S_TO_M3H = 3600.0 / 1_000_000.0
M2_TO_CM2 = 10000.0
CM2_TO_M2 = 1.0 / M2_TO_CM2
GPU_TO_STD_UNITS = 1e-6 * 76.0 

PROCESS_PARAMS_VOL = {
    "p_u_default": 8.00,
    "p_p_default": 0.80,
}

DEFAULT_L_GPU = np.array([36.0, 146.0, 300.0]) 
RAW_FEED_FLUX_M3H = 300.00 
RAW_FEED_COMP = np.array([0.807, 0.107, 0.086])
AREA_LIST_M2 = [100.0, 66.66, 50.0, 33.33]

# ==================================================================
# 2. MembraneStage 클래스
# ==================================================================
class MembraneStage:
    def __init__(self, name):
        self.name = name
        self.area = 0.0
        self.stage_cut = 0.0
        self.feed_flux = 0.0
        self.feed_comp = None
        self.permeate_flux = 0.0
        self.permeate_comp = None
        self.retentate_flux = 0.0
        self.retentate_comp = None

    def _calc_yi_system(self, x, params):
        n_comp = len(x)
        L, p_u, p_p = params["L"], params["p_u"], params["p_p"]
        x_safe = np.clip(x, 1e-12, 1.0)

        def equations(yi):
            yi_safe = np.clip(yi, 1e-12, 1.0)
            eqs = []
            for i in range(n_comp - 1):
                driving_force_i = L[i] * (p_u * x_safe[i] - p_p * yi_safe[i])
                driving_force_j = L[i + 1] * (p_u * x_safe[i + 1] - p_p * yi_safe[i + 1])
                eqs.append(yi_safe[i] * driving_force_j - yi_safe[i + 1] * driving_force_i)
            eqs.append(np.sum(yi) - 1.0)
            return eqs

        yi_sol, _, ier, _ = fsolve(equations, x_safe.copy(), full_output=True)
        if ier != 1: pass
        return np.clip(yi_sol, 1e-10, 1.0)

    def _odes(self, A, y_state, params):
        n_comp = len(params["L"])
        x = y_state[:n_comp]
        Lu = y_state[n_comp]
        x = np.maximum(x, 0)
        x /= np.sum(x)
        yi = self._calc_yi_system(x, params)
        Ji = params["L"] * (params["p_u"] * x - params["p_p"] * yi)
        Ji = np.maximum(Ji, 0)
        dLu_dA = -np.sum(Ji)
        if Lu < 1e-9:
            dxi_dA = np.zeros(n_comp)
        else:
            dxi_dA = (x * np.sum(Ji) - Ji) / Lu
        return np.hstack((dxi_dA, dLu_dA))

    def run(self, feed_flux, feed_comp, area_target, params):
        if not np.isclose(np.sum(feed_comp), 1.0):
            feed_comp = feed_comp / np.sum(feed_comp)
        
        self.feed_flux = feed_flux
        self.feed_comp = feed_comp
        n_comp = len(feed_comp)
        y_state0 = np.hstack((feed_comp, feed_flux))

        sol = solve_ivp(
            fun=self._odes,
            t_span=[0, area_target],
            y0=y_state0,
            method='RK45',
            args=(params,),
        )
        if sol.status != 0:
            raise RuntimeError(f"solve_ivp failed at {self.name} status {sol.status}")

        self.area = sol.t[-1]
        final_y_state = sol.y[:, -1]
        self.retentate_flux = final_y_state[n_comp]
        self.retentate_comp = np.maximum(final_y_state[:n_comp], 0)
        self.retentate_comp /= np.sum(self.retentate_comp)
        self.permeate_flux = self.feed_flux - self.retentate_flux
        if self.permeate_flux > 1e-9:
            permeate_moles = (self.feed_flux * self.feed_comp) - (self.retentate_flux * self.retentate_comp)
            self.permeate_comp = np.maximum(permeate_moles, 0)
            self.permeate_comp /= np.sum(self.permeate_comp)
        else:
            self.permeate_comp = np.zeros(n_comp)
        self.stage_cut = self.permeate_flux / self.feed_flux if self.feed_flux > 1e-9 else 0.0
        return True

# ==================================================================
# 3. Process 클래스
# ==================================================================
class Process:
    def __init__(self, params_list, area_list, stp_molar_volume=22414.0):
        self.params_list = params_list
        self.area_list = area_list
        self.stages = []
        self.stp_molar_volume = stp_molar_volume
        self.log_widget = st.empty()

    def _calculate_mixed_feed(self, raw_feed_flux, raw_feed_comp, ret_3, ret_4):
        n_comp = len(raw_feed_comp)
        raw_feed_moles = raw_feed_flux * raw_feed_comp
        ret_3_moles = ret_3['flux'] * ret_3['comp'] if ret_3 else np.zeros(n_comp)
        ret_4_moles = ret_4['flux'] * ret_4['comp'] if ret_4 else np.zeros(n_comp)
        total_moles = raw_feed_moles + ret_3_moles + ret_4_moles
        final_feed_flux = np.sum(total_moles)
        return (final_feed_flux, total_moles / final_feed_flux) if final_feed_flux > 1e-9 else (0.0, np.zeros(n_comp))

    def run_with_recycle(self, raw_feed_flux, raw_feed_comp, max_iterations=50, tolerance=1e-6):
        n_comp = len(raw_feed_comp)
        recycled_ret_3 = {'flux': 0.0, 'comp': np.zeros(n_comp)}
        recycled_ret_4 = {'flux': 0.0, 'comp': np.zeros(n_comp)}
        self.log_widget.text("====== Recycle Process Simulation Start ======")

        for i in range(max_iterations):
            stage1_flux, stage1_comp = self._calculate_mixed_feed(raw_feed_flux, raw_feed_comp, recycled_ret_3, recycled_ret_4)
            curr_flux, curr_comp = stage1_flux, stage1_comp
            current_stages = []

            try:
                for j, area_target in enumerate(self.area_list):
                    stage = MembraneStage(f"Stage-{j + 1}")
                    stage.run(curr_flux, curr_comp, area_target, self.params_list[j])
                    current_stages.append(stage)
                    curr_flux, curr_comp = stage.permeate_flux, stage.permeate_comp
            except Exception as e:
                self.log_widget.text(f"ERROR: {e}")
                return False

            new_ret_3 = {'flux': current_stages[2].retentate_flux, 'comp': current_stages[2].retentate_comp}
            new_ret_4 = {'flux': current_stages[3].retentate_flux, 'comp': current_stages[3].retentate_comp}
            error = abs(recycled_ret_3['flux'] - new_ret_3['flux']) + abs(recycled_ret_4['flux'] - new_ret_4['flux'])

            if error < tolerance:
                self.stages = current_stages
                self.log_widget.text(f"SUCCESS: Converged after {i+1} iterations.")
                return True
            recycled_ret_3, recycled_ret_4 = new_ret_3, new_ret_4
        return False

# ==================================================================
# 4. Streamlit UI
# ==================================================================
st.set_page_config(layout="wide")
st.title("🧪 4-Stage Membrane Simulator")

COMP_NAMES_FIXED = ['N2', 'O2', 'CO2']

with st.sidebar:
    st.header("1. 초기 원료 (Raw Feed)")
    feed_flux_m3h = st.number_input("총 유량 (m³/h)", value=RAW_FEED_FLUX_M3H)
    comp_1 = st.number_input(f"{COMP_NAMES_FIXED[0]} (Comp 1)", value=RAW_FEED_COMP[0], format="%.4f")
    comp_2 = st.number_input(f"{COMP_NAMES_FIXED[1]} (Comp 2)", value=RAW_FEED_COMP[1], format="%.4f")
    comp_3 = st.number_input(f"{COMP_NAMES_FIXED[2]} (Comp 3)", value=RAW_FEED_COMP[2], format="%.4f")

    st.header("2. 스테이지별 파라미터")
    p_u_def = PROCESS_PARAMS_VOL["p_u_default"]
    p_p_def = PROCESS_PARAMS_VOL["p_p_default"]

    def stage_input(idx):
        st.subheader(f"Stage {idx+1}")
        col1, col2 = st.columns(2)
        with col1: a = st.number_input(f"Area (m²)", value=AREA_LIST_M2[idx], key=f"a{idx}")
        with col2: pu = st.number_input(f"p_u (bar)", value=p_u_def, key=f"pu{idx}")
        pp = st.number_input(f"p_p (bar)", value=p_p_def, key=f"pp{idx}")
        gpu1 = st.number_input(f"GPU {COMP_NAMES_FIXED[0]}", value=DEFAULT_L_GPU[0], key=f"g1{idx}")
        gpu2 = st.number_input(f"GPU {COMP_NAMES_FIXED[1]}", value=DEFAULT_L_GPU[1], key=f"g2{idx}")
        gpu3 = st.number_input(f"GPU {COMP_NAMES_FIXED[2]}", value=DEFAULT_L_GPU[2], key=f"g3{idx}")
        return a, pu, pp, [gpu1, gpu2, gpu3]

    inputs = [stage_input(i) for i in range(4)]
    run_button = st.button("🚀 시뮬레이션 실행")

if run_button:
    try:
        raw_comp = np.array([comp_1, comp_2, comp_3])
        params_list = []
        for a, pu, pp, gpus in inputs:
            params_list.append({
                "L": np.array(gpus) * GPU_TO_STD_UNITS / STP_MOLAR_VOLUME,
                "p_u": pu * BAR_TO_ATM,
                "p_p": pp * BAR_TO_ATM
            })
        areas = [i[0] * M2_TO_CM2 for i in inputs]
        
        proc = Process(params_list, areas)
        if proc.run_with_recycle(feed_flux_m3h * M3H_TO_CM3S / STP_MOLAR_VOLUME, raw_comp):
            st.subheader("🏁 최종 수렴 결과")
            
            # [NEW] 스테이지 컷 경고 로직
            high_cut_stages = [s.name for s in proc.stages if s.stage_cut > STAGE_CUT_THRESHOLD]
            if high_cut_stages:
                st.warning(f"⚠️ **경고: 높은 스테이지 컷 검출!** ({', '.join(high_cut_stages)}) 의 스테이지 컷이 {STAGE_CUT_THRESHOLD*100:.0f}%를 초과했습니다. 이는 제품의 순도를 크게 떨어뜨릴 수 있습니다. 면적을 줄이거나 유량을 조절하세요.")

            vol_conv = STP_MOLAR_VOLUME * CM3S_TO_M3H
            results = []
            for s in proc.stages:
                row = {"Stage": s.name, "Area (m²)": s.area * CM2_TO_M2, "Stage Cut (θ)": s.stage_cut, "Feed (m³/h)": s.feed_flux * vol_conv}
                for i, n in enumerate(COMP_NAMES_FIXED): row[f"Perm {n}"] = s.permeate_comp[i]
                results.append(row)
            
            df = pd.DataFrame(results)

            # [NEW] 테이블 스타일링 (Stage Cut이 높으면 빨간색 표시)
            def highlight_high_cut(val):
                color = 'red' if isinstance(val, float) and val > STAGE_CUT_THRESHOLD else 'black'
                weight = 'bold' if color == 'red' else 'normal'
                return f'color: {color}; font-weight: {weight}'

            st.dataframe(df.style.format("{:.4f}", subset=df.columns[1:]).applymap(highlight_high_cut, subset=['Stage Cut (θ)']), use_container_width=True)
    except Exception as e:
        st.exception(e)
