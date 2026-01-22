import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import time
import streamlit as st
import pandas as pd

# --- 기본 상수 및 단위 환산 ---
STP_MOLAR_VOLUME = 22414.0  
STAGE_CUT_THRESHOLD = 0.7   # 스테이지 컷 경고 기준 (70%)

BAR_TO_ATM = 0.986923
ATM_TO_BAR = 1.01325
M3H_TO_CM3S = 1_000_000.0 / 3600.0
CM3S_TO_M3H = 3600.0 / 1_000_000.0
M2_TO_CM2 = 10000.0
CM2_TO_M2 = 1.0 / M2_TO_CM2
GPU_TO_STD_UNITS = 1e-6 * 76.0 

PROCESS_PARAMS_VOL = {"p_u_default": 8.00, "p_p_default": 0.80}
DEFAULT_L_GPU = np.array([36.0, 146.0, 300.0]) 
RAW_FEED_FLUX_M3H = 300.00 
RAW_FEED_COMP = np.array([0.807, 0.107, 0.086]) # N2, O2, CO2
AREA_LIST_M2 = [100.0, 66.66, 50.0, 33.33]

# ==================================================================
# 2. MembraneStage 클래스
# ==================================================================
class MembraneStage:
    def __init__(self, name):
        self.name = name
        self.area, self.stage_cut = 0.0, 0.0
        self.feed_flux, self.feed_comp = 0.0, None
        self.permeate_flux, self.permeate_comp = 0.0, None
        self.retentate_flux, self.retentate_comp = 0.0, None

    def _calc_yi_system(self, x, params):
        n_comp = len(x)
        L, p_u, p_p = params["L"], params["p_u"], params["p_p"]
        x_safe = np.clip(x, 1e-12, 1.0)
        def equations(yi):
            yi_safe = np.clip(yi, 1e-12, 1.0)
            eqs = []
            for i in range(n_comp - 1):
                dfi = L[i] * (p_u * x_safe[i] - p_p * yi_safe[i])
                dfj = L[i + 1] * (p_u * x_safe[i + 1] - p_p * yi_safe[i + 1])
                eqs.append(yi_safe[i] * dfj - yi_safe[i + 1] * dfi)
            eqs.append(np.sum(yi) - 1.0)
            return eqs
        yi_sol, _, ier, _ = fsolve(equations, x_safe.copy(), full_output=True)
        return np.clip(yi_sol, 1e-10, 1.0)

    def _odes(self, A, y_state, params):
        n_comp = len(params["L"])
        x, Lu = y_state[:n_comp], y_state[n_comp]
        x = np.maximum(x, 0); x /= np.sum(x)
        yi = self._calc_yi_system(x, params)
        Ji = params["L"] * (params["p_u"] * x - params["p_p"] * yi)
        Ji = np.maximum(Ji, 0); dLu_dA = -np.sum(Ji)
        dxi_dA = (x * np.sum(Ji) - Ji) / Lu if Lu > 1e-9 else np.zeros(n_comp)
        return np.hstack((dxi_dA, dLu_dA))

    def run(self, feed_flux, feed_comp, area_target, params):
        self.feed_flux, self.feed_comp = feed_flux, feed_comp / np.sum(feed_comp)
        sol = solve_ivp(fun=self._odes, t_span=[0, area_target], y0=np.hstack((self.feed_comp, feed_flux)), method='RK45', args=(params,))
        self.area, final_y = sol.t[-1], sol.y[:, -1]
        self.retentate_flux = final_y[-1]
        self.retentate_comp = np.maximum(final_y[:-1], 0); self.retentate_comp /= np.sum(self.retentate_comp)
        self.permeate_flux = np.maximum(self.feed_flux - self.retentate_flux, 0)
        if self.permeate_flux > 1e-9:
            self.permeate_comp = np.maximum((self.feed_flux * self.feed_comp) - (self.retentate_flux * self.retentate_comp), 0)
            self.permeate_comp /= np.sum(self.permeate_comp)
        else: self.permeate_comp = np.zeros(len(feed_comp))
        self.stage_cut = self.permeate_flux / self.feed_flux if self.feed_flux > 1e-9 else 0.0
        return True

# ==================================================================
# 3. Process 클래스 (4-Stage Recycle)
# ==================================================================
class Process:
    def __init__(self, params_list, area_list):
        self.params_list, self.area_list, self.stages = params_list, area_list, []
        self.log_widget = st.empty()

    def _calculate_mixed_feed(self, raw_flux, raw_comp, ret_3, ret_4):
        total_moles = (raw_flux * raw_comp) + (ret_3['flux'] * ret_3['comp']) + (ret_4['flux'] * ret_4['comp'])
        f_flux = np.sum(total_moles)
        return (f_flux, total_moles / f_flux) if f_flux > 1e-9 else (0.0, np.zeros(len(raw_comp)))

    def run_with_recycle(self, raw_flux, raw_comp):
        n_comp = len(raw_comp)
        r3, r4 = {'flux': 0.0, 'comp': np.zeros(n_comp)}, {'flux': 0.0, 'comp': np.zeros(n_comp)}
        for i in range(50):
            f_flux, f_comp = self._calculate_mixed_feed(raw_flux, raw_comp, r3, r4)
            curr_flux, curr_comp, current_stages = f_flux, f_comp, []
            for j, area in enumerate(self.area_list):
                s = MembraneStage(f"Stage-{j+1}")
                s.run(curr_flux, curr_comp, area, self.params_list[j])
                current_stages.append(s); curr_flux, curr_comp = s.permeate_flux, s.permeate_comp
            n3, n4 = {'flux': current_stages[2].retentate_flux, 'comp': current_stages[2].retentate_comp}, {'flux': current_stages[3].retentate_flux, 'comp': current_stages[3].retentate_comp}
            if (abs(r3['flux'] - n3['flux']) + abs(r4['flux'] - n4['flux'])) < 1e-6:
                self.stages = current_stages; return True
            r3, r4 = n3, n4
        return False

# ==================================================================
# 4. Streamlit UI
# ==================================================================
st.set_page_config(layout="wide")
st.title("🧪 4-Stage Membrane Simulator")
COMP_NAMES = ['N2', 'O2', 'CO2']

with st.sidebar:
    st.header("1. Raw Feed")
    f_m3h = st.number_input("총 유량 (m³/h)", value=RAW_FEED_FLUX_M3H)
    c1 = st.number_input(f"{COMP_NAMES[0]}", value=RAW_FEED_COMP[0], format="%.4f")
    c2 = st.number_input(f"{COMP_NAMES[1]}", value=RAW_FEED_COMP[1], format="%.4f")
    c3 = st.number_input(f"{COMP_NAMES[2]}", value=RAW_FEED_COMP[2], format="%.4f")
    st.header("2. Stage Parameters")
    def s_in(i):
        st.subheader(f"Stage {i+1}")
        a = st.number_input(f"Area (m²)", value=AREA_LIST_M2[i], key=f"a{i}")
        pu = st.number_input(f"p_u (bar)", value=PROCESS_PARAMS_VOL["p_u_default"], key=f"pu{i}")
        pp = st.number_input(f"p_p (bar)", value=PROCESS_PARAMS_VOL["p_p_default"], key=f"pp{i}")
        g1 = st.number_input(f"GPU {COMP_NAMES[0]}", value=DEFAULT_L_GPU[0], key=f"g1{i}")
        g2 = st.number_input(f"GPU {COMP_NAMES[1]}", value=DEFAULT_L_GPU[1], key=f"g2{i}")
        g3 = st.number_input(f"GPU {COMP_NAMES[2]}", value=DEFAULT_L_GPU[2], key=f"g3{i}")
        return a, pu, pp, [g1, g2, g3]
    inputs = [s_in(i) for i in range(4)]
    run_btn = st.button("🚀 시뮬레이션 실행")

if run_btn:
    p_list = [{"L": np.array(g) * GPU_TO_STD_UNITS / STP_MOLAR_VOLUME, "p_u": pu * BAR_TO_ATM, "p_p": pp * BAR_TO_ATM} for a, pu, pp, g in inputs]
    proc = Process(p_list, [i[0] * M2_TO_CM2 for i in inputs])
    if proc.run_with_recycle(f_m3h * M3H_TO_CM3S / STP_MOLAR_VOLUME, np.array([c1, c2, c3])):
        st.subheader("🏁 최종 수렴 결과")
        
        # 스테이지 컷 경고
        hc = [s.name for s in proc.stages if s.stage_cut > STAGE_CUT_THRESHOLD]
        if hc: st.warning(f"⚠️ **경고:** {', '.join(hc)}의 스테이지 컷이 {STAGE_CUT_THRESHOLD*100:.0f}%를 초과했습니다.")

        v_conv = STP_MOLAR_VOLUME * CM3S_TO_M3H
        res = []
        for j, s in enumerate(proc.stages):
            d = {"Stage": s.name, "Area (m²)": s.area * CM2_TO_M2, "p_u (bar)": inputs[j][1], "p_p (bar)": inputs[j][2], "Stage Cut (θ)": s.stage_cut, "Feed Flux (m³/h)": s.feed_flux * v_conv}
            for i, n in enumerate(COMP_NAMES): d[f"Feed {n}"] = s.feed_comp[i]
            d["Permeate Flux (m³/h)"] = s.permeate_flux * v_conv
            for i, n in enumerate(COMP_NAMES): d[f"Permeate {n}"] = s.permeate_comp[i]
            d["Retentate Flux (m³/h)"] = s.retentate_flux * v_conv
            for i, n in enumerate(COMP_NAMES): d[f"Retentate {n}"] = s.retentate_comp[i]
            res.append(d)
        
        df = pd.DataFrame(res)
        def style_sc(val): return f'color: red; font-weight: bold' if isinstance(val, float) and val > STAGE_CUT_THRESHOLD else ''
        st.dataframe(df.style.format("{:.4f}", subset=df.columns[1:]).applymap(style_sc, subset=['Stage Cut (θ)']), use_container_width=True)
