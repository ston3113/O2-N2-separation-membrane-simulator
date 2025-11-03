import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import time
import streamlit as st
import pandas as pd 

# 실행시  >> streamlit run membrane_app.py


# --- 기본 상수 및 파라미터 ---
STP_MOLAR_VOLUME = 22414.0

# [MODIFIED] 전역 p_u, p_p 제거. (기본값으로만 사용)
PROCESS_PARAMS_VOL = {
    "P": np.array([0.073e-12, 0.2178e-12, 0.2178e-12]), 
    "delta": 0.2e-6, 
    "p_u_default": 10.0, # 스테이지별 UI의 기본값으로 사용
    "p_p_default": 1.0,  # 스테이지별 UI의 기본값으로 사용
}
RAW_FEED_FLUX_VOL = 150 
RAW_FEED_COMP = np.array([0.79, 0.11, 0.10]) # 3성분 기준
AREA_LIST = [50000.0, 50000.0, 50000.0, 50000.0] # 4스테이지 기준

# ==================================================================
# 2. MembraneStage 클래스 (수정 없음)
# ==================================================================
class MembraneStage:
    """
    단일 멤브레인 스테이지의 거동을 계산하고 상태를 저장하는 클래스.
    """
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
        P, p_u, p_p = params["P"], params["p_u"], params["p_p"]
        x_safe = np.clip(x, 1e-12, 1.0) 

        def equations(yi):
            yi_safe = np.clip(yi, 1e-12, 1.0)
            eqs = []
            for i in range(n_comp - 1):
                driving_force_i = P[i] * (p_u * x_safe[i] - p_p * yi_safe[i])
                driving_force_j = P[i+1] * (p_u * x_safe[i+1] - p_p * yi_safe[i+1])
                eqs.append(yi_safe[i] * driving_force_j - yi_safe[i+1] * driving_force_i)
            eqs.append(np.sum(yi) - 1.0)
            return eqs

        yi_sol, _, ier, _ = fsolve(equations, x_safe.copy(), full_output=True)
        if ier != 1: pass 
        return np.clip(yi_sol, 1e-10, 1.0)

    def _odes(self, A, y_state, params):
        n_comp = len(params["P"])
        x = y_state[:n_comp] 
        Lu = y_state[n_comp] 

        x = np.maximum(x, 0)
        x /= np.sum(x)

        yi = self._calc_yi_system(x, params)
        Ji = (params["P"] / params["delta"]) * (params["p_u"] * x - params["p_p"] * yi)
        Ji = np.maximum(Ji, 0) 
        dLu_dA = -np.sum(Ji)

        if Lu < 1e-9: 
            dxi_dA = np.zeros(n_comp)
        else:
            dxi_dA = (x * np.sum(Ji) - Ji) / Lu
            
        return np.hstack((dxi_dA, dLu_dA))

    def run(self, feed_flux, feed_comp, area_target, params):
        if not np.isclose(np.sum(feed_comp), 1.0):
            if not np.isclose(np.sum(feed_comp), 1.0):
                st.warning(f"[{self.name}] Feed 조성의 합이 1이 아닙니다 (Sum={np.sum(feed_comp):.4f}). 정규화(normalize)하여 계산을 진행합니다.")
                feed_comp = feed_comp / np.sum(feed_comp)
            if np.any(feed_comp < 0):
                raise ValueError(f"{self.name}: 잘못된 Feed 조성입니다. Comp={feed_comp}")

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
             raise RuntimeError(f"solve_ivp failed at {self.name} with status {sol.status}: {sol.message}")
        
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

        if self.feed_flux > 1e-9:
            self.stage_cut = self.permeate_flux / self.feed_flux
        else:
            self.stage_cut = 0.0
            
        return True

# ==================================================================
# 3. Process 클래스 (수정됨)
# ==================================================================
class Process:
    # [MODIFIED] params 대신 params_list를 받도록 수정
    def __init__(self, params_list, area_list, stp_molar_volume=22414.0):
        # self.params = params # <-- OLD
        self.params_list = params_list # <-- NEW
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
        
        if final_feed_flux < 1e-9:
            return 0.0, np.zeros(n_comp)
            
        final_feed_comp = total_moles / final_feed_flux
        return final_feed_flux, final_feed_comp

    def run_with_recycle(self, raw_feed_flux, raw_feed_comp, max_iterations=50, tolerance=1e-6):
        n_comp = len(raw_feed_comp)
        
        recycled_ret_3 = {'flux': 0.0, 'comp': np.zeros(n_comp)}
        recycled_ret_4 = {'flux': 0.0, 'comp': np.zeros(n_comp)}

        log_output = "====== Recycle Process Simulation Start ======\n"
        self.log_widget.text(log_output)
        start_time = time.time()

        for i in range(max_iterations):
            log_output += f"\n--- Iteration {i+1} ---\n"
            self.log_widget.text(log_output) 

            stage1_feed_flux, stage1_feed_comp = self._calculate_mixed_feed(
                raw_feed_flux, raw_feed_comp, recycled_ret_3, recycled_ret_4
            )
            
            current_feed_flux = stage1_feed_flux
            current_feed_comp = stage1_feed_comp
            
            current_stages = []
            try:
                if len(self.area_list) < 4:
                    raise ValueError("Area 리스트는 최소 4개여야 합니다 (현재 4-stage 재활용 로직).")
                    
                for j, area_target in enumerate(self.area_list):
                    stage = MembraneStage(f"Stage-{j+1}")
                    
                    # [MODIFIED] self.params 대신 해당 스테이지의 파라미터(self.params_list[j])를 전달
                    stage_params = self.params_list[j]
                    stage.run(current_feed_flux, current_feed_comp, area_target, stage_params)
                    
                    current_stages.append(stage)
                    current_feed_flux = stage.permeate_flux
                    current_feed_comp = stage.permeate_comp
            except (ValueError, RuntimeError) as e:
                log_output += f"ERROR: Iteration failed during stage calculation. Reason: {e}\n"
                self.log_widget.text(log_output)
                return False 

            new_ret_3 = {'flux': current_stages[2].retentate_flux, 'comp': current_stages[2].retentate_comp}
            new_ret_4 = {'flux': current_stages[3].retentate_flux, 'comp': current_stages[3].retentate_comp}

            error = abs(recycled_ret_3['flux'] - new_ret_3['flux']) + abs(recycled_ret_4['flux'] - new_ret_4['flux'])
            
            log_line = (f"Recycle Flux (old->new): S3[{recycled_ret_3['flux']:.4f}->{new_ret_4['flux']:.4f}], "
                        f"S4[{recycled_ret_4['flux']:.4f}->{new_ret_4['flux']:.4f}] | Error = {error:.2e}\n")
            log_output += log_line
            self.log_widget.text(log_output)

            if error < tolerance:
                self.stages = current_stages
                log_output += f"\nSUCCESS: Converged after {i+1} iterations.\n"
                self.log_widget.text(log_output)
                end_time = time.time()
                log_output += f"\n====== Simulation Finished in {end_time - start_time:.4f} seconds ======"
                self.log_widget.text(log_output)
                return True 

            recycled_ret_3 = new_ret_3
            recycled_ret_4 = new_ret_4

        log_output += f"\nFAILURE: Did not converge after {max_iterations} iterations.\n"
        self.log_widget.text(log_output)
        return False

# ==================================================================
# 4. Streamlit UI 및 실행 로직 (스테이지별 압력 입력)
# ==================================================================

st.set_page_config(layout="wide")
st.title("🧪 4-Stage Membrane Process Simulator")

# 고정된 성분 이름 리스트
COMP_NAMES_FIXED = ['N2', 'O2', 'CO2'] 

# --- 입력창 (사이드바) ---
with st.sidebar:
    st.header("1. 공정 파라미터 (공통)")
    # [REMOVED] 전역 p_u, p_p 입력창 제거
    # p_u = st.number_input("공급측 압력 (p_u, atm)", value=PROCESS_PARAMS_VOL["p_u"])
    # p_p = st.number_input("투과측 압력 (p_p, atm)", value=PROCESS_PARAMS_VOL["p_p"])
    delta = st.number_input("막 두께 (delta, cm)", value=PROCESS_PARAMS_VOL["delta"], format="%.2e")
    
    st.subheader("막 투과도 (P)")
    p_1 = st.number_input(f"P for {COMP_NAMES_FIXED[0]}", value=PROCESS_PARAMS_VOL["P"][0], format="%.4e")
    p_2 = st.number_input(f"P for {COMP_NAMES_FIXED[1]}", value=PROCESS_PARAMS_VOL["P"][1], format="%.4e")
    p_3 = st.number_input(f"P for {COMP_NAMES_FIXED[2]}", value=PROCESS_PARAMS_VOL["P"][2], format="%.4e")

    st.header("2. 초기 원료 (Raw Feed)")
    feed_flux_vol = st.number_input("총 유량 (cm³/s)", value=RAW_FEED_FLUX_VOL)
    
    st.subheader("초기 조성 (몰분율)")
    comp_1 = st.number_input(f"{COMP_NAMES_FIXED[0]} (Comp 1)", value=RAW_FEED_COMP[0], format="%.4f")
    comp_2 = st.number_input(f"{COMP_NAMES_FIXED[1]} (Comp 2)", value=RAW_FEED_COMP[1], format="%.4f")
    comp_3 = st.number_input(f"{COMP_NAMES_FIXED[2]} (Comp 3)", value=RAW_FEED_COMP[2], format="%.4f")

    # --- [MODIFIED] 스테이지별 Area 및 Pressure 입력 ---
    st.header("3. 스테이지별 파라미터")
    
    # 기본값 변수
    p_u_default = PROCESS_PARAMS_VOL["p_u_default"]
    p_p_default = PROCESS_PARAMS_VOL["p_p_default"]
    
    # Stage 1
    st.subheader("Stage 1")
    area_1 = st.number_input("S1 Area (cm²)", value=AREA_LIST[0], format="%.2f", key="a1")
    p_u_1 = st.number_input("S1 Upstream (p_u, atm)", value=p_u_default, key="pu1")
    p_p_1 = st.number_input("S1 Permeate (p_p, atm)", value=p_p_default, key="pp1")
    
    # Stage 2
    st.subheader("Stage 2")
    area_2 = st.number_input("S2 Area (cm²)", value=AREA_LIST[1], format="%.2f", key="a2")
    p_u_2 = st.number_input("S2 Upstream (p_u, atm)", value=p_u_default, key="pu2")
    p_p_2 = st.number_input("S2 Permeate (p_p, atm)", value=p_p_default, key="pp2")

    # Stage 3
    st.subheader("Stage 3")
    area_3 = st.number_input("S3 Area (cm²)", value=AREA_LIST[2], format="%.2f", key="a3")
    p_u_3 = st.number_input("S3 Upstream (p_u, atm)", value=p_u_default, key="pu3")
    p_p_3 = st.number_input("S3 Permeate (p_p, atm)", value=p_p_default, key="pp3")

    # Stage 4
    st.subheader("Stage 4")
    area_4 = st.number_input("S4 Area (cm²)", value=AREA_LIST[3], format="%.2f", key="a4")
    p_u_4 = st.number_input("S4 Upstream (p_u, atm)", value=p_u_default, key="pu4")
    p_p_4 = st.number_input("S4 Permeate (p_p, atm)", value=p_p_default, key="pp4")
    # ----------------------------------------------------

    run_button = st.button("🚀 시뮬레이션 실행")

# --- 메인 화면 (결과 표시) ---
if run_button:
    main_area = st.container()
    
    try:
        # --- 1. 입력값 파싱 ---
        main_area.subheader("1. 입력값 파싱 중...")
        
        area_list_in = [area_1, area_2, area_3, area_4]
        
        p_in = np.array([p_1, p_2, p_3])
        raw_feed_comp_in = np.array([comp_1, comp_2, comp_3])
        comp_names_in = COMP_NAMES_FIXED 
        
        if len(p_in) != len(raw_feed_comp_in):
             st.error(f"오류: 막 투과도 P 갯수({len(p_in)})와 초기 조성 갯수({len(raw_feed_comp_in)})가 일치하지 않습니다.")
             st.stop()
        
        if len(comp_names_in) != len(raw_feed_comp_in):
             st.error(f"오류: 고정된 성분 이름 갯수({len(comp_names_in)})와 초기 조성 갯수({len(raw_feed_comp_in)})가 일치하지 않습니다.")
             st.stop()
             
        if not np.isclose(np.sum(raw_feed_comp_in), 1.0):
            st.warning(f"초기 조성의 합이 1이 아닙니다 (Sum={np.sum(raw_feed_comp_in):.4f}). 시뮬레이션 내부에서 정규화(normalize)하여 계산합니다.")
        
        main_area.success("입력값 파싱 완료.")

        # --- 2. 파라미터 준비 ---
        # [MODIFIED] 스테이지별 파라미터 리스트 생성
        
        process_params_list_mol = []
        p_u_list = [p_u_1, p_u_2, p_u_3, p_u_4]
        p_p_list = [p_p_1, p_p_2, p_p_3, p_p_4]
        
        # 공통 파라미터 (P, delta)
        p_mol = p_in / STP_MOLAR_VOLUME
        delta_in = delta # 사이드바에서 입력받은 delta 값
        
        for i in range(4): # 4-stage 기준
            stage_params = {
                "P": p_mol,
                "delta": delta_in,
                "p_u": p_u_list[i],  # 해당 스테이지의 p_u
                "p_p": p_p_list[i],  # 해당 스테이지의 p_p
            }
            process_params_list_mol.append(stage_params)
        
        raw_feed_flux_mol = feed_flux_vol / STP_MOLAR_VOLUME

        # --- 3. 시뮬레이션 실행 ---
        main_area.subheader("2. ⚙️ 시뮬레이션 실행 (재활용 루프)")
        
        # [MODIFIED] Process 객체에 params_list 전달
        membrane_process = Process(process_params_list_mol, area_list_in, stp_molar_volume=STP_MOLAR_VOLUME)
        
        success = membrane_process.run_with_recycle(
            raw_feed_flux=raw_feed_flux_mol,
            raw_feed_comp=raw_feed_comp_in
        )

        # --- 4. 최종 결과 표시 (테이블 형식) ---
        if success:
            main_area.subheader("3. 🏁 최종 수렴 결과")
            vol_conv = membrane_process.stp_molar_volume 
            
            results_data = [] 
            
            for stage in membrane_process.stages:
                # [NEW] 결과 테이블에 p_u, p_p 값 추가
                # (stage 객체는 params를 저장하지 않으므로, process 객체에서 가져와야 함)
                stage_idx = int(stage.name.split('-')[-1]) - 1 # Stage-1 -> 0
                stage_params = membrane_process.params_list[stage_idx]
                
                stage_data = {
                    "Stage": stage.name,
                    "Area (cm²)": stage.area,
                    "p_u (atm)": stage_params['p_u'], # [NEW]
                    "p_p (atm)": stage_params['p_p'], # [NEW]
                    "Stage Cut (θ)": stage.stage_cut,
                    "Feed Flux (cm³/s)": stage.feed_flux * vol_conv,
                }
                
                for i, name in enumerate(comp_names_in):
                    stage_data[f"Feed {name}"] = stage.feed_comp[i]
                
                stage_data["Permeate Flux (cm³/s)"] = stage.permeate_flux * vol_conv
                for i, name in enumerate(comp_names_in):
                    stage_data[f"Permeate {name}"] = stage.permeate_comp[i]

                stage_data["Retentate Flux (cm³/s)"] = stage.retentate_flux * vol_conv
                for i, name in enumerate(comp_names_in):
                    stage_data[f"Retentate {name}"] = stage.retentate_comp[i]
                
                results_data.append(stage_data)
            
            df = pd.DataFrame(results_data)
            
            formatters = {
                "Area (cm²)": "{:.2f}",
                "p_u (atm)": "{:.2f}", # [NEW]
                "p_p (atm)": "{:.2f}", # [NEW]
                "Stage Cut (θ)": "{:.4f}",
                "Feed Flux (cm³/s)": "{:.2f}",
                "Permeate Flux (cm³/s)": "{:.2f}",
                "Retentate Flux (cm³/s)": "{:.2f}",
            }
            for name in comp_names_in:
                formatters[f"Feed {name}"] = "{:.4f}"
                formatters[f"Permeate {name}"] = "{:.4f}"
                formatters[f"Retentate {name}"] = "{:.4f}" 

            main_area.dataframe(df.style.format(formatters), use_container_width=True)

        else:
            main_area.error("시뮬레이션이 수렴에 실패했습니다.")

    except Exception as e:
        st.error(f"스크립트 실행 중 오류가 발생했습니다:")
        st.exception(e)
        
