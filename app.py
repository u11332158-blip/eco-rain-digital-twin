import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. 多語言字典 (Translation Dictionary)
# ==========================================
TRANSLATIONS = {
    "English": {
        "title": "Eco-Rain: Digital Twin Platform",
        "sidebar_settings": "Global Settings",
        "target_material": "Target Material",
        "beam_len": "Beam Length L (cm)",
        "area": "Sensor Area (cm2)",
        "freq": "Resonance Freq (Hz)",
        "drainage_cost": "Drainage Energy Cost (%)",
        "dev_credit": "Developed for Science Edge 2026",
        "tab_theory": "Theory & Logic",
        "tab_lab": "Physics Lab",
        "tab_field": "Field Simulation",
        # Theory Tab
        "theory_header": "Physics Logic & Models",
        "theory_1_title": "1. Stochastic Input",
        "theory_1_desc": "Marshall-Palmer Distribution for drop size.",
        "theory_2_title": "2. Piezo Dynamics",
        "theory_2_desc": "2nd-order Damping System.",
        "theory_3_title": "3. Geometry & Mechanics",
        "theory_3_vec": "Effective Impact Angle",
        "theory_3_mom": "Moment Arm Effect",
        # Lab Tab
        "lab_ctrl": "Parameter Control",
        "lab_env": "Experiment A: Ghost Damping Effect",
        "lab_freq_sect": "Experiment B: Frequency Optimization",
        "lab_sweet_spot": "Set to Sweet Spot",
        "lab_monitor": "Physics Monitor",
        "lab_monitor_zeta": "Zeta (Damping)",
        "rain_rate": "Rain Rate (mm/hr)",
        "wind_speed": "Wind Speed (m/s)",
        "impact_freq": "Impact Freq (Hz)",
        # Field Tab
        "field_header": "Real-world Scenario Simulation",
        "sim_params": "Simulation Parameters",
        "sim_duration": "Duration (Hours)",
        "view_weather": "View Weather Data",
        "metric_fixed": "Fixed System Output",
        "metric_smart": "Smart System Output",
        "metric_eroi": "EROI (Return)",
        "chart_cum_title": "Cumulative Energy Generation",
        "chart_pos_title": "Impact Position Distribution",
        "unit_energy": "mJ",
        "sim_start_btn": "Run Monte Carlo Sim",
        "sim_success": "Generated {n} drops data."
    },
    "繁體中文": {
        "title": "Eco-Rain: 壓電雨能採集數位孿生",
        "sidebar_settings": "全域參數設定",
        "target_material": "目標材料模型",
        "beam_len": "懸臂樑長度 L (cm)",
        "area": "感測器有效面積 (cm2)",
        "freq": "裝置共振頻率 (Hz)",
        "drainage_cost": "主動排水耗能係數 (%)",
        "dev_credit": "為 Tsukuba Science Edge 2026 開發",
        "tab_theory": "理論架構",
        "tab_lab": "物理實驗室",
        "tab_field": "場域模擬",
        # Theory Tab
        "theory_header": "系統運算邏輯與物理模型",
        "theory_1_title": "1. 氣象輸入模型",
        "theory_1_desc": "採用 Marshall-Palmer 分佈模擬雨滴。",
        "theory_2_title": "2. 壓電動力學",
        "theory_2_desc": "二階阻尼彈簧-質量系統建模。",
        "theory_3_title": "3. 幾何追蹤與力學",
        "theory_3_vec": "有效撞擊角度模型",
        "theory_3_mom": "力臂效應",
        # Lab Tab
        "lab_ctrl": "變因控制實驗",
        "lab_env": "實驗 A：水膜阻尼效應 (Ghost Damping)",
        "lab_freq_sect": "實驗 B：頻率優化 (Frequency Optimization)",
        "lab_sweet_spot": "設定為甜蜜點頻率",
        "lab_monitor": "物理參數監控",
        "lab_monitor_zeta": "阻尼比 (Zeta)",
        "rain_rate": "降雨強度 (mm/hr)",
        "wind_speed": "環境風速 (m/s)",
        "impact_freq": "撞擊頻率 (Hz)",
        # Field Tab
        "field_header": "真實情境模擬",
        "sim_params": "模擬參數",
        "sim_duration": "模擬時長 (小時)",
        "view_weather": "查看氣象數據",
        "metric_fixed": "固定式總產出",
        "metric_smart": "智慧式總產出",
        "metric_eroi": "EROI (能源投報率)",
        "chart_cum_title": "累積發電量模擬",
        "chart_pos_title": "雨滴落點分佈與力臂分析",
        "unit_energy": "mJ",
        "sim_start_btn": "執行蒙地卡羅模擬",
        "sim_success": "成功生成 {n} 顆雨滴數據。"
    },
    "日本語": {
        "title": "Eco-Rain: 雨滴発電デジタルツイン",
        "sidebar_settings": "グローバル設定",
        "target_material": "ターゲット材料",
        "beam_len": "カンチレバー長さ L (cm)",
        "area": "センサー有効面積 (cm2)",
        "freq": "共振周波数 (Hz)",
        "drainage_cost": "排水エネルギーコスト (%)",
        "dev_credit": "Tsukuba Science Edge 2026 向け開発",
        "tab_theory": "理論とロジック",
        "tab_lab": "物理実験室",
        "tab_field": "フィールド・シミュレーション",
        # Theory Tab
        "theory_header": "物理ロジックとモデル",
        "theory_1_title": "1. 気象入力モデル",
        "theory_1_desc": "Marshall-Palmer分布を採用。",
        "theory_2_title": "2. 圧電ダイナミクス",
        "theory_2_desc": "二次減衰バネ-質量系モデル。",
        "theory_3_title": "3. 幾何追跡と力学",
        "theory_3_vec": "有効衝突角度モデル",
        "theory_3_mom": "モーメントアーム効果",
        # Lab Tab
        "lab_ctrl": "パラメータ制御",
        "lab_env": "実験 A：水膜減衰効果",
        "lab_freq_sect": "実験 B：周波数最適化",
        "lab_sweet_spot": "スイートスポット設定",
        "lab_monitor": "物理パラメータ",
        "lab_monitor_zeta": "減衰比 (Zeta)",
        "rain_rate": "降雨強度 (mm/hr)",
        "wind_speed": "風速 (m/s)",
        "impact_freq": "衝突周波数 (Hz)",
        # Field Tab
        "field_header": "実環境シミュレーション",
        "sim_params": "シミュレーションパラメータ",
        "sim_duration": "時間 (Hours)",
        "view_weather": "気象データ表示",
        "metric_fixed": "固定式システム出力",
        "metric_smart": "スマートシステム出力",
        "metric_eroi": "EROI (エネルギー収支)",
        "chart_cum_title": "累積発電量シミュレーション",
        "chart_pos_title": "雨滴衝突位置分布",
        "unit_energy": "mJ",
        "sim_start_btn": "モンテカルロ法を実行",
        "sim_success": "{n} 個の雨滴データを生成しました。"
    }
}

# ==========================================
# 2. 物理常數定義區 (Physical Config)
# ==========================================
class PhysConfig:
    PIEZO_SENSITIVITY_V_PM = 50000.0  
    IMPACT_DURATION_SEC = 0.002       
    
    # [校正點 1] 將 0.008 改為 0.04
    # 目的：讓波形在 33.3Hz 時剛好呈現 Full Decay，符合論文數據
    DAMPING_RATIO_DRY = 0.04         
    
    # [校正點 2] 保持 0.35 以模擬水膜重阻尼
    DAMPING_COEFF_WET = 0.35          
    
    SATURATION_RAIN_RATE = 120.0      
    SMART_SYSTEM_WETNESS_RATIO = 0.2  
    BASE_POWER_FACTOR = 0.5           
    TRUNCATION_SHAPE_FACTOR = 0.6     

# ==========================================
# 3. 核心物理運算區 (Physics Core)
# ==========================================
def generate_storm_profile(n_drops=1000, rain_rate_mmph=50):
    lam = 4.1 * (rain_rate_mmph ** -0.21)
    u = np.random.uniform(0, 1, n_drops)
    diameters_mm = -np.log(1 - u) / lam
    diameters_mm = np.clip(diameters_mm, 0.1, 6.0)
    velocities = 9.65 - 10.3 * np.exp(-0.6 * diameters_mm)
    velocities = np.clip(velocities, 0, None)
    masses_mg = (4/3) * np.pi * (diameters_mm / 2)**3
    return masses_mg, velocities

def rk4_solver(mass_beam, k_spring, dt, total_time, drop_mass, drop_velocity, wetness):
    wn = np.sqrt(k_spring / mass_beam)
    zeta = PhysConfig.DAMPING_RATIO_DRY + (PhysConfig.DAMPING_COEFF_WET * wetness)
    c_damp = 2 * zeta * mass_beam * wn
    state = np.array([0.0, 0.0])
    peak_force = (drop_mass * 1e-6 * drop_velocity) / (PhysConfig.IMPACT_DURATION_SEC / 2)
    t_steps = np.arange(0, total_time, dt)
    voltages = []
    
    def derivatives(t, y):
        x, v = y
        F_ext = 0
        if t < PhysConfig.IMPACT_DURATION_SEC:
            if t < PhysConfig.IMPACT_DURATION_SEC/2:
                F_ext = peak_force * (t / (PhysConfig.IMPACT_DURATION_SEC/2))
            else:
                F_ext = peak_force * (2 - t / (PhysConfig.IMPACT_DURATION_SEC/2))
        a = (F_ext - c_damp * v - k_spring * x) / mass_beam
        return np.array([v, a])

    for t in t_steps:
        k1 = derivatives(t, state)
        k2 = derivatives(t + dt/2, state + k1*dt/2)
        k3 = derivatives(t + dt/2, state + k2*dt/2)
        k4 = derivatives(t + dt, state + k3*dt)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        voltages.append(state[0] * PhysConfig.PIEZO_SENSITIVITY_V_PM) 
    return t_steps, np.array(voltages)

# ==========================================
# 4. 主程式 (Main App)
# ==========================================
st.set_page_config(page_title="Eco-Rain Digital Twin", layout="wide")

# --- CSS (Dark Mode Safe) ---
st.markdown("""
<style>
    .metric-card { background-color: #f5f5f5 !important; border: 1px solid #e0e0e0; border-radius: 5px; padding: 15px; border-left: 5px solid #2e7d32; margin-bottom: 10px; }
    .metric-card h4, .metric-card p, .metric-card span, .metric-card div { color: #000000 !important; }
    .theory-box { background-color: #ffffff !important; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .theory-box h4 { color: #1565c0 !important; font-weight: bold; margin-bottom: 10px; }
    .theory-box p, .theory-box li, .theory-box span, .theory-box div, .theory-box b { color: #212121 !important; font-size: 1.05em; line-height: 1.6; }
    .citation-box { background-color: #fff3e0 !important; padding: 15px; border-radius: 5px; border-left: 5px solid #ff9800; font-size: 0.9em; margin-top: 20px; }
    .citation-box p, .citation-box i, .citation-box b, .citation-box span { color: #333333 !important; }
</style>
""", unsafe_allow_html=True)

# --- 語言選擇 (Language Selector) ---
st.sidebar.markdown("### Language / 語言 / 言語")
selected_lang = st.sidebar.selectbox("Select Language", ["English", "繁體中文", "日本語"], label_visibility="collapsed")
t = TRANSLATIONS[selected_lang] 

class PhysicsEngine:
    def __init__(self, area=2.5, fn=100, length=5.0):
        self.area = area
        self.fn = fn
        self.length = length

    def get_params(self, rain, wind, mode="Fixed", freq_override=None):
        if rain <= 0: return 0, 0.04, 0, 0, 0, 0, 1.0, 0.0 
        D0 = 0.9 * (rain ** 0.21) 
        V_term = 3.778 * (D0 ** 0.67) 
        if freq_override is not None:
            freq_est = freq_override
        else:
            freq_est = (rain / 100.0) * 60.0 
            if freq_est < 1: freq_est = 1 
        wetness = min(1.0, rain / PhysConfig.SATURATION_RAIN_RATE)
        if mode == "Smart": wetness *= PhysConfig.SMART_SYSTEM_WETNESS_RATIO 
        zeta = PhysConfig.DAMPING_RATIO_DRY + (PhysConfig.DAMPING_COEFF_WET * wetness) 
        if mode == "Smart":
            eff_angle = 1.0 
        else:
            theta = np.arctan(wind / (V_term if V_term>0 else 1))
            eff_angle = max(0, np.cos(theta))
        wn = 2 * np.pi * self.fn
        tau = 1 / (zeta * wn)
        wd = wn * np.sqrt(1 - zeta**2)
        rand_loc = np.random.normal(loc=self.length*0.7, scale=self.length*0.2)
        rand_loc = np.clip(rand_loc, 0, self.length)
        pos_factor = (rand_loc / self.length) ** 2 
        return freq_est, zeta, eff_angle, tau, wd, V_term, pos_factor, rand_loc

# --- 側邊欄 UI ---
st.title(t["title"])
st.caption("Physics-Informed Digital Twin Platform")
st.sidebar.markdown(f"### {t['sidebar_settings']}")
st.sidebar.markdown(f"**{t['target_material']}:**")
st.sidebar.info("TE Connectivity LDT0-028K (PVDF)")

param_beam_len = st.sidebar.number_input(t["beam_len"], 3.0, 10.0, 5.0, step=0.5)
param_area = st.sidebar.number_input(t["area"], 0.5, 10.0, 2.5, format="%.1f")
param_fn = st.sidebar.number_input(t["freq"], 50, 200, 100, format="%d")
drainage_cost_pct = st.sidebar.slider(t["drainage_cost"], 1.0, 10.0, 5.0)

engine = PhysicsEngine(area=param_area, fn=param_fn, length=param_beam_len)

st.sidebar.markdown("---")
st.sidebar.text(t["dev_credit"])

# --- Tabs ---
tab_theory, tab_lab, tab_field = st.tabs([t["tab_theory"], t["tab_lab"], t["tab_field"]])

# ================= TAB 1: 理論 (文獻修正區) =================
with tab_theory:
    st.header(t["theory_header"])
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f"""
        <div class="theory-box">
        <h4>{t['theory_1_title']}</h4>
        <p>{t['theory_1_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"N(D) = N_0 e^{-\Lambda D}")
    with col_t2:
        st.markdown(f"""
        <div class="theory-box">
        <h4>{t['theory_2_title']}</h4>
        <p>{t['theory_2_desc']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"m_{\text{eff}} \ddot{x} + c \dot{x} + k x = F_{\text{impact}}(t)")
        st.latex(r"\zeta(t) = \zeta_{dry} + \kappa \cdot h_{film}(t)")

    st.subheader(t["theory_3_title"])
    col_t3, col_t4 = st.columns(2)
    with col_t3:
        st.markdown(f"""
        <div class="theory-box">
        <h4>{t['theory_3_vec']}</h4>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\theta_{\text{eff}} = \arctan\left(\frac{V_{\text{wind}}}{V_{\text{term}}}\right)")
    with col_t4:
        st.markdown(f"""
        <div class="theory-box">
        <h4>{t['theory_3_mom']}</h4>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"E_{gen} \propto \left(\frac{x_{impact}}{L}\right)^2")

    # APA References (Standard Numbered)
    st.markdown("---")
    st.markdown("### References (IEEE Standard)")
    st.markdown("""
    <div class="citation-box">
    <p><b>[1]</b> Marshall, J. S., & Palmer, W. M. (1948). The distribution of raindrops with size. <i>Journal of meteorology</i>, <i>5</i>(4), 165-166.</p>
    <p><b>[2]</b> Gunn, R., & Kinzer, G. D. (1949). The terminal velocity of fall for water droplets in stagnant air. <i>Journal of meteorology</i>, <i>6</i>(4), 243-248.</p>
    <p><b>[3]</b> Li, S., Crovetto, A., et al. (2016). Bi-resonant structure with piezoelectric PVDF films. <i>Sensors and Actuators A</i>.</p>
    <p><b>[4]</b> Gregorio, R., Jr., & Ueno, E. M. (1999). Effect of crystalline phase on PVDF properties. <i>Journal of Materials Science</i>.</p>
    <p><b>[5]</b> Yuk, J., Leem, A., Thomas, K., & Jung, S. (2025). Leaf-inspired rain-energy harvesting device. <i>Cornell University</i>.</p>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 2: 物理機制探討 (Independent Physics Labs) =================
with tab_lab:
    st.markdown(f"#### {t['lab_ctrl']}")
    st.caption("Here we conduct two independent experiments to isolate physical variables.")
    
    st.markdown("---")

    # =========================================================
    # 實驗 A: 水膜阻尼效應 (Ghost Damping)
    # 變因：降雨率 (Rain Rate) | 固定：頻率 (5 Hz - 慢速以利觀察)
    # =========================================================
    st.markdown(f"##### {t['lab_env']}")
    
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        # [控制區 A] 只調整雨量
        val_rain_a = st.slider(f"{t['rain_rate']}", 0, 150, 150, key="exp_a_rain") # 預設最大以顯示失效
        
        # 固定頻率為 5Hz，確保波形看得很清楚
        FIXED_FREQ_A = 5 
        
        # 計算參數
        _, z_f, eff_f, _, wd, _, _, _ = engine.get_params(val_rain_a, 0, "Fixed", freq_override=FIXED_FREQ_A)
        _, z_s, eff_s, _, _, _, _, _  = engine.get_params(val_rain_a, 0, "Smart", freq_override=FIXED_FREQ_A)
        
        st.info(f"""
        **Comparison:**
        * **Fixed Zeta (Red):** `{z_f:.4f}`
        * **Smart Zeta (Green):** `{z_s:.4f}`
        * *Frequency fixed at 5 Hz for visibility.*
        """)

    with col_a2:
        # [繪圖區 A] 紅綠對決
        VIEW_WINDOW_A = 0.2
        t_arr = np.linspace(0, VIEW_WINDOW_A, 2000)
        
        # 產生 5Hz 的波形
        wave_f = (1.0 * eff_f) * np.exp(-z_f * 2 * np.pi * param_fn * t_arr) * np.sin(wd * t_arr)
        wave_s = (1.0 * eff_s) * np.exp(-z_s * 2 * np.pi * param_fn * t_arr) * np.sin(wd * t_arr)
        
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=t_arr*1000, y=wave_s, mode='lines', name='Smart (Active)', line=dict(color='#2e7d32', width=3)))
        fig_a.add_trace(go.Scatter(x=t_arr*1000, y=wave_f, mode='lines', name='Fixed (Passive)', line=dict(color='#c62828', width=2, dash='dot')))
        
        fig_a.update_layout(
            title=f"Fig 2: Failure Mode Analysis (Rain: {val_rain_a} mm/hr)",
            xaxis_title="Time (ms)", yaxis_title="Voltage (V)", height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, 150]), yaxis=dict(range=[-1.2, 1.2])
        )
        st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("---")

    # =========================================================
    # 實驗 B: 頻率截斷效應 (Frequency Truncation)
    # 變因：撞擊頻率 (Freq) | 固定：降雨率 (50 mm/hr - 中等以利震盪)
    # =========================================================
    st.markdown(f"##### {t['lab_freq_sect']}")
    
    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        # [控制區 B] 只調整頻率
        # 鎖定論文甜蜜點
        optimal_freq = 33.3
        
        # 甜蜜點按鈕
        if st.button(f"{t['lab_sweet_spot']} ({optimal_freq:.1f} Hz)", key="btn_sweet"):
            st.session_state['exp_b_freq'] = int(optimal_freq)
        
        if 'exp_b_freq' not in st.session_state: st.session_state['exp_b_freq'] = 30
        val_freq_b = st.slider(f"{t['impact_freq']}", 5, 120, 
                             value=st.session_state['exp_b_freq'], key="exp_b_slider")
        st.session_state['exp_b_freq'] = val_freq_b

        # 固定雨量為 50，確保有足夠的震盪來觀察截斷
        FIXED_RAIN_B = 50 
        
        # 計算參數 (只關心 Smart 系統的表現)
        _, z_s, eff_s, tau_s, wd, _, _, _ = engine.get_params(FIXED_RAIN_B, 0, "Smart")
        
        # 狀態判定
        T_impact = 1 / val_freq_b
        deviation = abs(val_freq_b - optimal_freq)
        if deviation < 5.0:
            status_color, status_text = "#fbc02d", "SWEET SPOT"
        elif val_freq_b > optimal_freq:
            status_color, status_text = "#d32f2f", "Truncated (Too Fast)"
        else:
            status_color, status_text = "#1976d2", "Inefficient (Too Slow)"
            
        st.markdown(f"""<div style="padding:10px; border-left:5px solid {status_color}; background:{status_color}10;"><b>Status:</b> {status_text}</div>""", unsafe_allow_html=True)
        st.caption("*Rain Rate fixed at 50 mm/hr.*")

    with col_b2:
        # [繪圖區 B] 綠線 vs 幽靈線
        VIEW_WINDOW_B = 0.06 # 微觀視角
        t_arr_b = np.linspace(0, VIEW_WINDOW_B, 2000)
        T_cycle = 1 / val_freq_b
        time_in_cycle = t_arr_b % T_cycle
        
        # 實際波形 (被切斷)
        wave_actual = (1.0 * eff_s) * np.exp(-z_s * 2 * np.pi * param_fn * time_in_cycle) * np.sin(wd * time_in_cycle)
        # 幽靈波形 (理想完整衰減)
        wave_ghost = (1.0 * eff_s) * np.exp(-z_s * 2 * np.pi * param_fn * t_arr_b) * np.sin(wd * t_arr_b)
        
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=t_arr_b*1000, y=wave_ghost, mode='lines', name='Ideal Potential', line=dict(color='gray', width=2, dash='dot'), opacity=0.5))
        fig_b.add_trace(go.Scatter(x=t_arr_b*1000, y=wave_actual, mode='lines', name='Actual Response', line=dict(color='#2e7d32', width=3)))
        
        # 標示切斷點
        fig_b.add_vline(x=T_cycle*1000, line_dash="solid", line_color="white", opacity=0.5, annotation_text="Reset", annotation_position="top left")
        
        fig_b.update_layout(
            title=f"Fig 3: Frequency Analysis @ {val_freq_b} Hz",
            xaxis_title="Time (ms)", yaxis_title="Voltage (V)", height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, 50]), yaxis=dict(range=[-1.2, 1.2])
        )
        st.plotly_chart(fig_b, use_container_width=True)

# ================= TAB 3: 場域模擬 (Field Simulation / Monte Carlo) =================
with tab_field:
    st.markdown(f"#### {t['field_header']}")
    
    # --- 1. 時間序列模擬 ---
    st.markdown("##### 1. Long-term Rainfall Simulation")
    col_input, col_sim = st.columns([1, 3])
    
    with col_input:
        st.subheader(t["sim_params"])
        sim_duration = st.slider(t["sim_duration"], 1, 24, 12)
        
        h = np.arange(0, sim_duration + 1, 1) 
        peak_time = sim_duration / 2
        r = 10 + 100 * np.exp(-0.5 * (h - peak_time)**2/2.5) 
        w = 5 + 25 * np.exp(-0.5 * (h - peak_time)**2/3) + np.random.normal(0, 2, len(h))
        df = pd.DataFrame({'Time': h, 'Rain': np.clip(r, 0, None), 'Wind': np.clip(w, 0, None)})
        
        with st.expander(t["view_weather"]):
            st.dataframe(df, height=150)

    with col_sim:
        acc_s_list, acc_f_list = [], []
        cum_s, cum_f = 0, 0
        loc_history, eff_history = [], []
        
        for idx, row in df.iterrows():
            R, W = row['Rain'], row['Wind']
            f_s, z_s, eff_s, tau_s, _, _, pos_s, loc_s = engine.get_params(R, W, "Smart")
            trunc_s = 1 / (1 + PhysConfig.TRUNCATION_SHAPE_FACTOR * f_s * tau_s) 
            
            f_f, z_f, eff_f, tau_f, _, _, pos_f, loc_f = engine.get_params(R, W, "Fixed")
            trunc_f = 1 / (1 + PhysConfig.TRUNCATION_SHAPE_FACTOR * f_f * tau_f)
            
            energy_s_raw = f_s * (eff_s**2) * trunc_s * (R**0.5) * pos_s * PhysConfig.BASE_POWER_FACTOR
            drainage_loss = energy_s_raw * (drainage_cost_pct / 100.0)
            energy_s_net = energy_s_raw - drainage_loss
            energy_f = f_f * (eff_f**2) * trunc_f * (R**0.5) * pos_f * PhysConfig.BASE_POWER_FACTOR
            
            cum_s += energy_s_net
            cum_f += energy_f
            acc_s_list.append(cum_s)
            acc_f_list.append(cum_f)
            loc_history.append(loc_s)
            eff_history.append(pos_s)
            
        gain = ((cum_s - cum_f) / cum_f) * 100 if cum_f > 0 else 0
        eroi = cum_s / (cum_s * (drainage_cost_pct/100)) if cum_s > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric(t["metric_fixed"], f"{int(cum_f):,} {t['unit_energy']}", "Baseline")
        m2.metric(t["metric_smart"], f"{int(cum_s):,} {t['unit_energy']}", f"+{gain:.1f}%")
        m3.metric(t["metric_eroi"], f"{eroi:.1f}", f"Cost: {drainage_cost_pct}%")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_s_list, fill='tozeroy', name='Smart', line=dict(color='#2e7d32')))
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_f_list, fill='tozeroy', name='Fixed', line=dict(color='#c62828')))
        fig2.update_layout(title=t["chart_cum_title"], yaxis_title=f"Total Energy ({t['unit_energy']})", height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("---")

    # --- 2. 蒙地卡羅驗證 ---
    st.markdown("##### 2. Monte Carlo Stochastic Validation")
    st.caption("Testing Robustness with 1000 Random Drops.")
    
    col_ui1, col_ui2 = st.columns(2)
    mc_rain = col_ui1.slider(f"{t['rain_rate']} (Monte Carlo)", 10, 100, 50)
    mc_wet = col_ui2.slider("Wetness Factor (0=Dry, 1=Sat)", 0.0, 1.0, 0.1)

    if st.button(t["sim_start_btn"]):
        masses, velocities = generate_storm_profile(n_drops=1000, rain_rate_mmph=mc_rain)
        st.success(t["sim_success"].format(n=len(masses)))
        
        c1, c2 = st.columns(2)
        with c1:
            fig_mc1, ax = plt.subplots(figsize=(5, 4))
            ax.hist(velocities, bins=25, color='#4A90E2', alpha=0.7)
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Count")
            ax.set_title("Raindrop Velocity Distribution")
            st.pyplot(fig_mc1)
            
        with c2:
            idx = np.random.randint(0, len(masses))
            t_rk, v_rk = rk4_solver(0.005, 150, 0.0001, 0.1, masses[idx], velocities[idx], mc_wet)
            fig_mc2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(t_rk*1000, v_rk, color='#FF6B6B')
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Voltage (V)")
            ax2.set_title(f"Single Stochastic Drop Response")
            st.pyplot(fig_mc2)
