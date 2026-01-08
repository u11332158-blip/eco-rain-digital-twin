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
        "theory_header": "Physics Logic & Models",
        "lab_ctrl": "Parameter Control",
        "lab_env": "Experiment A: Ghost Damping Effect",
        "lab_freq_sect": "Experiment B: Solenoid Hardware Limit",
        "lab_sweet_spot": "Set to Solenoid Limit",
        "lab_monitor": "Physics Monitor",
        "rain_rate": "Rain Rate (mm/hr)",
        "impact_freq": "Impact Freq (Hz)",
        "field_header": "Real-world Scenario Simulation",
        "sim_params": "Simulation Parameters",
        "sim_duration": "Duration (Hours)",
        "view_weather": "View Weather Data",
        "upload_csv": "Upload Weather CSV",
        "use_sim": "Using Generated Simulation Data",
        "use_csv": "Using Uploaded CSV Data",
        "metric_fixed": "Fixed System Output",
        "metric_smart": "Smart System Output",
        "metric_eroi": "EROI (Return)",
        "chart_cum_title": "Cumulative Energy Generation",
        "unit_energy": "mJ",
        "sim_start_btn": "Run Monte Carlo Sim",
        "sim_success": "Generated {n} drops data.",
        "solenoid_eff": "Solenoid Force Efficiency"
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
        "theory_header": "系統運算邏輯與物理模型",
        "lab_ctrl": "變因控制實驗",
        "lab_env": "實驗 A：水膜阻尼效應 (Ghost Damping)",
        "lab_freq_sect": "實驗 B：電磁閥物理限制 (Solenoid Limit)",
        "lab_sweet_spot": "設定為電磁閥極限",
        "lab_monitor": "物理參數監控",
        "rain_rate": "降雨強度 (mm/hr)",
        "impact_freq": "撞擊頻率 (Hz)",
        "field_header": "真實情境模擬",
        "sim_params": "模擬參數",
        "sim_duration": "模擬時長 (小時)",
        "view_weather": "查看氣象數據",
        "upload_csv": "上傳氣象 CSV 檔",
        "use_sim": "使用系統生成模擬數據",
        "use_csv": "使用上傳的 CSV 數據",
        "metric_fixed": "固定式總產出",
        "metric_smart": "智慧式總產出",
        "metric_eroi": "EROI (能源投報率)",
        "chart_cum_title": "累積發電量模擬",
        "unit_energy": "mJ",
        "sim_start_btn": "執行蒙地卡羅模擬",
        "sim_success": "成功生成 {n} 顆雨滴數據。",
        "solenoid_eff": "電磁閥力道效率"
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
        "theory_header": "物理ロジックとモデル",
        "lab_ctrl": "パラメータ制御",
        "lab_env": "実験 A：水膜減衰効果",
        "lab_freq_sect": "実験 B：ソレノイド物理限界",
        "lab_sweet_spot": "ソレノイド限界設定",
        "lab_monitor": "物理パラメータ",
        "rain_rate": "降雨強度 (mm/hr)",
        "impact_freq": "衝突周波数 (Hz)",
        "field_header": "実環境シミュレーション",
        "sim_params": "シミュレーションパラメータ",
        "sim_duration": "時間 (Hours)",
        "view_weather": "気象データ表示",
        "upload_csv": "CSVアップロード",
        "use_sim": "シミュレーションデータを使用",
        "use_csv": "アップロードデータを使用",
        "metric_fixed": "固定式システム出力",
        "metric_smart": "スマートシステム出力",
        "metric_eroi": "EROI (エネルギー収支)",
        "chart_cum_title": "累積発電量シミュレーション",
        "unit_energy": "mJ",
        "sim_start_btn": "モンテカルロ法を実行",
        "sim_success": "{n} 個の雨滴データを生成しました。",
        "solenoid_eff": "ソレノイド効率"
    }
}

# ==========================================
# 2. 物理常數定義區 (Physical Config)
# ==========================================
class PhysConfig:
    PIEZO_SENSITIVITY_V_PM = 50000.0  
    IMPACT_DURATION_SEC = 0.002       
    
    # 校正後參數 (33.3Hz Sweet Spot)
    DAMPING_RATIO_DRY = 0.04         
    DAMPING_COEFF_WET = 0.35          
    
    SATURATION_RAIN_RATE = 120.0      
    SMART_SYSTEM_WETNESS_RATIO = 0.2  
    BASE_POWER_FACTOR = 0.5           
    TRUNCATION_SHAPE_FACTOR = 0.6     

# ==========================================
# 3. 核心物理運算區 (Physics Core)
# ==========================================
class PhysicsEngine:
    def __init__(self, area=2.5, fn=100, length=5.0):
        self.area = area
        self.fn = fn
        self.length = length

    def get_params(self, rain, wind, mode="Fixed", freq_override=None):
        if rain <= 0: return 0, 0.04, 0, 0, 0, 0, 1.0, 0.0 
        D0 = 0.9 * (rain ** 0.21) 
        V_term = 3.778 * (D0 ** 0.67) 
        
        # 頻率估算
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

# --- CSS ---
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

# --- 語言 ---
st.sidebar.markdown("### Language / 語言 / 言語")
selected_lang = st.sidebar.selectbox("Select Language", ["English", "繁體中文", "日本語"], label_visibility="collapsed")
t = TRANSLATIONS[selected_lang] 

# --- 側邊欄 ---
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

# ================= TAB 1: 理論架構 (Theory & Logic - V5.1 Updated) =================
with tab_theory:
    st.header(t["theory_header"])
    st.caption("Governing Equations of the Digital Twin: Bridging Lab & Nature")
    
    st.markdown("---")

    # --- Part 1: 環境物理模型 (Environmental Physics) ---
    st.subheader("1. Environmental Input Models (Nature's Physics)")
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown(f"""
        <div class="theory-box">
        <h4>Eq. 1: Stochastic Rain (Marshall-Palmer)</h4>
        <p>Models the random distribution of raindrop sizes in a storm.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"N(D) = N_0 e^{-\Lambda D}")
        
        st.markdown(f"""
        <div class="theory-box">
        <h4>Eq. 2: Terminal Velocity (Gunn-Kinzer)</h4>
        <p>Corrects impact momentum for air resistance.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"V_{term}(D) = 9.65 - 10.3 e^{-0.6D}")

    with col_t2:
        st.markdown(f"""
        <div class="theory-box">
        <h4>Eq. 3: Effective Impact Angle</h4>
        <p>Vector analysis of wind speed ($V_w$) and rain velocity ($V_t$).</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\theta_{eff} = \arctan\left(\frac{V_{wind}}{V_{term}}\right)")

    st.markdown("---")

    # --- Part 2: 系統動力模型 (System Dynamics) ---
    st.subheader("2. System Dynamics & Constraints (Hardware vs. Nature)")
    col_t3, col_t4 = st.columns(2)
    
    with col_t3:
        st.markdown(f"""
        <div class="theory-box">
        <h4>Eq. 4: Piezo-Dynamics & Moment Arm</h4>
        <p>2nd-order mass-spring-damper system with position scaling.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"m_{eff} \ddot{x} + c \dot{x} + k x = F(t) \cdot \left(\frac{x_{pos}}{L}\right)^2")

        st.markdown(f"""
        <div class="theory-box">
        <h4>Eq. 5: Ghost Damping (Water Film)</h4>
        <p><b>Nature's Limit:</b> Damping spikes as water film ($h$) accumulates.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\zeta(t) = \zeta_{dry} + \kappa \cdot h_{film}(t)")
        st.caption("Explains why high-freq fails in nature ($\zeta \to 0.35$).")

    with col_t4:
        st.markdown(f"""
        <div class="theory-box">
        <h4>Eq. 6: Solenoid Inductance Limit</h4>
        <p><b>Lab's Limit:</b> Force decays due to magnetic lag at high freq.</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"F_{eff}(f) = F_{max} \cdot \left(\frac{33.3}{f}\right)^{1.5}")
        st.caption("Explains why high-freq fails in the lab ($F \to 0$).")

    # APA References
    st.markdown("---")
    st.markdown("### 📚 References (IEEE Standard)")
    st.markdown("""
    <div class="citation-box">
    <p><b>[1]</b> Marshall, J. S., & Palmer, W. M. (1948). The distribution of raindrops with size. <i>Journal of meteorology</i>, <i>5</i>(4), 165-166.</p>
    <p><b>[2]</b> Gunn, R., & Kinzer, G. D. (1949). The terminal velocity of fall for water droplets in stagnant air. <i>Journal of meteorology</i>, <i>6</i>(4), 243-248.</p>
    <p><b>[3]</b> Li, S., Crovetto, A., et al. (2016). Bi-resonant structure with piezoelectric PVDF films. <i>Sensors and Actuators A</i>.</p>
    </div>
    """, unsafe_allow_html=True)
# ================= TAB 2: 物理實驗室 (Core Update) =================
with tab_lab:
    st.markdown(f"#### {t['lab_ctrl']}")
    st.caption("Experiments isolating Ghost Damping (Physics) and Solenoid Limits (Hardware).")
    st.markdown("---")

    # === 實驗 A: 水膜阻尼 (Ghost Damping) ===
    st.markdown(f"##### {t['lab_env']}")
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        val_rain_a = st.slider(f"{t['rain_rate']}", 0, 150, 150, key="exp_a_rain")
        FIXED_FREQ_A = 5 
        _, z_f, eff_f, _, wd, _, _, _ = engine.get_params(val_rain_a, 0, "Fixed", freq_override=FIXED_FREQ_A)
        _, z_s, eff_s, _, _, _, _, _  = engine.get_params(val_rain_a, 0, "Smart", freq_override=FIXED_FREQ_A)
        st.info(f"""
        **Zeta Comparison:**
        * **Fixed (Red):** `{z_f:.4f}`
        * **Smart (Green):** `{z_s:.4f}`
        """)

    with col_a2:
        VIEW_WINDOW_A = 0.2
        t_arr = np.linspace(0, VIEW_WINDOW_A, 2000)
        wave_f = (1.0 * eff_f) * np.exp(-z_f * 2 * np.pi * param_fn * t_arr) * np.sin(wd * t_arr)
        wave_s = (1.0 * eff_s) * np.exp(-z_s * 2 * np.pi * param_fn * t_arr) * np.sin(wd * t_arr)
        
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=t_arr*1000, y=wave_s, mode='lines', name='Smart (Active)', line=dict(color='#2e7d32', width=3)))
        fig_a.add_trace(go.Scatter(x=t_arr*1000, y=wave_f, mode='lines', name='Fixed (Passive)', line=dict(color='#c62828', width=2, dash='dot')))
        fig_a.update_layout(title=f"Fig 2: Failure Mode (Rain: {val_rain_a} mm/hr)", height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("---")

    # === 實驗 B: 電磁閥物理限制 (Solenoid Limit) ===
    # 核心邏輯修正：頻率越高 -> 力道(振幅)越弱
    st.markdown(f"##### {t['lab_freq_sect']}")
    
    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        optimal_freq = 33.3
        
        if st.button(f"{t['lab_sweet_spot']} ({optimal_freq:.1f} Hz)", key="btn_sweet"):
            st.session_state['exp_b_freq'] = int(optimal_freq)
        
        if 'exp_b_freq' not in st.session_state: st.session_state['exp_b_freq'] = 30
        val_freq_b = st.slider(f"{t['impact_freq']}", 5, 120, 
                             value=st.session_state['exp_b_freq'], key="exp_b_slider")
        st.session_state['exp_b_freq'] = val_freq_b

        # === 核心算法：電磁閥效率模型 ===
        # 超過 33.3 Hz 後，力道按 (33.3/f)^1.5 衰減
        if val_freq_b <= optimal_freq:
            solenoid_eff = 1.0
            status_color, status_text = "#fbc02d", "✅ Full Force (Sweet Spot)"
        else:
            solenoid_eff = (optimal_freq / val_freq_b) ** 1.5
            status_color, status_text = "#d32f2f", "⚠️ Force Drop (Valve Lag)"

        st.metric(label=t["solenoid_eff"], value=f"{solenoid_eff*100:.1f}%", delta=f"Loss: -{(1-solenoid_eff)*100:.1f}%" if solenoid_eff < 1 else "Max Power")
        st.markdown(f"""<div style="padding:10px; border-left:5px solid {status_color}; background:{status_color}10;"><b>Status:</b> {status_text}</div>""", unsafe_allow_html=True)
        
        FIXED_RAIN_B = 50 
        _, z_s, eff_s, tau_s, wd, _, _, _ = engine.get_params(FIXED_RAIN_B, 0, "Smart")

    with col_b2:
        # 繪圖：顯示 "理想最大力道" vs "實際衰減後力道"
        # 觀察 120Hz 時，綠線會變得很矮
        FULL_DECAY_TIME = tau_s * 3 
        t_arr_b = np.linspace(0, FULL_DECAY_TIME, 1000)
        
        # 1. 幽靈波形 (理想最大力道 - 假設電磁閥無敵)
        wave_ghost = (1.0 * eff_s) * np.exp(-z_s * 2 * np.pi * param_fn * t_arr_b) * np.sin(wd * t_arr_b)
        
        # 2. 實際波形 (受電磁閥物理限制，振幅縮水)
        # 同時也受到時間截斷 (Truncation) 的影響
        T_impact_limit = 1 / val_freq_b
        wave_actual = wave_ghost * solenoid_eff # <--- 關鍵修正：乘上效率係數
        
        # 視覺化：超過時間變 NaN
        wave_viz = np.where(t_arr_b <= T_impact_limit, wave_actual, None)

        fig_b = go.Figure()
        
        # 灰色虛線：原本可以打出的力道 (Potential)
        fig_b.add_trace(go.Scatter(x=t_arr_b*1000, y=wave_ghost, mode='lines', name='Ideal Impact Force', 
                                  line=dict(color='gray', width=2, dash='dot'), opacity=0.5))
        
        # 綠色實線：實際打出的力道 (弱化版)
        fig_b.add_trace(go.Scatter(x=t_arr_b*1000, y=wave_viz, mode='lines', name='Actual Impact (Weakened)', 
                                  line=dict(color='#2e7d32', width=3), fill='tozeroy'))
        
        # 切斷點
        fig_b.add_vline(x=T_impact_limit*1000, line_dash="solid", line_color="#d32f2f", opacity=0.8, 
                       annotation_text="Next Hit", annotation_position="top right")
        
        fig_b.update_layout(
            title=f"Fig 3: Solenoid Physics @ {val_freq_b} Hz (Force Eff: {solenoid_eff*100:.0f}%)",
            xaxis_title="Time (ms)", yaxis_title="Voltage (V)", height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(range=[-1.2, 1.2]) # 固定 Y 軸，更能看出振幅變矮
        )
        st.plotly_chart(fig_b, use_container_width=True)

# ================= TAB 3: 場域模擬 =================
with tab_field:
    st.markdown(f"#### {t['field_header']}")
    st.markdown("##### 1. Long-term Rainfall Simulation")
    col_input, col_sim = st.columns([1, 3])
    
    with col_input:
        uploaded_file = st.file_uploader(t["upload_csv"], type=["csv"])
        if uploaded_file is not None:
            st.success(t["use_csv"])
            df = pd.read_csv(uploaded_file)
            if not {'Time', 'Rain', 'Wind'}.issubset(df.columns):
                st.error("CSV must contain: Time, Rain, Wind")
                df = None
        else:
            st.info(t["use_sim"])
            sim_duration = st.slider(t["sim_duration"], 1, 24, 12)
            h = np.arange(0, sim_duration + 1, 1) 
            peak_time = sim_duration / 2
            r = 10 + 100 * np.exp(-0.5 * (h - peak_time)**2/2.5) 
            w = 5 + 25 * np.exp(-0.5 * (h - peak_time)**2/3) + np.random.normal(0, 2, len(h))
            df = pd.DataFrame({'Time': h, 'Rain': np.clip(r, 0, None), 'Wind': np.clip(w, 0, None)})
        
        if df is not None:
            with st.expander(t["view_weather"]):
                st.dataframe(df, height=150)

    with col_sim:
        if df is not None:
            acc_s_list, acc_f_list = [], []
            cum_s, cum_f = 0, 0
            
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
            
            gain = ((cum_s - cum_f) / cum_f) * 100 if cum_f > 0 else 0
            eroi = cum_s / (cum_s * (drainage_cost_pct/100)) if cum_s > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric(t["metric_fixed"], f"{int(cum_f):,} {t['unit_energy']}", "Baseline")
            m2.metric(t["metric_smart"], f"{int(cum_s):,} {t['unit_energy']}", f"+{gain:.1f}%")
            m3.metric(t["metric_eroi"], f"{eroi:.1f}", f"Cost: {drainage_cost_pct}%")
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Time'], y=acc_s_list, fill='tozeroy', name='Smart', line=dict(color='#2e7d32')))
            fig2.add_trace(go.Scatter(x=df['Time'], y=acc_f_list, fill='tozeroy', name='Fixed', line=dict(color='#c62828')))
            fig2.update_layout(title=t["chart_cum_title"], height=350, margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 2. Monte Carlo Validation")
    col_ui1, col_ui2 = st.columns(2)
    mc_rain = col_ui1.slider(f"{t['rain_rate']}", 10, 100, 50, key="mc_rain")
    mc_wet = col_ui2.slider("Wetness Factor", 0.0, 1.0, 0.1)

    if st.button(t["sim_start_btn"]):
        masses, velocities = generate_storm_profile(n_drops=1000, rain_rate_mmph=mc_rain)
        st.success(t["sim_success"].format(n=len(masses)))
        c1, c2 = st.columns(2)
        with c1:
            fig_mc1, ax = plt.subplots(figsize=(5, 4))
            ax.hist(velocities, bins=25, color='#4A90E2', alpha=0.7)
            st.pyplot(fig_mc1)
        with c2:
            idx = np.random.randint(0, len(masses))
            t_rk, v_rk = rk4_solver(0.005, 150, 0.0001, 0.1, masses[idx], velocities[idx], mc_wet)
            fig_mc2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(t_rk*1000, v_rk, color='#FF6B6B')
            st.pyplot(fig_mc2)

