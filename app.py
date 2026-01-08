import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 多語言字典 (Translation Dictionary)
# ==========================================
TRANSLATIONS = {
    "English": {
        "title": "Eco-Rain: Digital Twin Platform",
        "sidebar_settings": "Global Settings",
        "target_material": "Target Material",
        "beam_len": "Beam Length L (cm)",
        "area": "Sensor Area (cm²)",
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
        "theory_3_title": "3. Geometry & Moment",
        "theory_3_vec": "Vector Analysis",
        "theory_3_mom": "Moment Arm Effect",
        # Lab Tab
        "lab_ctrl": "Parameter Control",
        "rain_rate": "Rain Rate (mm/hr)",
        "wind_speed": "Wind Speed (m/s)",
        "impact_freq": "Impact Freq (Hz)",
        "lab_analysis": "State Analysis",
        "lab_wave_title": "Micro-view: Damped Oscillation",
        "status_trunc": "Waveform Truncated",
        "status_full": "Full Decay",
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
        "area": "感測器有效面積 (cm²)",
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
        "theory_3_title": "3. 幾何追蹤與力臂",
        "theory_3_vec": "向量合成分析",
        "theory_3_mom": "力臂效應",
        # Lab Tab
        "lab_ctrl": "變因控制實驗",
        "rain_rate": "降雨強度 (mm/hr)",
        "wind_speed": "環境風速 (m/s)",
        "impact_freq": "撞擊頻率 (Hz)",
        "lab_analysis": "物理狀態分析",
        "lab_wave_title": "微觀視圖：阻尼震盪波形",
        "status_trunc": "波形截斷 (Truncated)",
        "status_full": "完整釋放 (Full Decay)",
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
        "area": "センサー有効面積 (cm²)",
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
        "theory_3_title": "3. 幾何追跡とモーメント",
        "theory_3_vec": "ベクトル解析",
        "theory_3_mom": "モーメントアーム効果",
        # Lab Tab
        "lab_ctrl": "パラメータ制御",
        "rain_rate": "降雨強度 (mm/hr)",
        "wind_speed": "風速 (m/s)",
        "impact_freq": "衝突周波数 (Hz)",
        "lab_analysis": "状態分析",
        "lab_wave_title": "ミクロ視点：減衰振動波形",
        "status_trunc": "波形切断 (Truncated)",
        "status_full": "完全減衰 (Full Decay)",
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
# 物理常數定義區 (Physical Config)
# ==========================================
class PhysConfig:
    PIEZO_SENSITIVITY_V_PM = 50000.0  
    IMPACT_DURATION_SEC = 0.002       
    DAMPING_RATIO_DRY = 0.008         
    DAMPING_COEFF_WET = 0.35
    SATURATION_RAIN_RATE = 120.0      
    SMART_SYSTEM_WETNESS_RATIO = 0.2  
    BASE_POWER_FACTOR = 0.5           
    TRUNCATION_SHAPE_FACTOR = 0.6     

# ==========================================
# 核心物理運算區 (Physics Core)
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
# 主程式 (Main App)
# ==========================================
st.set_page_config(page_title="Eco-Rain Digital Twin", page_icon="⛈️", layout="wide")

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
st.sidebar.markdown("### 🌐 Language / 語言 / 言語")
selected_lang = st.sidebar.selectbox("Select Language", ["English", "繁體中文", "日本語"], label_visibility="collapsed")
t = TRANSLATIONS[selected_lang] 

class PhysicsEngine:
    def __init__(self, area=2.5, fn=100, length=5.0):
        self.area = area
        self.fn = fn
        self.length = length

    def get_params(self, rain, wind, mode="Fixed", freq_override=None):
        if rain <= 0: return 0, 0.008, 0, 0, 0, 0, 1.0, 0.0
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
st.sidebar.error("DEBUG: 版本 V3.1 (修正文獻版)") # 更新了版本號，方便您確認

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
        st.latex(r"\zeta(t) = 0.045 + 0.275 \cdot W(t)")

    st.subheader(t["theory_3_title"])
    col_t3, col_t4 = st.columns(2)
    with col_t3:
        st.markdown(f"""
        <div class="theory-box">
        <h4>{t['theory_3_vec']}</h4>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\theta_{\text{impact}} = \arctan\left(\frac{V_{\text{wind}}}{V_{\text{term}}}\right)")
    with col_t4:
        st.markdown(f"""
        <div class="theory-box">
        <h4>{t['theory_3_mom']}</h4>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"E_{gen} \propto \left(\frac{x}{L}\right)^2")

    # APA References (已補回 [3])
    st.markdown("---")
    st.markdown("### 📚 References (APA)")
    st.markdown("""
    <div class="citation-box">
    <p><b>[1] Raindrop Physics:</b><br>
    Marshall, J. S., & Palmer, W. M. (1948). The distribution of raindrops with size. <i>Journal of meteorology</i>, <i>5</i>(4), 165-166.<br>
    Gunn, R., & Kinzer, G. D. (1949). The terminal velocity of fall for water droplets in stagnant air. <i>Journal of meteorology</i>, <i>6</i>(4), 243-248.</p>
    
    <p><b>[2] Piezoelectric Dynamics:</b><br>
    Li, S., Crovetto, A., et al. (2016). Bi-resonant structure with piezoelectric PVDF films. <i>Sensors and Actuators A</i>.<br>
    Gregorio, R., Jr., & Ueno, E. M. (1999). Effect of crystalline phase on PVDF properties. <i>Journal of Materials Science</i>.</p>
    
    <p><b>[3] Related Works & Inspiration:</b><br>
    Yuk, J., Leem, A., Thomas, K., & Jung, S. (2025). Leaf-inspired rain-energy harvesting device. <i>Biological and Environmental Engineering, Cornell University</i>.<br>
    Bowland, A., et al. (2010). New concepts in modeling damping in structures. <i>10th CCEE</i>.</p>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 2: 物理機制探討 (Pure Theory / Lab) =================
with tab_lab:
    st.markdown(f"#### {t['lab_ctrl']} (Theoretical Verification)")
    st.caption("在此模式下，我們固定變因來尋找「能量甜蜜點 (Energy Sweet Spot)」。")
    
    col_ctrl, col_viz = st.columns([1, 2])
    with col_ctrl:
        # --- A. 環境設定 ---
        st.markdown("##### A. 環境設定 (Environment)")
        # 預設把雨量調到 50 (中等雨量)
        val_rain = st.slider(f"{t['rain_rate']}", 0, 150, 50, key="lab_rain")
        
        _, z_f, eff_f, tau_f, wd, _, _, _ = engine.get_params(val_rain, 0, "Fixed")
        _, z_s, eff_s, tau_s, _, _, _, _  = engine.get_params(val_rain, 0, "Smart")

        # 計算理論上的最佳頻率 (讓週期 T 剛好等於 3 倍時間常數)
        optimal_period = tau_s * 3 
        optimal_freq = 1 / optimal_period if optimal_period > 0 else 30
        
        st.info(f"""
        **物理參數:**
        * **Smart Zeta:** `{z_s:.4f}`
        * **Relaxation Time (tau):** `{tau_s*1000:.1f} ms`
        * **理論最佳頻率:** `{optimal_freq:.1f} Hz`
        """)

        st.markdown("---")

        # --- B. 頻率與甜蜜點設定 ---
        st.markdown("##### B. 頻率優化 (Frequency Optimization)")
        
        # [功能] 一鍵設定到甜蜜點按鈕 (無 Emoji)
        if st.button(f"Set to Sweet Spot ({optimal_freq:.1f} Hz)"):
            st.session_state['lab_freq_val'] = int(optimal_freq)
        
        # 頻率滑桿
        if 'lab_freq_val' not in st.session_state: st.session_state['lab_freq_val'] = 30
        
        val_freq = st.slider(f"{t['impact_freq']}", 5, 120, 
                             value=st.session_state['lab_freq_val'], key="lab_freq_slider")
        st.session_state['lab_freq_val'] = val_freq

        # --- 狀態判斷邏輯 ---
        T_impact = 1 / val_freq
        ratio = T_impact / tau_s
        
        if ratio < 2.0:
            status_color = "#d32f2f" # 紅色
            status_text = "Waveform Truncated"
            status_desc = "撞擊太快，能量未釋放完即被切斷。"
        elif 2.0 <= ratio <= 4.0:
            status_color = "#fbc02d" # 金色 (甜蜜點)
            status_text = "SWEET SPOT (Optimal)"
            status_desc = "完美匹配！週期 T 接近 3tau，能量最大化。"
        else:
            status_color = "#1976d2" # 藍色
            status_text = "Interval Too Long (Inefficient)"
            status_desc = "雖然波形完整，但撞擊密度太低，總功率低。"
        
        st.markdown(f"""
        <div style="padding:15px; border:2px solid {status_color}; background-color: {status_color}10; border-radius:8px;">
            <h4 style="margin:0; color:{status_color};">{status_text}</h4>
            <small style="color:#333;">{status_desc}</small>
        </div>
        """, unsafe_allow_html=True)

    with col_viz:
        st.subheader(t["lab_wave_title"])
        
        # --- 繪圖設定 ---
        VIEW_WINDOW = 0.2 # 固定視窗 200ms
        t_arr = np.linspace(0, VIEW_WINDOW, 2000) 
        T_cycle = 1 / val_freq 
        
        # 模擬連續波形
        time_in_cycle = t_arr % T_cycle
        
        # 只畫 Smart System (綠線)，因為這是我們探討頻率優化的主角
        wave_s = (1.0 * eff_s) * np.exp(-z_s * 2 * np.pi * param_fn * time_in_cycle) * \
                 np.sin(wd * time_in_cycle)
        
        # 為了比較，也畫出紅線 (Fixed)，證明在大雨下它會死掉
        # 使用相同的 time_in_cycle 來模擬連續撞擊
        wave_f = (1.0 * eff_f) * np.exp(-z_f * 2 * np.pi * param_fn * time_in_cycle) * \
                 np.sin(wd * time_in_cycle)

        fig = go.Figure()
        
        # 畫波形 (Smart)
        fig.add_trace(go.Scatter(x=t_arr*1000, y=wave_s, mode='lines', name='Smart (Active Drainage)', 
                                line=dict(color='#2e7d32', width=3)))
        
        # 畫波形 (Fixed) - 讓使用者能看到失效對比
        fig.add_trace(go.Scatter(x=t_arr*1000, y=wave_f, mode='lines', name='Fixed (Passive)', 
                                line=dict(color='#c62828', width=2, dash='dot')))
        
        # 畫垂直虛線標示撞擊點
        for i in range(1, int(VIEW_WINDOW/T_cycle) + 1):
            fig.add_vline(x=i*T_cycle*1000, line_dash="solid", line_color="gray", opacity=0.2)
            
        fig.update_layout(
            title=f"Frequency Response Analysis @ {val_freq} Hz",
            xaxis_title="Time (ms)", 
            yaxis_title="Voltage (V)", 
            height=450, 
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(range=[0, 150], showgrid=True), 
            yaxis=dict(range=[-1.2, 1.2]),
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: 場域模擬 (Field Simulation / Monte Carlo) =================
with tab_field:
    st.markdown(f"#### {t['field_header']}")
    
    # --- 1. 時間序列模擬 (Time-Series Simulation) ---
    st.markdown("##### 1. Long-term Rainfall Simulation")
    col_input, col_sim = st.columns([1, 3])
    
    with col_input:
        st.subheader(t["sim_params"])
        sim_duration = st.slider(t["sim_duration"], 1, 24, 12)
        
        # 生成模擬氣象數據
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
            # Smart System
            f_s, z_s, eff_s, tau_s, _, _, pos_s, loc_s = engine.get_params(R, W, "Smart")
            trunc_s = 1 / (1 + PhysConfig.TRUNCATION_SHAPE_FACTOR * f_s * tau_s) 
            
            # Fixed System
            f_f, z_f, eff_f, tau_f, _, _, pos_f, loc_f = engine.get_params(R, W, "Fixed")
            trunc_f = 1 / (1 + PhysConfig.TRUNCATION_SHAPE_FACTOR * f_f * tau_f)
            
            # 能量計算
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
        # 單位顯示 (Unit Display)
        m1.metric(t["metric_fixed"], f"{int(cum_f):,} {t['unit_energy']}", "Baseline")
        m2.metric(t["metric_smart"], f"{int(cum_s):,} {t['unit_energy']}", f"+{gain:.1f}%")
        m3.metric(t["metric_eroi"], f"{eroi:.1f}", f"Cost: {drainage_cost_pct}%")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_s_list, fill='tozeroy', name='Smart', line=dict(color='#2e7d32')))
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_f_list, fill='tozeroy', name='Fixed', line=dict(color='#c62828')))
        # 圖表標題加入單位
        fig2.update_layout(title=t["chart_cum_title"], yaxis_title=f"Total Energy ({t['unit_energy']})", height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig2, use_container_width=True)
        
    st.markdown("---")

    # --- 2. 蒙地卡羅驗證 (Monte Carlo Verification) ---
    st.markdown("##### 2. Monte Carlo Stochastic Validation")
    st.caption("利用 Marshall-Palmer 分佈生成隨機雨滴，驗證系統在非理想條件下的魯棒性。")
    
    col_ui1, col_ui2 = st.columns(2)
    mc_rain = col_ui1.slider(f"{t['rain_rate']} (Monte Carlo)", 10, 100, 50)
    mc_wet = col_ui2.slider("Wetness Factor (0=Dry, 1=Sat)", 0.0, 1.0, 0.1)

    if st.button(t["sim_start_btn"]):
        masses, velocities = generate_storm_profile(n_drops=1000, rain_rate_mmph=mc_rain)
        st.success(t["sim_success"].format(n=len(masses)))
        
        c1, c2 = st.columns(2)
        with c1:
            # 雨滴分佈圖
            fig_mc1, ax = plt.subplots(figsize=(5, 4))
            ax.hist(velocities, bins=25, color='#4A90E2', alpha=0.7)
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Count")
            ax.set_title("Raindrop Velocity Distribution")
            st.pyplot(fig_mc1)
            
        with c2:
            # 單顆隨機雨滴響應
            idx = np.random.randint(0, len(masses))
            t_rk, v_rk = rk4_solver(0.005, 150, 0.0001, 0.1, masses[idx], velocities[idx], mc_wet)
            fig_mc2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(t_rk*1000, v_rk, color='#FF6B6B')
            ax2.set_xlabel("Time (ms)")
            ax2.set_





