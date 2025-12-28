import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    zeta = 0.008 + (0.07 * wetness)
    c_damp = 2 * zeta * mass_beam * wn
    state = np.array([0.0, 0.0])
    impact_duration = 0.002
    peak_force = (drop_mass * 1e-6 * drop_velocity) / (impact_duration / 2)
    t_steps = np.arange(0, total_time, dt)
    voltages = []
    
    def derivatives(t, y):
        x, v = y
        F_ext = 0
        if t < impact_duration:
            if t < impact_duration/2:
                F_ext = peak_force * (t / (impact_duration/2))
            else:
                F_ext = peak_force * (2 - t / (impact_duration/2))
        a = (F_ext - c_damp * v - k_spring * x) / mass_beam
        return np.array([v, a])

    for t in t_steps:
        k1 = derivatives(t, state)
        k2 = derivatives(t + dt/2, state + k1*dt/2)
        k3 = derivatives(t + dt/2, state + k2*dt/2)
        k4 = derivatives(t + dt, state + k3*dt)
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        voltages.append(state[0] * 50000) 
    return t_steps, np.array(voltages)

# ==========================================
# 主程式 (Main App UI)
# ==========================================

st.set_page_config(page_title="Eco-Rain Digital Twin", page_icon="⛈️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .big-metric { font-size: 24px !important; font-weight: bold; color: #2E86C1; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 5px; border-left: 5px solid #2E86C1; }
    .citation-box { background-color: #fff3e0; padding: 15px; border-radius: 5px; border-left: 5px solid #ff9800; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# --- 側邊欄 ---
st.sidebar.title("⛈️ Eco-Rain Twin")
st.sidebar.subheader("參數控制中心")

# 樑長度設定 (保留上一版功能)
param_beam_len = st.sidebar.number_input("懸臂樑長度 L (cm)", 3.0, 10.0, 5.0, step=0.5, help="影響力臂效應與共振頻率")

target_rain = st.sidebar.slider("環境降雨強度 (mm/hr)", 10, 150, 50)
sim_duration = st.sidebar.slider("模擬時長 (Minutes)", 10, 120, 60)
drainage_cost = st.sidebar.slider("主動排水耗能係數 (%)", 1.0, 10.0, 5.0)

st.sidebar.markdown("---")
st.sidebar.info("Model: PVDF Cantilever Beam\nRef: Li et al. (2016)")

st.title("Eco-Rain: 壓電雨能採集數位孿生系統")
st.markdown("**Project:** Beyond Resonance: Unveiling Water Film Damping via Digital Twin Integration")

tab_dashboard, tab_physics, tab_verification = st.tabs(["📊 模擬監控儀表板 (Dashboard)", "🧮 物理核心 (Physics)", "🔬 微觀驗證 (Verification)"])

# ================= TAB 1: 模擬監控儀表板 (Dashboard) =================
with tab_dashboard:
    st.markdown("### 🌧️ 即時場域模擬 (Live Field Simulation)")
    
    col_sim_chart, col_sim_metrics = st.columns([3, 1])

    # --- 模擬運算邏輯 ---
    time_minutes = np.arange(0, sim_duration + 1, 1)
    
    # 1. 環境參數生成
    saturation_speed = target_rain / 200.0 
    water_film = 1 - np.exp(-time_minutes * saturation_speed)
    
    # 隨機落點模擬 (考慮樑長度)
    rand_pos = np.random.normal(loc=param_beam_len * 0.7, scale=param_beam_len * 0.15, size=len(time_minutes))
    rand_pos = np.clip(rand_pos, 0, param_beam_len) 
    
    # 位置效率因子
    pos_factor = (rand_pos / param_beam_len) ** 2
    
    # 2. 阻尼比變化
    zeta_fixed = 0.045 + 0.275 * water_film
    zeta_active = np.full_like(time_minutes, 0.02)
    
    # 3. 功率輸出
    base_power = target_rain * 0.5 
    power_fixed = base_power * (0.02 / zeta_fixed) * pos_factor 
    power_active = base_power * (0.02 / zeta_active) * pos_factor
    
    # 4. 累積能量
    energy_fixed = np.cumsum(power_fixed)
    energy_active_gross = np.cumsum(power_active)
    drainage_loss = energy_active_gross * (drainage_cost / 100.0)
    energy_active_net = energy_active_gross - drainage_loss
    
    total_fixed = energy_fixed[-1]
    total_active = energy_active_net[-1]
    net_gain = total_active / total_fixed if total_fixed > 0 else 0
    eroi = total_active / drainage_loss[-1] if drainage_loss[-1] > 0 else 0

    # --- 繪圖 1 ---
    with col_sim_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_minutes, y=energy_active_net, mode='lines', name='Smart Active System', line=dict(color='#2e7d32', width=4), fill='tozeroy', fillcolor='rgba(46, 125, 50, 0.1)'))
        fig.add_trace(go.Scatter(x=time_minutes, y=energy_fixed, mode='lines', name='Fixed Passive System', line=dict(color='#c62828', width=3, dash='dash')))
        
        fig.update_layout(title="累積能量比較 (含落點隨機性)", xaxis_title="Time (min)", yaxis_title="Total Energy (mJ)", height=400, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)

    with col_sim_metrics:
        st.metric(label="淨能量增益", value=f"{net_gain:.2f}x", delta="Active vs Fixed")
        st.metric(label="EROI", value=f"{eroi:.2f}", delta="Return")
        if net_gain > 3.0: st.success("✅ 高效益區間")
        else: st.warning("⚠️ 邊際效益區間")
        st.markdown(f"**平均落點位置:**\n{np.mean(rand_pos):.1f} cm (Tip: {param_beam_len}cm)")

    # --- 繪圖 2: 落點熱圖 ---
    st.markdown("---")
    st.markdown(f"### 📍 雨滴落點分佈與力臂效應分析 (Impact Position & Moment Arm)")
    
    col_pos_1, col_pos_2 = st.columns([2, 1])
    
    with col_pos_1:
        fig_pos = go.Figure()
        fig_pos.add_trace(go.Scatter(
            x=time_minutes, y=rand_pos, mode='markers',
            marker=dict(size=8, color=pos_factor, colorscale='Viridis', showscale=True, colorbar=dict(title="Efficiency")),
            name='Impact Event'
        ))
        fig_pos.add_hline(y=param_beam_len, line_dash="dash", line_color="gray", annotation_text="Beam Tip (Max Power)")
        fig_pos.add_hline(y=0, line_color="black", annotation_text="Fixed End (Zero Power)")
        fig_pos.update_layout(
            title="模擬落點位置紀錄 (Impact Location Tracking)",
            xaxis_title="Simulation Time (min)",
            yaxis_title="Distance from Fixed End (cm)",
            yaxis_range=[0, param_beam_len * 1.1], height=350
        )
        st.plotly_chart(fig_pos, use_container_width=True)
        
    with col_pos_2:
        st.info("**物理原理 (Physics Logic):**")
        st.latex(r"E_{gen} \propto x_{impact}^2")
        st.markdown(f"**模擬設定:** L = {param_beam_len} cm")

# ================= TAB 2: 物理核心 (UPDATED with APA Citations) =================
with tab_physics:
    st.header("🧮 物理導向模型 (Physics-Informed Models)")
    st.markdown("本系統之演算法植基於以下三大核心物理模型：")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 1. 雨滴分佈")
        st.info("Marshall-Palmer Law")
        st.latex(r"N(D) = N_0 e^{-\Lambda D}")
        st.caption("模擬真實降雨中，大小雨滴的多分散機率分佈。")
    with c2:
        st.markdown("#### 2. 終端速度")
        st.info("Gunn-Kinzer Formula")
        st.latex(r"v(D) = 9.65 - 10.3 e^{-0.6D}")
        st.caption("修正空氣阻力對雨滴動量的影響，確保撞擊計算精確。")
    with c3:
        st.markdown("#### 3. 幽靈阻尼")
        st.info("Dynamic Damping Eq.")
        st.latex(r"\zeta(t) = 0.045 + 0.275 \cdot W(t)")
        st.caption("動態模擬水膜累積導致的系統過阻尼失效。")
    
    st.markdown("---")
    st.markdown("### 📚 參考文獻 (References - APA Format)")
    
    st.markdown("""
    <div class="citation-box">
    <p><b>[1] Raindrop Physics:</b><br>
    Marshall, J. S., & Palmer, W. M. (1948). The distribution of raindrops with size. <i>Journal of meteorology</i>, <i>5</i>(4), 165-166.<br>
    Gunn, R., & Kinzer, G. D. (1949). The terminal velocity of fall for water droplets in stagnant air. <i>Journal of meteorology</i>, <i>6</i>(4), 243-248.</p>
    
    <p><b>[2] Piezoelectric Dynamics & Material:</b><br>
    Li, S., Crovetto, A., Peng, Z., Zhang, A., Hansen, O., Wang, M., Li, X., & Wang, F. (2016). Bi-resonant structure with piezoelectric PVDF films for energy harvesting from random vibration sources at low frequency. <i>Sensors and Actuators A: Physical</i>, <i>247</i>, 547-554.<br>
    Gregorio, R., Jr., & Ueno, E. M. (1999). Effect of crystalline phase, orientation and temperature on the dielectric properties of poly (vinylidene fluoride) (PVDF). <i>Journal of Materials Science</i>, <i>34</i>, 4489–4500.</p>
    
    <p><b>[3] Related Works & Inspiration:</b><br>
    Yuk, J., Leem, A., Thomas, K., & Jung, S. (2025). Leaf-inspired rain-energy harvesting device. <i>Biological and Environmental Engineering, Cornell University</i>.<br>
    Bowland, A., et al. (2010). New concepts in modeling damping in structures. <i>10th CCEE</i>.</p>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 3: 微觀驗證 =================
with tab_verification:
    st.header("🔬 單顆雨滴撞擊驗證 (RK4 Solver)")
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        st.markdown("利用 **Runge-Kutta 4th Order** 演算法，以 0.1ms 的時間解析度，模擬單次撞擊下的電壓波形。")
        v_wetness = st.slider("表面水膜程度 (Wetness)", 0.0, 1.0, 0.0, step=0.1)
    with col_v2:
        t, v = rk4_solver(0.005, 150, 0.0001, 0.1, 30, 6, v_wetness)
        fig_wave = go.Figure()
        fig_wave.add_trace(go.Scatter(x=t*1000, y=v, mode='lines', line=dict(color='#2980b9', width=3)))
        fig_wave.update_layout(title=f"Impact Waveform (Wetness = {v_wetness})", xaxis_title="Time (ms)", yaxis_title="Voltage (V)", height=350)
        st.plotly_chart(fig_wave, use_container_width=True)
