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
# 主程式 (Main App)
# ==========================================

st.set_page_config(
    page_title="Eco-Rain: Digital Twin Platform",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS 設定 ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 10px;
    }
    .metric-card h4 {
        margin-top: 0; color: #000000 !important; font-size: 16px; text-transform: uppercase;
    }
    .theory-box {
        background-color: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .citation-box { 
        background-color: #fff3e0; padding: 15px; border-radius: 5px; border-left: 5px solid #ff9800; font-size: 0.9em; margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

class PhysicsEngine:
    def __init__(self, area=2.5, fn=100, length=5.0):
        self.area = area
        self.fn = fn
        self.length = length # NEW! 新增樑長度屬性

    def get_params(self, rain, wind, mode="Fixed", freq_override=None):
        if rain <= 0: return 0, 0.008, 0, 0, 0, 0, 1.0 # 新增 pos_factor 返回值
        
        D0 = 0.9 * (rain ** 0.21) 
        V_term = 3.778 * (D0 ** 0.67) 
        
        if freq_override is not None:
            freq_est = freq_override
        else:
            freq_est = (rain / 100.0) * 60.0 
            if freq_est < 1: freq_est = 1 
            
        wetness = min(1.0, rain / 120.0)
        if mode == "Smart": wetness *= 0.2 
        
        zeta = 0.008 + (0.07 * wetness) 
        
        if mode == "Smart":
            eff_angle = 1.0 
        else:
            # 簡單的風偏角模型
            theta = np.arctan(wind / (V_term if V_term>0 else 1))
            eff_angle = max(0, np.cos(theta))
            
        wn = 2 * np.pi * self.fn
        tau = 1 / (zeta * wn)
        wd = wn * np.sqrt(1 - zeta**2)
        
        # NEW! 隨機落點因子 (Position Factor)
        # 模擬雨滴打在樑上的隨機位置：越靠近尖端(Length)，力臂越大
        rand_loc = np.random.normal(loc=self.length*0.7, scale=self.length*0.2)
        rand_loc = np.clip(rand_loc, 0, self.length)
        pos_factor = (rand_loc / self.length) ** 2 # 力矩平方正比
        
        return freq_est, zeta, eff_angle, tau, wd, V_term, pos_factor, rand_loc

# --- 側邊欄 ---
st.title("Eco-Rain: 壓電雨能採集數位孿生系統")
st.caption("Physics-Informed Digital Twin Platform")
st.sidebar.markdown("### 全域設定 (Global Settings)")
st.sidebar.markdown("**目標材料模型 (Target Material):**")
st.sidebar.info("TE Connectivity LDT0-028K (PVDF)")

# NEW! 新增樑長度設定
param_beam_len = st.sidebar.number_input("懸臂樑長度 L (cm)", 3.0, 10.0, 5.0, step=0.5)
param_area = st.sidebar.number_input("感測器有效面積 (cm^2)", 0.5, 10.0, 2.5, format="%.1f")
param_fn = st.sidebar.number_input("裝置共振頻率 (Hz)", 50, 200, 100, format="%d")

# NEW! 排水耗能設定
drainage_cost_pct = st.sidebar.slider("主動排水耗能係數 (%)", 1.0, 10.0, 5.0)

engine = PhysicsEngine(area=param_area, fn=param_fn, length=param_beam_len)

st.sidebar.markdown("---")
st.sidebar.text("Developed for Science Edge 2026")

# --- 分頁內容 ---
tab_theory, tab_lab, tab_field = st.tabs(["理論架構與邏輯 (Theory)", "物理實驗室 (Lab Mode)", "場域模擬 (Field Mode)"])

# ================= TAB 1: 理論架構 (保留您的原版 + 新增 APA) =================
with tab_theory:
    st.header("系統運算邏輯與物理模型")
    st.markdown("本數位孿生系統結合流體力學、壓電材料動力學與幾何向量分析，透過數值預測系統表現。")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(r"""
        <div class="theory-box">
        <h4>1. 氣象輸入模型 (Stochastic Input)</h4>
        <p>雨滴並非均勻大小。我們採用 <b>Marshall-Palmer 分佈</b> 來描述真實降雨中的雨滴粒徑機率密度：</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"N(D) = N_0 e^{-\Lambda D}")
        st.markdown(r"""其中 $\Lambda$ 取決於降雨強度 (Rain Rate)。基於此，我們利用 **Gunn-Kinzer** 的經驗公式推算終端速度：""")
        st.latex(r"v_t = 9.65 - 10.3 e^{-0.6D}")

    with col_t2:
        st.markdown(r"""
        <div class="theory-box">
        <h4>2. 壓電動力學模型 (Dynamics)</h4>
        <p>壓電懸臂樑被建模為一個<b>二階阻尼彈簧-質量系統</b>。動態阻尼係數 $\zeta(t)$ 隨水膜累積而變化：</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"m_{\text{eff}} \ddot{x} + c \dot{x} + k x = F_{\text{impact}}(t)")
        st.latex(r"\zeta(t) = 0.045 + 0.275 \cdot W(t)")
        st.info("Logic: 透過 RK4 積分器，以 0.1ms 的解析度還原波形。")

    st.subheader("3. 幾何追蹤與力臂效應 (Smart Tracking & Moment Arm)")
    col_t3, col_t4 = st.columns(2)
    with col_t3:
        st.markdown(r"""
        <div class="theory-box">
        <h4>向量合成 (Vector Analysis)</h4>
        <p>撞擊角度 $\theta$ 由水平風速與垂直雨速決定：</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\theta_{\text{impact}} = \arctan\left(\frac{V_{\text{wind}}}{V_{\text{term}}}\right)")
    with col_t4:
        st.markdown(r"""
        <div class="theory-box">
        <h4>力臂效應 (Moment Arm)</h4>
        <p>雨滴落點 $x$ 對發電量的影響呈平方關係：</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"E_{gen} \propto \left(\frac{x}{L}\right)^2")

    # NEW! APA 引用區塊
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
    Yuk, J., Leem, A., Thomas, K., & Jung, S. (2025). Leaf-inspired rain-energy harvesting device. <i>Biological and Environmental Engineering, Cornell University</i>.</p>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 2: 物理實驗室 (保留您的原版 Lab Mode) =================
with tab_lab:
    st.markdown("#### 變因控制實驗")
    col_ctrl, col_viz = st.columns([1, 2])
    with col_ctrl:
        st.subheader("參數控制")
        val_rain = st.slider("1. 降雨強度 (mm/hr)", 0, 150, 50)
        val_wind = st.slider("2. 風速 (m/s)", 0.0, 30.0, 5.0)
        val_freq = st.slider("3. 撞擊頻率 (Hz)", 5, 120, 30)

        # 這裡調用 PhysicsEngine 取得參數
        # 注意: 我們這裡不使用隨機落點(pos_factor=1)，因為這是實驗室定點測試
        _, z_f, eff_f, tau_f, wd, _, _, _ = engine.get_params(val_rain, val_wind, "Fixed", freq_override=val_freq)
        _, z_s, eff_s, tau_s, _, _, _, _  = engine.get_params(val_rain, val_wind, "Smart", freq_override=val_freq)
        
        time_window = 3 * tau_f * 1000 
        impact_period = 1000 / val_freq 
        is_truncated = impact_period < time_window
        status_color = "#d32f2f" if is_truncated else "#2e7d32"
        status_text = "波形截斷 (Waveform Truncated)" if is_truncated else "完整釋放 (Full Decay)"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>物理狀態分析</h4>
            <p><b>阻尼比 (Zeta):</b> <span style="color:#d32f2f">{z_f:.4f} (Fixed)</span> vs <span style="color:#2e7d32">{z_s:.4f} (Smart)</span></p>
            <p><b>能量釋放窗:</b> {time_window:.1f} ms</p>
            <p class="status-text" style="color:{status_color};">{status_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_viz:
        st.subheader("微觀視圖：阻尼震盪波形")
        t = np.linspace(0, 0.15, 1000) 
        T_impact = 1 / val_freq
        amp_f = 1.0 * eff_f
        wave_f = amp_f * np.exp(-z_f * 2 * np.pi * param_fn * t) * np.sin(wd * t)
        wave_s = 1.0 * eff_s * np.exp(-z_s * 2 * np.pi * param_fn * t) * np.sin(wd * t)
        mask = t <= T_impact
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t[mask]*1000, y=wave_s[mask], mode='lines', name='Smart', line=dict(color='#2e7d32', width=3)))
        fig.add_trace(go.Scatter(x=t[mask]*1000, y=wave_f[mask], mode='lines', name='Fixed', line=dict(color='#c62828', width=3)))
        fig.add_vline(x=T_impact*1000, line_dash="dash", line_color="black", annotation_text="Next Impact")
        fig.update_layout(xaxis_title="Time (ms)", yaxis_title="Voltage", height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: 場域模擬 (融合版：原本的氣象生成 + EROI + 落點熱圖) =================
with tab_field:
    st.markdown("#### 真實情境模擬 (Real-world Scenario Simulation)")
    col_input, col_sim = st.columns([1, 3])
    
    with col_input:
        st.subheader("模擬參數")
        sim_duration = st.slider("模擬時長 (小時)", 1, 24, 12)
        
        # 保留您原本寫得很好的高斯分佈氣象生成
        h = np.arange(0, sim_duration + 1, 1) 
        peak_time = sim_duration / 2
        r = 10 + 100 * np.exp(-0.5 * (h - peak_time)**2/2.5) 
        w = 5 + 25 * np.exp(-0.5 * (h - peak_time)**2/3) + np.random.normal(0, 2, len(h))
        df = pd.DataFrame({'Time': h, 'Rain': np.clip(r, 0, None), 'Wind': np.clip(w, 0, None)})
        
        with st.expander("查看氣象數據"):
            st.dataframe(df, height=150)
            st.caption("Rain/Wind profile based on Gaussian distribution.")

    with col_sim:
        # 準備儲存數據的列表
        acc_s_list, acc_f_list = [], []
        cum_s, cum_f = 0, 0
        loc_history, eff_history = [], [] # 用於熱圖
        
        # 模擬迴圈
        for idx, row in df.iterrows():
            R, W = row['Rain'], row['Wind']
            
            # Smart Mode:
            f_s, z_s, eff_s, tau_s, _, _, pos_s, loc_s = engine.get_params(R, W, "Smart")
            trunc_s = 1 / (1 + 0.6 * f_s * tau_s) 
            
            # Fixed Mode:
            f_f, z_f, eff_f, tau_f, _, _, pos_f, loc_f = engine.get_params(R, W, "Fixed")
            trunc_f = 1 / (1 + 0.6 * f_f * tau_f)
            
            # 能量計算 (加入 pos_factor 力臂效應)
            # 智慧系統排水成本扣除
            energy_s_raw = f_s * (eff_s**2) * trunc_s * (R**0.5) * pos_s
            drainage_loss = energy_s_raw * (drainage_cost_pct / 100.0)
            energy_s_net = energy_s_raw - drainage_loss
            
            energy_f = f_f * (eff_f**2) * trunc_f * (R**0.5) * pos_f
            
            cum_s += energy_s_net
            cum_f += energy_f
            
            acc_s_list.append(cum_s)
            acc_f_list.append(cum_f)
            
            # 收集熱圖數據
            loc_history.append(loc_s)
            eff_history.append(pos_s)
            
        # 計算指標
        gain = ((cum_s - cum_f) / cum_f) * 100 if cum_f > 0 else 0
        eroi = cum_s / (cum_s * (drainage_cost_pct/100)) if cum_s > 0 else 0
        
        # 指標顯示區
        m1, m2, m3 = st.columns(3)
        m1.metric("固定式總產出", f"{int(cum_f):,}", "Baseline")
        m2.metric("智慧式總產出", f"{int(cum_s):,}", f"+{gain:.1f}%")
        m3.metric("EROI (能源投報率)", f"{eroi:.1f}", f"Cost: {drainage_cost_pct}%")
        
        # 圖表 1: 累積能量
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_s_list, fill='tozeroy', name='Smart (Active)', line=dict(color='#2e7d32')))
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_f_list, fill='tozeroy', name='Fixed (Passive)', line=dict(color='#c62828')))
        fig2.update_layout(title="累積發電量模擬 (Cumulative Energy)", height=350, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig2, use_container_width=True)
        
        # 圖表 2: NEW! 落點熱圖 (Impact Heatmap)
        st.markdown(f"**雨滴隨機落點分佈 (Impact Position Analysis)** - L={param_beam_len}cm")
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Scatter(
            x=df['Time'], y=loc_history,
            mode='markers',
            marker=dict(size=8, color=eff_history, colorscale='Viridis', showscale=True, colorbar=dict(title="Efficiency")),
            name='Impact Event'
        ))
        fig_heat.add_hline(y=param_beam_len, line_dash="dash", line_color="gray", annotation_text="Tip (Max Moment)")
        fig_heat.update_layout(yaxis_title="Position (cm)", height=300, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

# --- 下方模擬區 (保留原本的按鈕，因為它是很好的 Demo) ---
st.markdown("---")
st.header("數位孿生驗證：蒙地卡羅雨滴模擬")
col_ui1, col_ui2 = st.columns(2)
mc_rain = col_ui1.slider("降雨強度 (Rate)", 10, 100, 50)
mc_wet = col_ui2.slider("水膜係數 (Wetness)", 0.0, 1.0, 0.1)

if st.button("執行蒙地卡羅模擬 (Monte Carlo)"):
    masses, velocities = generate_storm_profile(n_drops=1000, rain_rate_mmph=mc_rain)
    st.success(f"生成 {len(masses)} 顆符合 Marshall-Palmer 分佈的雨滴數據。")
    
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(velocities, bins=25, color='#4A90E2', alpha=0.7)
        ax.set_xlabel("Velocity (m/s)")
        ax.set_title("Distribution Check")
        st.pyplot(fig)
    with c2:
        idx = np.random.randint(0, len(masses))
        # 呼叫 rk4 畫圖
        t, v = rk4_solver(0.005, 150, 0.0001, 0.1, masses[idx], velocities[idx], mc_wet)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(t*1000, v, color='#FF6B6B')
        ax2.set_xlabel("Time (ms)")
        ax2.set_title(f"Single Drop Response (m={masses[idx]:.1f}mg)")
        st.pyplot(fig2)
