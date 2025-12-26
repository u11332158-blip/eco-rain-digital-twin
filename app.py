import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
        margin-top: 0;
        color: #000000 !important;
        font-size: 16px;
        text-transform: uppercase;
    }
    .metric-card p, .metric-card span {
        color: #333333 !important;
    }
    .theory-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .theory-box h4 {
        color: #1565c0 !important;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .theory-box p, .theory-box li, .theory-box span, .theory-box div {
        color: #212121 !important; 
        font-size: 1.05em;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

class PhysicsEngine:
    def __init__(self, area=2.5, fn=100):
        self.area = area
        self.fn = fn 

    def get_params(self, rain, wind, mode="Fixed", freq_override=None):
        if rain <= 0: return 0, 0.008, 0, 0, 0, 0
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
            theta = np.arctan(wind / (V_term if V_term>0 else 1))
            eff_angle = max(0, np.cos(theta))
        wn = 2 * np.pi * self.fn
        tau = 1 / (zeta * wn)
        wd = wn * np.sqrt(1 - zeta**2)
        return freq_est, zeta, eff_angle, tau, wd, V_term

# --- 側邊欄 ---
st.title("Eco-Rain: 壓電雨能採集數位孿生系統")
st.caption("Physics-Informed Digital Twin Platform")
st.sidebar.markdown("### 全域設定 (Global Settings)")
st.sidebar.markdown("**目標材料模型 (Target Material):**")
st.sidebar.info("TE Connectivity LDT0-028K (PVDF)")

with st.sidebar.expander("查看材料物理參數"):
    # 在 HTML 區塊內使用雙斜線 \\
    st.markdown(r"""
    <div style='font-size: 0.85em; color: #555;'>
    <b>Physical Properties:</b><br>
    • Piezo Coefficient ($d_{31}$): 23e-12 C/N<br>
    • Young's Modulus: 2-4 GPa<br>
    • Capacitance: ~500 pF<br>
    • Sensitivity: ~10 mV/$\mu\epsilon$
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("**幾何與頻率參數:**")
param_area = st.sidebar.number_input("感測器有效面積 (cm^2)", 0.5, 10.0, 2.5, format="%.1f")
param_fn = st.sidebar.number_input("裝置共振頻率 (Hz)", 50, 200, 100, format="%d")
engine = PhysicsEngine(area=param_area, fn=param_fn)
st.sidebar.markdown("---")
st.sidebar.text("Developed for Science Edge 2025")

# --- 分頁內容 ---
tab_theory, tab_lab, tab_field = st.tabs(["理論架構與邏輯 (Theory)", "物理實驗室 (Lab Mode)", "場域模擬 (Field Mode)"])

# ================= TAB 1: 理論架構 (重點修正區：使用雙斜線) =================
with tab_theory:
    st.header("系統運算邏輯與物理模型")
    st.markdown("本數位孿生系統結合流體力學、壓電材料動力學與幾何向量分析，透過數值預測系統表現。")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        # 注意：在 HTML tag 內的 LaTeX，\Lambda 改為 \\Lambda
        st.markdown(r"""
        <div class="theory-box">
        <h4>1. 氣象輸入模型 (Stochastic Input)</h4>
        <p>雨滴並非均勻大小。我們採用 <b>Marshall-Palmer 分佈</b> 來描述真實降雨中的雨滴粒徑機率密度：</p>
        </div>
        """, unsafe_allow_html=True)
        
        # st.latex 以外部獨立調用時，單斜線即可
        st.latex(r"N(D) = N_0 e^{-\Lambda D}")
        
        st.markdown(r"""
        其中 $\Lambda$ 取決於降雨強度 (Rain Rate)。基於此，我們利用 **Atlas et al.** 的經驗公式推算終端速度：
        """)
        st.latex(r"v_t = 9.65 - 10.3 e^{-0.6D}")
        st.info("Logic: 程式根據輸入的 mm/hr，逆推產生符合物理統計特性的隨機雨滴群。")

    with col_t2:
        st.markdown(r"""
        <div class="theory-box">
        <h4>2. 壓電動力學模型 (Dynamics)</h4>
        <p>壓電懸臂樑被建模為一個<b>二階阻尼彈簧-質量系統</b> (Second-order Spring-Mass-Damper System)。</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"m_{\text{eff}} \ddot{x} + c \dot{x} + k x = F_{\text{impact}}(t)")
        
        st.markdown(r"""
        為了精確模擬雨滴撞擊瞬間的非線性響應，我們採用 **Runge-Kutta 4th Order (RK4)** 數值方法進行微分方程求解。
        """)
        st.latex(r"V_{\text{out}}(t) \propto d_{31} \cdot \epsilon(t)")
        st.info("Logic: 透過 RK4 積分器，以 0.1ms 的解析度還原波形。")

    st.markdown("---")
    
    st.subheader("3. 壓電薄膜自動追蹤機制 (Smart Tracking Design)")
    st.markdown("針對戶外側風造成的能量損失，本系統導入向量追蹤演算法。")
    
    col_t3, col_t4 = st.columns([1, 1])
    
    with col_t3:
        # 關鍵修正：HTML 內部的 LaTeX 全部改為雙斜線 (\\text, \\theta, \\arctan)
        st.markdown(r"""
        <div class="theory-box">
        <h4>向量合成原理 (Vector Analysis)</h4>
        <p>雨滴在風場中受到水平風速 ($V_{\text{wind}}$) 與垂直終端速度 ($V_{\text{term}}$) 的共同作用，形成合成速度向量 ($V_{\text{resultant}}$)。</p>
        <p>撞擊角度 $\theta$ 計算如下：</p>
        </div>
        """, unsafe_allow_html=True)
        
        # st.latex 這裡維持單斜線，因為它不包在 HTML tag 裡
        st.latex(r"\theta_{\text{impact}} = \arctan\left(\frac{V_{\text{wind}}}{V_{\text{term}}}\right)")
        
        st.markdown(r"""
        **能量損失機制：**<br>
        若壓電片保持水平，有效撞擊力僅為垂直分量，造成餘弦損失 (Cosine Loss)：
        """, unsafe_allow_html=True)
        st.latex(r"E_{\text{fixed}} \propto (F \cdot \cos\theta)^2")

    with col_t4:
        # 關鍵修正：HTML 內部的 LaTeX 全部改為雙斜線 (\\phi)
        st.markdown(r"""
        <div class="theory-box">
        <h4>自動補償邏輯 (Optimization Logic)</h4>
        <p>系統透過風速計回傳數據，即時計算最佳傾角 $\phi_{\text{opt}}$，使壓電片法向量與雨滴路徑平行。</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"\phi_{\text{opt}} = \theta_{\text{impact}}")
        st.markdown(r"""
        **優化效益：**
        1. **動能最大化**：消除餘弦損失，使 $F_{\text{eff}} \approx F_{\text{total}}$。
        2. **頻率響應優化**：垂直撞擊能更有效激發 $d_{31}$ 模式的形變。
        """)
        st.success("Logic: 數位孿生模型即時計算幾何關係，動態調整效率係數。")

# ================= TAB 2: 物理實驗室 =================
with tab_lab:
    st.markdown("#### 變因控制實驗")
    col_ctrl, col_viz = st.columns([1, 2])
    with col_ctrl:
        st.subheader("參數控制")
        val_rain = st.slider("1. 降雨強度 (mm/hr)", 0, 150, 50)
        val_wind = st.slider("2. 風速 (m/s)", 0.0, 30.0, 5.0)
        val_freq = st.slider("3. 撞擊頻率 (Hz)", 5, 120, 30)

        _, z_f, eff_f, tau_f, wd, _ = engine.get_params(val_rain, val_wind, "Fixed", freq_override=val_freq)
        _, z_s, eff_s, tau_s, _, _  = engine.get_params(val_rain, val_wind, "Smart", freq_override=val_freq)
        
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
        fig.add_vline(x=T_impact*1000, line_dash="dash", line_color="black")
        fig.update_layout(xaxis_title="Time (ms)", yaxis_title="Voltage", height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: 場域模擬 =================
with tab_field:
    st.markdown("#### 真實情境模擬")
    col_input, col_sim = st.columns([1, 3])
    with col_input:
        st.subheader("模擬參數")
        sim_duration = st.slider("模擬時長 (小時)", 1, 24, 12)
        h = np.arange(0, sim_duration + 1, 1) 
        peak_time = sim_duration / 2
        r = 10 + 100 * np.exp(-0.5 * (h - peak_time)**2/2.5) 
        w = 5 + 25 * np.exp(-0.5 * (h - peak_time)**2/3) + np.random.normal(0, 2, len(h))
        df = pd.DataFrame({'Time': h, 'Rain': np.clip(r, 0, None), 'Wind': np.clip(w, 0, None)})
        with st.expander("查看氣象數據"):
            st.dataframe(df, height=150)

    with col_sim:
        acc_s_list, acc_f_list = [], []
        cum_s, cum_f = 0, 0
        for idx, row in df.iterrows():
            R, W = row['Rain'], row['Wind']
            f_s, z_s, eff_s, tau_s, _, _ = engine.get_params(R, W, "Smart")
            trunc_s = 1 / (1 + 0.6 * f_s * tau_s) 
            cum_s += f_s * (eff_s**2) * trunc_s * (R**0.5) 
            acc_s_list.append(cum_s)
            
            f_f, z_f, eff_f, tau_f, _, _ = engine.get_params(R, W, "Fixed")
            trunc_f = 1 / (1 + 0.6 * f_f * tau_f)
            cum_f += f_f * (eff_f**2) * trunc_f * (R**0.5)
            acc_f_list.append(cum_f)
            
        gain = ((cum_s - cum_f) / cum_f) * 100 if cum_f > 0 else 0
        m1, m2 = st.columns(2)
        m1.metric("固定式總產出", f"{int(cum_f):,}", "Baseline")
        m2.metric("智慧式總產出", f"{int(cum_s):,}", f"+{gain:.1f}%")
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_s_list, fill='tozeroy', name='Smart', line=dict(color='#2e7d32')))
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_f_list, fill='tozeroy', name='Fixed', line=dict(color='#c62828')))
        fig2.update_layout(title="累積發電量模擬", height=400)
        st.plotly_chart(fig2, use_container_width=True)

# --- 下方模擬區 ---
st.markdown("---")
st.header("數位孿生驗證：蒙地卡羅雨滴模擬")
col_ui1, col_ui2 = st.columns(2)
mc_rain = col_ui1.slider("降雨強度 (Rate)", 10, 100, 50)
mc_wet = col_ui2.slider("水膜係數 (Wetness)", 0.0, 1.0, 0.1)

if st.button("執行蒙地卡羅模擬"):
    masses, velocities = generate_storm_profile(n_drops=1000, rain_rate_mmph=mc_rain)
    st.success(f"生成 {len(masses)} 顆雨滴數據。")
    
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(velocities, bins=25, color='#4A90E2', alpha=0.7)
        ax.set_xlabel("Velocity (m/s)")
        ax.set_title("Marshall-Palmer Dist.")
        st.pyplot(fig)
    with c2:
        idx = np.random.randint(0, len(masses))
        t, v = rk4_solver(0.005, 150, 0.0001, 0.1, masses[idx], velocities[idx], mc_wet)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(t*1000, v, color='#FF6B6B')
        ax2.set_xlabel("Time (ms)")
        ax2.set_title("Impact Response")
        st.pyplot(fig2)
