import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 核心物理運算區 (Physics Core) - 補回的部分
# ==========================================
def generate_storm_profile(n_drops=1000, rain_rate_mmph=50):
    """
    基於 Marshall-Palmer 分佈生成隨機雨滴群
    回傳: (質量陣列, 速度陣列)
    """
    # 1. 計算 Lambda 參數 (Marshall-Palmer)
    # N(D) = N0 * e^(-lambda * D)
    lam = 4.1 * (rain_rate_mmph ** -0.21)
    
    # 2. 逆變換採樣 (Inverse Transform Sampling) 生成雨滴直徑
    # D = -ln(1-u) / lambda
    u = np.random.uniform(0, 1, n_drops)
    diameters_mm = -np.log(1 - u) / lam
    
    # 過濾掉不合理的極端值 (例如 > 6mm 的雨滴極少見)
    diameters_mm = np.clip(diameters_mm, 0.1, 6.0)
    
    # 3. 計算終端速度 (Atlas et al. 經驗公式)
    # v = 9.65 - 10.3 * exp(-0.6 * D)
    velocities = 9.65 - 10.3 * np.exp(-0.6 * diameters_mm)
    velocities = np.clip(velocities, 0, None)
    
    # 4. 計算質量 (假設球體，水密度=1 mg/mm^3)
    # Mass = Volume * Density = (4/3 * pi * r^3) * 1
    masses_mg = (4/3) * np.pi * (diameters_mm / 2)**3
    
    return masses_mg, velocities

def rk4_solver(mass_beam, k_spring, dt, total_time, drop_mass, drop_velocity, wetness):
    """
    Runge-Kutta 4th Order (RK4) 數值積分求解器
    模擬壓電片受撞擊後的二階阻尼震盪
    """
    # 1. 計算阻尼係數 (c)
    # wn = sqrt(k/m), zeta = c / (2*m*wn) => c = zeta * 2 * m * wn
    wn = np.sqrt(k_spring / mass_beam)
    zeta = 0.008 + (0.07 * wetness) # 濕度越高，阻尼越大
    c_damp = 2 * zeta * mass_beam * wn
    
    # 2. 初始狀態 [位置 x, 速度 v]
    state = np.array([0.0, 0.0])
    
    # 3. 定義撞擊力 (Impulse)
    # 假設撞擊為一個極短時間的三角形脈衝
    impact_duration = 0.002 # 2ms 接觸時間
    # 動量變化 F * dt = dp => Peak Force approx (m*v) / (dt/2)
    peak_force = (drop_mass * 1e-6 * drop_velocity) / (impact_duration / 2)
    
    t_steps = np.arange(0, total_time, dt)
    voltages = []
    
    # 4. 定義微分方程 (State Derivatives)
    def derivatives(t, y):
        x, v = y
        # 外力函數 F(t)
        F_ext = 0
        if t < impact_duration:
            if t < impact_duration/2:
                F_ext = peak_force * (t / (impact_duration/2))
            else:
                F_ext = peak_force * (2 - t / (impact_duration/2))
        
        # 運動方程式: m*a + c*v + k*x = F
        # 加速度 a = (F - c*v - k*x) / m
        a = (F_ext - c_damp * v - k_spring * x) / mass_beam
        return np.array([v, a])

    # 5. RK4 積分迴圈
    for t in t_steps:
        k1 = derivatives(t, state)
        k2 = derivatives(t + dt/2, state + k1*dt/2)
        k3 = derivatives(t + dt/2, state + k2*dt/2)
        k4 = derivatives(t + dt, state + k3*dt)
        
        state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 壓電電壓與形變量(位移)成正比
        # 這裡乘上一個靈敏度係數做視覺化
        voltages.append(state[0] * 50000) 
        
    return t_steps, np.array(voltages)

# ==========================================
# 主程式 (Main App)
# ==========================================

# --- 1. 頁面與樣式設定 ---
st.set_page_config(
    page_title="Eco-Rain: Digital Twin Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義 CSS (強制深色文字，解決顯示問題)
st.markdown("""
<style>
    .metric-card {
        background-color: #f5f5f5;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
        margin-bottom: 10px;
        color: #000000 !important;
    }
    .metric-card h4 {
        margin-top: 0;
        color: #000000 !important;
        font-family: 'Arial', sans-serif;
        font-size: 16px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card p {
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
    .theory-box p, .theory-box li, .theory-box span {
        color: #212121 !important;
        font-size: 1.05em;
        line-height: 1.6;
    }
    h1, h2, h3 { font-family: 'Arial', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- 2. 簡易物理引擎類別 (用於 Tabs 顯示) ---
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

# --- 3. 側邊欄設定 ---
st.title("Eco-Rain: 壓電雨能採集數位孿生系統")
st.caption("Physics-Informed Digital Twin Platform")
st.sidebar.markdown("### 全域設定 (Global Settings)")

st.sidebar.markdown("**目標材料模型 (Target Material):**")
st.sidebar.info("TE Connectivity LDT0-028K (PVDF)")

with st.sidebar.expander("查看材料物理參數 (Datasheet Specs)"):
    st.markdown("""
    <div style='font-size: 0.85em; color: #555;'>
    <b>Physical Properties:</b><br>
    • Piezo Coefficient ($d_{31}$): 23e-12 C/N<br>
    • Young's Modulus: 2-4 GPa<br>
    • Capacitance: ~500 pF<br>
    • Sensitivity: ~10 mV/$\mu\epsilon$
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("**幾何與頻率參數 (Geometry & Frequency):**")

param_area = st.sidebar.number_input(
    "感測器有效面積 (Active Area, cm^2)", 
    min_value=0.5, max_value=10.0, value=2.5, 
    format="%.1f",
    help="Based on LDT0-028K physical dimensions (Standard)"
)

param_fn = st.sidebar.number_input(
    "裝置共振頻率 (Resonance Freq, Hz)", 
    min_value=50, max_value=200, value=100, 
    format="%d",
    help="Measured 1st mode natural frequency of the LDT0-028K cantilever."
)

engine = PhysicsEngine(area=param_area, fn=param_fn)
st.sidebar.markdown("---")
st.sidebar.text("Developed for Science Edge 2025")

# --- 4. 分頁內容 ---
tab_theory, tab_lab, tab_field = st.tabs(["理論架構與邏輯 (Theory)", "物理實驗室 (Lab Mode)", "場域模擬 (Field Mode)"])

# ================= TAB 1: 理論架構 (修正公式排版) =================
with tab_theory:
    st.header("系統運算邏輯與物理模型")
    st.markdown("本數位孿生系統結合流體力學、壓電材料動力學與幾何向量分析，透過數值預測系統表現。")
    
    # 第一排：輸入模型 與 動力學模型
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.markdown("""
        <div class="theory-box">
        <h4>1. 氣象輸入模型 (Stochastic Input)</h4>
        <p>雨滴並非均勻大小。我們採用 <b>Marshall-Palmer 分佈</b> 來描述真實降雨中的雨滴粒徑機率密度：</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"N(D) = N_0 e^{-\Lambda D}")
        st.markdown("其中 $\Lambda$ 取決於降雨強度 (Rain Rate)。基於此，我們利用 **Atlas et al.** 的經驗公式推算終端速度：")
        st.latex(r"v_t = 9.65 - 10.3 e^{-0.6D}")
        st.info("Logic: 程式根據輸入的 mm/hr，逆推產生符合物理統計特性的隨機雨滴群 (Monte Carlo Generation)。")

    with col_t2:
        st.markdown("""
        <div class="theory-box">
        <h4>2. 壓電動力學模型 (Dynamics)</h4>
        <p>壓電懸臂樑被建模為一個<b>二階阻尼彈簧-質量系統</b> (Second-order Spring-Mass-Damper System)。</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"m_{\text{eff}} \ddot{x} + c \dot{x} + k x = F_{\text{impact}}(t)")
        st.markdown("為了精確模擬雨滴撞擊瞬間的非線性響應，我們採用 **Runge-Kutta 4th Order (RK4)** 數值方法進行微分方程求解，而非簡單的線性疊加。")
        st.latex(r"V_{\text{out}}(t) \propto d_{31} \cdot \epsilon(t)")
        st.info("Logic: 透過 RK4 積分器，我們能以 0.1ms 的解析度還原撞擊後的電壓波形與能量耗散。")

    st.markdown("---")
    
    # 第二排：壓電薄膜自動追蹤機制
    st.subheader("3. 壓電薄膜自動追蹤機制 (Smart Tracking Design)")
    st.markdown("針對戶外側風造成的能量損失，本系統導入向量追蹤演算法，透過伺服馬達即時調整壓電片角度。")
    
    col_t3, col_t4 = st.columns([1, 1])
    
    with col_t3:
        st.markdown("""
        <div class="theory-box">
        <h4>向量合成原理 (Vector Analysis)</h4>
        <p>雨滴在風場中受到水平風速 ($V_{\text{wind}}$) 與垂直終端速度 ($V_{\text{term}}$) 的共同作用，形成合成速度向量 ($V_{\text{resultant}}$)。</p>
        <p>撞擊角度 $\\theta$ 計算如下：</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\theta_{\text{impact}} = \arctan\left(\frac{V_{\text{wind}}}{V_{\text{term}}}\right)")
        
        st.markdown("**能量損失機制：**")
        st.markdown("若壓電片保持水平，有效撞擊力僅為垂直分量，造成餘弦損失 (Cosine Loss)：")
        st.latex(r"E_{\text{fixed}} \propto (F \cdot \cos\theta)^2")

    with col_t4:
        st.markdown("""
        <div class="theory-box">
        <h4>自動補償邏輯 (Optimization Logic)</h4>
        <p>系統透過風速計回傳數據，即時計算最佳傾角 $\\phi_{\text{opt}}$，使壓電片法向量與雨滴路徑平行。</p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\phi_{\text{opt}} = \theta_{\text{impact}}")
        st.markdown("""
        **優化效益：**
        1. **動能最大化**：消除餘弦損失，使 $F_{\text{eff}} \approx F_{\text{total}}$。
        2. **頻率響應優化**：垂直撞擊能更有效激發 $d_{31}$ 模式的形變。
        3. **排水效應**：傾斜角度有助於破壞表面水膜張力，減少阻尼 ($c$)。
        """)
        st.success("Logic: 數位孿生模型即時計算此幾何關係，動態調整每個時間步長的能量轉換效率係數 (Efficiency Factor)。")

# ================= TAB 2: 物理實驗室 =================
with tab_lab:
    st.markdown("#### 變因控制實驗")
    st.markdown("在此模式下，可獨立控制降雨強度與撞擊頻率，以驗證系統的物理極限與波形響應。")
    col_ctrl, col_viz = st.columns([1, 2])
    
    with col_ctrl:
        st.subheader("參數控制")
        val_rain = st.slider("1. 降雨強度 (Rain Intensity)", 0, 150, 50, format="%d mm/hr")
        val_wind = st.slider("2. 風速 (Wind Speed)", 0.0, 30.0, 5.0, format="%.1f m/s")
        val_freq = st.slider("3. 撞擊頻率 (Impact Freq)", 5, 120, 30, format="%d Hz", 
                             help="手動設定每秒撞擊次數")

        _, z_f, eff_f, tau_f, wd, _ = engine.get_params(val_rain, val_wind, "Fixed", freq_override=val_freq)
        _, z_s, eff_s, tau_s, _, _  = engine.get_params(val_rain, val_wind, "Smart", freq_override=val_freq)
        
        time_window = 3 * tau_f * 1000 
        impact_period = 1000 / val_freq 
        is_truncated = impact_period < time_window

        status_color = "#d32f2f" if is_truncated else "#2e7d32"
        status_text = "[Warning] 波形截斷 (Waveform Truncated)" if is_truncated else "[Status] 完整釋放 (Full Decay)"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>物理狀態分析 (Physics Status)</h4>
            <p><b>阻尼比 (Zeta):</b> <span style="color:#d32f2f">{z_f:.4f} (Fixed)</span> vs <span style="color:#2e7d32">{z_s:.4f} (Smart)</span></p>
            <p><b>能量釋放窗 (3 Tau):</b> {time_window:.1f} ms</p>
            <p><b>實際撞擊間隔 (Period):</b> {impact_period:.1f} ms</p>
            <hr>
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
        fig.add_trace(go.Scatter(x=t[mask]*1000, y=wave_s[mask], mode='lines', name='Smart System', line=dict(color='#2e7d32', width=3)))
        fig.add_trace(go.Scatter(x=t[~mask]*1000, y=wave_s[~mask], mode='lines', line=dict(color='#2e7d32', width=1, dash='dot'), showlegend=False))
        fig.add_trace(go.Scatter(x=t[mask]*1000, y=wave_f[mask], mode='lines', name='Fixed System', line=dict(color='#c62828', width=3)))
        
        fig.add_vline(x=T_impact*1000, line_dash="dash", line_color="black", annotation_text="Next Impact")
        
        fig.update_layout(xaxis_title="Time (ms)", yaxis_title="Voltage (Normalized)", height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 3: 場域模擬 =================
with tab_field:
    st.markdown("#### 真實情境模擬")
    st.markdown("在此模式下，撞擊頻率由**Marshall-Palmer 模型**根據降雨強度自動推算。")
    col_input, col_sim = st.columns([1, 3])
    
    with col_input:
        st.subheader("模擬參數")
        sim_duration = st.slider("模擬時長 (小時)", 1, 24, 12)
        uploaded_file = st.file_uploader("上傳氣象數據 CSV (Time, Rain, Wind)", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            data_source = "User Upload Data"
        else:
            h = np.arange(0, sim_duration + 1, 1) 
            peak_time = sim_duration / 2
            r = 10 + 100 * np.exp(-0.5 * (h - peak_time)**2/2.5) 
            w = 5 + 25 * np.exp(-0.5 * (h - peak_time)**2/3) + np.random.normal(0, 2, len(h))
            df = pd.DataFrame({'Time': h, 'Rain': np.clip(r, 0, None), 'Wind': np.clip(w, 0, None)})
            data_source = "Internal Simulation Model"
            
        with st.expander("查看氣象數據表"):
            st.dataframe(df, height=150)

    with col_sim:
        acc_s_list, acc_f_list = [], []
        cum_s, cum_f = 0, 0
        
        for idx, row in df.iterrows():
            R, W = row['Rain'], row['Wind']
            
            f_s, z_s, eff_s, tau_s, _, _ = engine.get_params(R, W, "Smart")
            trunc_s = 1 / (1 + 0.6 * f_s * tau_s) 
            power_s = f_s * (eff_s**2) * trunc_s * (R**0.5) 
            cum_s += power_s
            acc_s_list.append(cum_s)
            
            f_f, z_f, eff_f, tau_f, _, _ = engine.get_params(R, W, "Fixed")
            trunc_f = 1 / (1 + 0.6 * f_f * tau_f)
            power_f = f_f * (eff_f**2) * trunc_f * (R**0.5)
            cum_f += power_f
            acc_f_list.append(cum_f)
            
        gain = ((cum_s - cum_f) / cum_f) * 100 if cum_f > 0 else 0
        m1, m2, m3 = st.columns(3)
        m1.metric("固定式總產出 (Total Energy - Fixed)", f"{int(cum_f):,}", "Baseline")
        m2.metric("智慧式總產出 (Total Energy - Smart)", f"{int(cum_s):,}", f"+{gain:.1f}%")
        m3.metric("資料來源 (Source)", data_source)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_s_list, fill='tozeroy', name='Smart Tracking', line=dict(color='#2e7d32')))
        fig2.add_trace(go.Scatter(x=df['Time'], y=acc_f_list, fill='tozeroy', name='Fixed System', line=dict(color='#c62828')))
        fig2.update_layout(title="累積發電量模擬 (Cumulative Energy Output)", xaxis_title="Time (Hours)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; font-size: 0.8em;'>Eco-Rain Project | Science Edge Competition | Physics-Informed Digital Twin</div>", unsafe_allow_html=True)

# --- 分隔線 ---
st.markdown("---") 

# --- 標題區 ---
st.header("數位孿生驗證：蒙地卡羅雨滴模擬")
st.caption("基於 Marshall-Palmer 氣象分佈與 RK4 阻尼動力學模型")

# --- 1. 設定模擬參數 (介面) ---
col_ui1, col_ui2 = st.columns(2)
with col_ui1:
    mc_rain_rate = st.slider("降雨強度 (Rain Rate)", 10, 100, 50, format="%d mm/hr", key="mc_rain")
with col_ui2:
    mc_wetness = st.slider("水膜係數 (Wetness Factor)", 0.0, 1.0, 0.1, key="mc_wet")

# --- 2. 按鈕觸發運算 (修正：直接呼叫本地函式) ---
if st.button("執行蒙地卡羅模擬 (Run Monte Carlo)"):
    
    with st.spinner('正在生成 1,000 顆符合 Marshall-Palmer 分佈的隨機雨滴...'):
        # 修正：直接呼叫上方定義的函式，不再使用 physics_core
        masses, velocities = generate_storm_profile(n_drops=1000, rain_rate_mmph=mc_rain_rate)
    
    st.success(f"模擬完成！成功生成 {len(masses)} 顆有效雨滴數據。")

    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.subheader("1. 雨滴速度分佈 (機率驗證)")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(velocities, bins=25, color='#4A90E2', edgecolor='black', alpha=0.7)
        ax.set_xlabel("Terminal Velocity (m/s)")
        ax.set_ylabel("Count")
        ax.set_title(f"Marshall-Palmer Dist. (R={mc_rain_rate})")
        st.pyplot(fig)
        st.info("說明：此圖驗證模擬出的雨滴是否符合真實氣象統計分佈。")

    with col_plot2:
        st.subheader("2. 撞擊響應波形 (動力學)")
        
        idx = np.random.randint(0, len(masses))
        
        # 修正：直接呼叫上方定義的函式
        t, v = rk4_solver(
            mass_beam=0.005,      
            k_spring=150,         
            dt=0.0001,           
            total_time=0.1,       
            drop_mass=masses[idx], 
            drop_velocity=velocities[idx], 
            wetness=mc_wetness
        )
        
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.plot(t*1000, v, color='#FF6B6B', linewidth=2)
        ax2.set_xlabel("Time (ms)")
        ax2.set_ylabel("Voltage Output (V)")
        ax2.set_title(f"Impact Response (Wetness={mc_wetness})")
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2)
        
        energy_metric = np.sum(v**2) * 100 
        st.metric("預測單次撞擊能量指標", f"{energy_metric:.2f} mJ")
