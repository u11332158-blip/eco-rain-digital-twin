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
        "dev_credit": "Developed for Global Link Singapore 2026",
        "tab_theory": "Theory & Logic",
        "tab_lab": "Physics Lab",
        "tab_field": "Field Simulation",
        "theory_header": "Physics Logic & Models",
        "theory_sec1": "1. Environmental Input Models (Nature)",
        "theory_sec2": "2. System Dynamics (Hardware vs. Nature)",
        "eq1_title": "Eq. 1: Stochastic Rain",
        "eq1_desc": "Marshall-Palmer distribution for raindrop sizes.",
        "eq2_title": "Eq. 2: Terminal Velocity",
        "eq2_desc": "Gunn-Kinzer relation for air resistance correction.",
        "eq3_title": "Eq. 3: Effective Impact Angle",
        "eq3_desc": "Vector synthesis of wind and rain velocity.",
        "eq4_title": "Eq. 4: Piezo-Dynamics",
        "eq4_desc": "2nd-order system with moment arm scaling.",
        "eq5_title": "Eq. 5: Ghost Damping (Nature)",
        "eq5_desc": "Damping spikes up to 0.35 as water film accumulates.",
        "eq6_title": "Eq. 6: Solenoid Limit (Lab)",
        "eq6_desc": "Force decays due to hardware stroke limit at high freq.",
        "lab_ctrl": "Parameter Control",
        "lab_env": "Experiment A: Ghost Damping Effect (Paralysis)",
        "lab_freq_sect": "Experiment B: Solenoid Hardware Limit",
        "lab_sweet_spot": "Set to Sweet Spot",
        "rain_rate": "Rain Rate (mm/hr)",
        "wind_speed": "Wind Speed (m/s)",
        "impact_freq": "Impact Freq (Hz)",
        "solenoid_eff": "Solenoid Stroke Efficiency",
        "field_header": "Real-world Scenario Simulation (Water Film Accumulation)",
        "sim_params": "Simulation Parameters",
        "sim_duration": "Duration (Hours)",
        "view_weather": "View Weather Data",
        "upload_csv": "Upload Weather CSV",
        "use_sim": "Using Generated Simulation Data",
        "use_csv": "Using Uploaded CSV Data",
        "metric_ideal": "Ideal Output (Dry)",
        "metric_real": "Real Output (Water Film)",
        "metric_loss": "Energy Loss",
        "chart_cum_title": "Cumulative Energy: Ideal vs. Real (System Paralysis)",
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
        "dev_credit": "為 Global Link Singapore 2026 開發",
        "tab_theory": "理論架構",
        "tab_lab": "物理實驗室",
        "tab_field": "場域模擬",
        "theory_header": "系統運算邏輯與物理模型",
        "theory_sec1": "1. 環境物理模型 (大自然輸入)",
        "theory_sec2": "2. 系統動力模型 (硬體限制 vs 自然限制)",
        "eq1_title": "Eq. 1: 隨機降雨模型",
        "eq1_desc": "Marshall-Palmer 雨滴粒徑分佈模型。",
        "eq2_title": "Eq. 2: 終端速度修正",
        "eq2_desc": "Gunn-Kinzer 空氣阻力修正公式。",
        "eq3_title": "Eq. 3: 有效撞擊角度",
        "eq3_desc": "風速與雨速的向量合成分析。",
        "eq4_title": "Eq. 4: 壓電動力學",
        "eq4_desc": "二階阻尼系統與力臂效應。",
        "eq5_title": "Eq. 5: 幽靈阻尼 (自然限制)",
        "eq5_desc": "積水導致阻尼比飆升至極限 0.35。",
        "eq6_title": "Eq. 6: 電磁閥限制 (實驗室極限)",
        "eq6_desc": "高頻時因機械行程不足導致撞擊力衰減。",
        "lab_ctrl": "變因控制實驗",
        "lab_env": "實驗 A：水膜阻尼效應 (系統癱瘓測試)",
        "lab_freq_sect": "實驗 B：電磁閥物理限制 (硬體失真)",
        "lab_sweet_spot": "設定為極限甜蜜點",
        "rain_rate": "降雨強度 (mm/hr)",
        "wind_speed": "環境風速 (m/s)",
        "impact_freq": "撞擊頻率 (Hz)",
        "solenoid_eff": "電磁閥行程效率",
        "field_header": "真實情境模擬 (水膜累積效應)",
        "sim_params": "模擬參數",
        "sim_duration": "模擬時長 (小時)",
        "view_weather": "查看氣象數據",
        "upload_csv": "上傳氣象 CSV 檔",
        "use_sim": "使用系統生成模擬數據",
        "use_csv": "使用上傳的 CSV 數據",
        "metric_ideal": "理想環境總產出",
        "metric_real": "真實積水總產出",
        "metric_loss": "能量流失率",
        "chart_cum_title": "累積發電量：理想環境 vs 系統癱瘓 (Zero Usable Power)",
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
        "dev_credit": "Global Link Singapore 2026 向け開発",
        "tab_theory": "理論とロジック",
        "tab_lab": "物理実験室",
        "tab_field": "シミュレーション",
        "theory_header": "物理ロジックとモデル",
        "theory_sec1": "1. 環境入力モデル",
        "theory_sec2": "2. システムダイナミクス",
        "eq1_title": "Eq. 1: 確率降雨モデル",
        "eq1_desc": "Marshall-Palmer 雨滴粒径分布。",
        "eq2_title": "Eq. 2: 終端速度補正",
        "eq2_desc": "Gunn-Kinzer 空気抵抗補正。",
        "eq3_title": "Eq. 3: 有効衝突角度",
        "eq3_desc": "風速と雨速のベクトル合成。",
        "eq4_title": "Eq. 4: 圧電ダイナミクス",
        "eq4_desc": "二次減衰系とモーメントアーム効果。",
        "eq5_title": "Eq. 5: ゴースト減衰",
        "eq5_desc": "水膜による減衰比の急増 (最大0.35)。",
        "eq6_title": "Eq. 6: ソレノイド限界",
        "eq6_desc": "高周波時のストローク不足による力減衰。",
        "lab_ctrl": "パラメータ制御",
        "lab_env": "実験 A：水膜減衰による麻痺",
        "lab_freq_sect": "実験 B：ソレノイド物理限界",
        "lab_sweet_spot": "最適周波数に設定",
        "rain_rate": "降雨強度 (mm/hr)",
        "wind_speed": "風速 (m/s)",
        "impact_freq": "衝突周波数 (Hz)",
        "solenoid_eff": "ソレノイド効率",
        "field_header": "実環境シミュレーション (水膜蓄積)",
        "sim_params": "パラメータ",
        "sim_duration": "時間 (Hours)",
        "view_weather": "データ表示",
        "upload_csv": "CSVアップロード",
        "use_sim": "シミュレーションデータを使用",
        "use_csv": "アップロードデータを使用",
        "metric_ideal": "理想システム出力",
        "metric_real": "現実システム出力",
        "metric_loss": "エネルギー損失",
        "chart_cum_title": "累積発電量: 理想 vs 現実 (システム麻痺)",
        "unit_energy": "mJ",
        "sim_start_btn": "モンテカルロ法を実行",
        "sim_success": "{n} 個のデータを生成。"
    }
}

# ==========================================
# 2. 物理常數定義區 (Physical Config)
# ==========================================
class PhysConfig:
    PIEZO_SENSITIVITY_V_PM = 50000.0  
    IMPACT_DURATION_SEC = 0.002       
    
    # 完美對齊摘要的關鍵數值
    DAMPING_RATIO_DRY = 0.02          
    DAMPING_RATIO_WET_MAX = 0.35      
    
    SATURATION_RAIN_RATE = 120.0      
    BASE_POWER_FACTOR = 0.5           
    TRUNCATION_SHAPE_FACTOR = 0.6     

# ==========================================
# 3. 主程式 (Main App)
# ==========================================
st.set_page_config(page_title="Eco-Rain Digital Twin", layout="wide")

st.markdown("""
<style>
    .metric-card { background-color: #f5f5f5 !important; border: 1px solid #e0e0e0; border-radius: 5px; padding: 15px; border-left: 5px solid #2e7d32; margin-bottom: 10px; }
    .theory-box { background-color: #ffffff !important; padding: 20px; border-radius: 8px; border: 1px solid #ddd; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .theory-box h4 { color: #1565c0 !important; font-weight: bold; margin-bottom: 10px; }
    .citation-box { background-color: #fff3e0 !important; padding: 15px; border-radius: 5px; border-left: 5px solid #ff9800; font-size: 0.9em; margin-top: 20px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("### Language / 語言 / 言語")
selected_lang = st.sidebar.selectbox("Select Language", ["English", "繁體中文", "日本語"], label_visibility="collapsed")
t = TRANSLATIONS[selected_lang] 

st.title(t["title"])
st.caption("Physics-Informed Digital Twin Platform")
st.sidebar.markdown(f"### {t['sidebar_settings']}")
st.sidebar.info("TE Connectivity LDT0-028K (PVDF)")

param_beam_len = st.sidebar.number_input(t["beam_len"], 3.0, 10.0, 5.0, step=0.5)
param_area = st.sidebar.number_input(t["area"], 0.5, 10.0, 2.5, format="%.1f")
param_fn = st.sidebar.number_input(t["freq"], 50, 200, 100, format="%d")

st.sidebar.markdown("---")
st.sidebar.text(t["dev_credit"])

tab_theory, tab_lab, tab_field = st.tabs([t["tab_theory"], t["tab_lab"], t["tab_field"]])

# ================= TAB 1: 理論架構 =================
with tab_theory:
    st.header(t["theory_header"])
    st.markdown("---")
    
    st.subheader(t["theory_sec1"])
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(f'<div class="theory-box"><h4>{t["eq1_title"]}</h4><p>{t["eq1_desc"]}</p></div>', unsafe_allow_html=True)
        st.latex(r"N(D) = N_0 e^{-\Lambda D}")
        st.markdown(f'<div class="theory-box"><h4>{t["eq2_title"]}</h4><p>{t["eq2_desc"]}</p></div>', unsafe_allow_html=True)
        st.latex(r"V_{term}(D) = 9.65 - 10.3 e^{-0.6D}")
    with col_t2:
        st.markdown(f'<div class="theory-box"><h4>{t["eq3_title"]}</h4><p>{t["eq3_desc"]}</p></div>', unsafe_allow_html=True)
        st.latex(r"\theta_{eff} = \arctan\left(\frac{V_{wind}}{V_{term}}\right)")

    st.markdown("---")
    st.subheader(t["theory_sec2"])
    col_t3, col_t4 = st.columns(2)
    with col_t3:
        st.markdown(f'<div class="theory-box"><h4>{t["eq4_title"]}</h4><p>{t["eq4_desc"]}</p></div>', unsafe_allow_html=True)
        st.latex(r"m_{eff} \ddot{x} + c \dot{x} + k x = F(t) \cdot \left(\frac{x_{pos}}{L}\right)^2")
        st.markdown(f'<div class="theory-box"><h4>{t["eq5_title"]}</h4><p>{t["eq5_desc"]}</p></div>', unsafe_allow_html=True)
        st.latex(r"\zeta(t) = \zeta_{dry} + \kappa \cdot h_{film}(t)")
    with col_t4:
        st.markdown(f'<div class="theory-box"><h4>{t["eq6_title"]}</h4><p>{t["eq6_desc"]}</p></div>', unsafe_allow_html=True)
        st.latex(r"F_{eff}(f) = F_{max} \cdot \left(\frac{33.3}{f}\right)^{1.5}")

# ================= TAB 2: 物理實驗室 =================
with tab_lab:
    st.markdown(f"#### {t['lab_ctrl']}")
    st.markdown("---")

    st.markdown(f"##### {t['lab_env']}")
    col_a1, col_a2 = st.columns([1, 2])
    with col_a1:
        val_rain_a = st.slider(f"{t['rain_rate']}", 0, 150, 150, key="exp_a_rain")
        
        # 簡單計算用於波形顯示
        z_ideal = PhysConfig.DAMPING_RATIO_DRY
        wetness = min(1.0, val_rain_a / PhysConfig.SATURATION_RAIN_RATE)
        z_real = PhysConfig.DAMPING_RATIO_DRY + (PhysConfig.DAMPING_RATIO_WET_MAX - PhysConfig.DAMPING_RATIO_DRY) * wetness
        
        st.info(f"**Zeta Comparison:**\n* **Ideal (Green):** `{z_ideal:.3f}`\n* **Real (Red):** `{z_real:.3f}`")

    with col_a2:
        t_arr = np.linspace(0, 0.2, 2000)
        wd_ideal = 2 * np.pi * param_fn * np.sqrt(1 - z_ideal**2)
        wd_real = 2 * np.pi * param_fn * np.sqrt(1 - z_real**2)
        
        # 水膜吸收衝擊力
        impulse_real = 1.0 * (1 - (0.8 * wetness)) 
        
        wave_ideal = 1.0 * np.exp(-z_ideal * 2 * np.pi * param_fn * t_arr) * np.sin(wd_ideal * t_arr)
        wave_real = impulse_real * np.exp(-z_real * 2 * np.pi * param_fn * t_arr) * np.sin(wd_real * t_arr)
        
        fig_a = go.Figure()
        fig_a.add_trace(go.Scatter(x=t_arr*1000, y=wave_ideal, mode='lines', name='Ideal (Dry)', line=dict(color='#2e7d32', width=2)))
        fig_a.add_trace(go.Scatter(x=t_arr*1000, y=wave_real, mode='lines', name='Real (Water Film)', line=dict(color='#c62828', width=3)))
        fig_a.update_layout(title=f"Fig 2: System Paralysis (Rain: {val_rain_a} mm/hr)", height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("---")
    st.markdown(f"##### {t['lab_freq_sect']}")
    
    if 'freq_b' not in st.session_state:
        st.session_state.freq_b = 30

    def set_optimal_freq():
        st.session_state.freq_b = 33

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        optimal_freq = 33.3
        st.button(f"{t['lab_sweet_spot']} (33.3 Hz)", on_click=set_optimal_freq)
        val_freq_b = st.slider(f"{t['impact_freq']}", 5, 120, key="freq_b")

        if val_freq_b <= optimal_freq:
            solenoid_eff = 1.0
            st.success("✅ Full Force (Perfect Stroke)")
        else:
            solenoid_eff = (optimal_freq / val_freq_b) ** 1.5
            st.error("⚠️ Force Drop (Stroke Limitation)")

        st.metric(label=t["solenoid_eff"], value=f"{solenoid_eff*100:.1f}%")

    with col_b2:
        t_arr_b = np.linspace(0, 0.1, 1000)
        wave_ghost = 1.0 * np.exp(-0.02 * 2 * np.pi * param_fn * t_arr_b) * np.sin(wd_ideal * t_arr_b)
        T_impact_limit = 1 / val_freq_b
        wave_actual = wave_ghost * solenoid_eff 
        wave_viz = np.where(t_arr_b <= T_impact_limit, wave_actual, None)

        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(x=t_arr_b*1000, y=wave_ghost, mode='lines', name='Perfect Electronic Signal', line=dict(color='gray', width=2, dash='dot'), opacity=0.5))
        fig_b.add_trace(go.Scatter(x=t_arr_b*1000, y=wave_viz, mode='lines', name='Actual Hardware Stroke', line=dict(color='#1565c0', width=3), fill='tozeroy'))
        fig_b.add_vline(x=T_impact_limit*1000, line_dash="solid", line_color="#d32f2f")
        fig_b.update_layout(title=f"Fig 3: Hardware Distortion @ {val_freq_b} Hz", xaxis_title="Time (ms)", yaxis_title="Voltage (V)", height=350, margin=dict(l=20, r=20, t=40, b=20), yaxis=dict(range=[-1.2, 1.2]))
        st.plotly_chart(fig_b, use_container_width=True)

# ================= TAB 3: 場域模擬 (無馬達、證明癱瘓的新邏輯) =================
with tab_field:
    st.markdown(f"#### {t['field_header']}")
    col_input, col_sim = st.columns([1, 3])
    
    with col_input:
        sim_duration = st.slider(t["sim_duration"], 1, 24, 16) 
        h = np.arange(0, sim_duration + 1, 1) 
        peak_time = sim_duration / 2
        r_base = 100 * np.exp(-0.5 * ((h - peak_time) / (sim_duration / 4))**2) 
        r = np.clip(r_base + np.random.normal(0, 5, len(h)), 0, None)
        r[r < 5.0] = 0.0  
        df = pd.DataFrame({'Time': h, 'Rain': r})
        st.info("Continuous Heavy Rain Profile")

    with col_sim:
        if df is not None:
            acc_ideal_list, acc_real_list = [], []
            cum_ideal, cum_real = 0, 0
            water_film_thickness = 0.0 # 水膜累積變數
            
            for idx, row in df.iterrows():
                R = row['Rain']
                
                # 1. 計算理想狀態 (永遠保持 0.02 阻尼)
                energy_i = R * PhysConfig.BASE_POWER_FACTOR * 1.5 
                cum_ideal += energy_i
                
                # 2. 計算真實狀態 (水膜累積死亡方程式)
                if R > 5:
                    water_film_thickness += (R / 100.0) * 0.25 # 大雨時累積
                else:
                    water_film_thickness -= 0.15 # 雨停慢慢乾
                water_film_thickness = max(0.0, min(1.0, water_film_thickness))
                
                # 死亡方程式核心: 阻尼飆升 + 衝擊力被水吸走
                current_damping = PhysConfig.DAMPING_RATIO_DRY + (PhysConfig.DAMPING_RATIO_WET_MAX - PhysConfig.DAMPING_RATIO_DRY) * water_film_thickness
                sponge_absorption = 1.0 - (0.85 * water_film_thickness) # 最多吸收 85% 力量
                
                # 發電量因為高阻尼和海綿效應被強制壓平
                energy_r = energy_i * sponge_absorption * (PhysConfig.DAMPING_RATIO_DRY / current_damping)
                cum_real += energy_r
                
                acc_ideal_list.append(cum_ideal * 10)
                acc_real_list.append(cum_real * 10)
            
            loss_percentage = ((acc_ideal_list[-1] - acc_real_list[-1]) / acc_ideal_list[-1]) * 100 if acc_ideal_list[-1] > 0 else 0
            
            m1, m2, m3 = st.columns(3)
            m1.metric(t["metric_ideal"], f"{int(acc_ideal_list[-1]):,} {t['unit_energy']}", "Baseline")
            m2.metric(t["metric_real"], f"{int(acc_real_list[-1]):,} {t['unit_energy']}", "System Paralysis")
            m3.metric(t["metric_loss"], f"-{loss_percentage:.1f}%", "Zero Usable Power")
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df['Time'], y=acc_ideal_list, name='Ideal (Dry 0.02)', fill='tozeroy', line=dict(color='#2e7d32', width=3)))
            fig2.add_trace(go.Scatter(x=df['Time'], y=acc_real_list, name='Real (Water Film 0.35)', fill='tozeroy', line=dict(color='#c62828', width=3)))
            
            fig2.update_layout(title=t["chart_cum_title"], height=350, margin=dict(l=0,r=0,t=30,b=0), yaxis_title="Energy (mJ)")
            st.plotly_chart(fig2, use_container_width=True)
