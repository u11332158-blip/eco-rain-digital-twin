# 檔案名稱：physics_core.py
import numpy as np

# --- 1. Marshall-Palmer 雨滴生成器 (蒙地卡羅核心) ---
def generate_storm_profile(n_drops, rain_rate_mmph):
    """
    輸入：雨滴數量, 降雨強度 (mm/hr)
    輸出：符合自然分佈的雨滴質量 (kg) 與 速度 (m/s) 陣列
    """
    # 計算分佈參數 Lambda (依據 Marshall-Palmer 公式)
    lambda_param = 4.1 * (rain_rate_mmph ** -0.21)
    
    # 使用指數分佈隨機生成直徑
    diameters_mm = np.random.exponential(scale=1.0/lambda_param, size=n_drops)
    
    # 物理過濾：移除不合理的霧氣(<0.5mm)與過大雨滴(>6mm)
    valid_mask = (diameters_mm > 0.5) & (diameters_mm < 6.0)
    valid_diameters = diameters_mm[valid_mask]
    
    # 計算質量 (kg)
    masses_kg = 1000 * (4/3) * np.pi * ((valid_diameters / 2000)**3)
    
    # 計算終端速度 (m/s) - Gunn-Kinzer 經驗公式
    velocities_ms = 9.65 - 10.3 * np.exp(-0.6 * valid_diameters)
    
    return masses_kg, velocities_ms

# --- 2. RK4 物理求解器 (動力學核心) ---
def rk4_solver(mass_beam, k_spring, dt, total_time, drop_mass, drop_velocity, wetness):
    """
    輸入：懸臂樑參數, 時間步長, 雨滴參數, 濕度係數
    輸出：時間陣列, 電壓陣列
    """
    steps = int(total_time / dt)
    time = np.linspace(0, total_time, steps)
    voltage = np.zeros(steps)
    
    x = 0.0
    # 簡單動量守恆估算初速
    v = drop_velocity * (drop_mass / (mass_beam + drop_mass)) 
    
    # 狀態相依阻尼 (State-Dependent Damping)
    zeta = 0.045 + (wetness * 0.275)
    omega_n = np.sqrt(k_spring / (mass_beam + drop_mass))
    
    for i in range(1, steps):
        def get_accel(v_curr, x_curr):
            return -(2 * zeta * omega_n * v_curr) - (omega_n**2 * x_curr)
        
        k1_v = get_accel(v, x)
        k1_x = v
        
        k2_v = get_accel(v + 0.5*dt*k1_v, x + 0.5*dt*k1_x)
        k2_x = v + 0.5*dt*k1_v
        
        k3_v = get_accel(v + 0.5*dt*k2_v, x + 0.5*dt*k2_x)
        k3_x = v + 0.5*dt*k2_v
        
        k4_v = get_accel(v + dt*k3_v, x + dt*k3_x)
        k4_x = v + dt*k3_v
        
        v = v + (dt/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)
        x = x + (dt/6)*(k1_x + 2*k2_x + 2*k3_x + k4_x)
        
        # 假設壓電轉換係數 (Piezo Coefficient) 放大顯示用
        voltage[i] = x * 8500000 
        
    return time, voltage
