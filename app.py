"""Eco-Rain 4.1 standalone — run: python -m streamlit run app.py
All project code and six original dry CSVs are embedded in this file.
Install third-party packages using requirements.txt; no physics_core package needed.
Model assumptions and numerical methods are unchanged from version 4.0.
"""


"""Dry measured-response superposition and rainfall events, in SI units.

No material damping ratio, wetting law, force-frequency law or electrical
conversion efficiency is inferred from the supplied unlabelled impacts.
"""
from dataclasses import dataclass
from functools import lru_cache
import csv
import io
import math
import numpy as np

MAX_SAMPLES = 1_000_001
MAX_EVENTS = 1_000_000
WATER_DENSITY = 1000.0


@dataclass
class ScopeTrace:
    name: str
    time_s: np.ndarray
    voltage_V: np.ndarray
    metadata: dict

    @property
    def dt(self):
        return float(np.median(np.diff(self.time_s)))


def _finite(name, values):
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name}: values must be finite.")


def read_scope_csv(data: bytes, name="uploaded.csv") -> ScopeTrace:
    """Parse oscilloscope CSV or plain time_s,voltage_V CSV; preserve V scaling."""
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    metadata, points, started = {}, [], False
    has_marker = any(row and row[0].strip() == "Waveform Data" for row in rows)
    for index, row in enumerate(rows, 1):
        if not row or not any(x.strip() for x in row):
            continue
        if row[0].strip() == "Waveform Data":
            started = True
            continue
        if has_marker and not started:
            metadata[row[0].strip()] = row[1].strip() if len(row) > 1 else ""
            continue
        if not has_marker and not points and row[0].strip().lower() in {"time_s", "time", "t"}:
            continue
        try:
            points.append((float(row[0]), float(row[1])))
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{name}: invalid waveform at CSV row {index}.") from exc
    if len(points) < 50 or len(points) > MAX_SAMPLES:
        raise ValueError(f"{name}: waveform requires 50–{MAX_SAMPLES} samples.")
    a = np.asarray(points, dtype=float)
    _finite(name, a)
    t, v = a.T
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{name}: time must increase strictly, in seconds.")
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=.002, atol=1e-10):
        raise ValueError(f"{name}: sample times must be uniformly spaced.")
    if has_marker:
        if metadata.get("Vertical Units") != "V" or metadata.get("Horizontal Units", "").upper() != "S":
            raise ValueError(f"{name}: expected volts and seconds.")
        if int(metadata.get("Memory Length", len(t))) != len(t):
            raise ValueError(f"{name}: memory length does not match data.")
        if not np.isclose(float(metadata.get("Sampling Period", dt)), dt, rtol=.002):
            raise ValueError(f"{name}: sampling-period metadata mismatch.")
    return ScopeTrace(name, t, v, metadata)


def prepare_kernel(trace, threshold_V=1.0, pre_s=.005, tail_s=.250, taper_s=.020):
    """A finite voltage template, not a unit-force impulse response.

    t=0 is 5 ms before the first threshold crossing by default; no contact-time
    measurement is implied. Median baseline subtraction and a final 20 ms
    cosine taper remove an artificial hard jump at the end of the template.
    """
    _finite("template settings", [threshold_V, pre_s, tail_s, taper_s])
    if threshold_V <= 0 or pre_s < 0 or tail_s <= taper_s or taper_s < 0:
        raise ValueError("Require threshold > 0, pre >= 0, tail > taper >= 0.")
    t, v = trace.time_s, trace.voltage_V
    base = float(np.median(v[t < t[0] + min(.1, (t[-1]-t[0])/5)]))
    hits = np.flatnonzero(np.abs(v-base) > threshold_V)
    if len(hits) == 0:
        raise ValueError("No signal exceeds the alignment threshold; lower the threshold.")
    onset = int(hits[0])
    start = max(0, onset-int(round(pre_s/trace.dt)))
    end = int(np.searchsorted(t, t[onset]+tail_s, side="right"))
    if end >= len(t):
        raise ValueError("The trace does not contain the requested post-impact tail.")
    tt, vv = t[start:end]-t[start], v[start:end].copy()-base
    if len(tt) < 50 or np.max(np.abs(vv)) == 0:
        raise ValueError("The selected template has insufficient signal.")
    # Make the event start continuous without changing the measured main pulse.
    vv[0] = 0.0
    if taper_s > 0:
        edge = tt >= tt[-1]-taper_s
        phase = np.clip((tt[edge]-(tt[-1]-taper_s))/taper_s, 0, 1)
        vv[edge] *= .5*(1+np.cos(np.pi*phase))
    vv[-1] = 0.0
    return tt, vv, {"baseline_V": base, "threshold_V": threshold_V,
        "source_start_s": float(t[start]), "threshold_source_time_s": float(t[onset]),
        "tail_s": tail_s, "taper_s": taper_s, "state": "dry"}


def time_grid(duration_s, max_dt_s, start_s=0.0):
    _finite("time settings", [duration_s, max_dt_s, start_s])
    if duration_s <= 0 or max_dt_s <= 0:
        raise ValueError("Duration and time step must be positive.")
    count = int(np.ceil(duration_s/max_dt_s))
    if count+1 > MAX_SAMPLES:
        raise ValueError("Too many waveform samples; shorten the waveform window.")
    return start_s + np.arange(count+1, dtype=float)*(duration_s/count)


def periodic_events(frequency_hz, duration_s):
    _finite("periodic events", [frequency_hz, duration_s])
    if frequency_hz < 0 or duration_s <= 0:
        raise ValueError("Require frequency >= 0 and duration > 0.")
    if frequency_hz == 0:
        return np.array([], dtype=float)
    count = int(np.ceil(frequency_hz*duration_s))
    if count > MAX_EVENTS:
        raise ValueError("Too many impact events; shorten the duration.")
    events = np.arange(count, dtype=float)/frequency_hz
    return events[events < duration_s]


def superpose(kernel_t, kernel_v, event_t, gains, duration_s, dt_s, start_s=0.0):
    """Sum shifted templates, including impacts before the visible window.

    Direct interpolation respects fractional event times, without resetting or
    cutting an existing response when another event arrives.
    """
    kt, kv, events, gains = [np.asarray(x, dtype=float) for x in (kernel_t, kernel_v, event_t, gains)]
    for name, arr in [("kernel time", kt), ("kernel voltage", kv), ("events", events), ("gains", gains)]:
        _finite(name, arr)
        if arr.ndim != 1:
            raise ValueError(f"{name}: expected one-dimensional array.")
    if len(kt) < 2 or kv.shape != kt.shape or events.shape != gains.shape:
        raise ValueError("Kernel and event array lengths must match.")
    if kt[0] != 0 or np.any(np.diff(kt) <= 0):
        raise ValueError("Kernel time must start at zero and strictly increase.")
    if len(events) > MAX_EVENTS:
        raise ValueError("Too many events.")
    t = time_grid(duration_s, dt_s, start_s)
    out = np.zeros_like(t)
    visible = (events <= t[-1]) & (events+kt[-1] >= t[0]) & (gains != 0)
    for event, gain in zip(events[visible], gains[visible]):
        lo = int(np.searchsorted(t, event, side="left"))
        hi = int(np.searchsorted(t, event+kt[-1], side="right"))
        if hi > lo:
            out[lo:hi] += gain*np.interp(t[lo:hi]-event, kt, kv, left=0, right=0)
    return t, out


def squared_signal_integral(t, signal):
    """Cumulative trapezoid integral; for voltage the units are V² s."""
    t, signal = np.asarray(t, float), np.asarray(signal, float)
    _finite("signal integral", t); _finite("signal integral", signal)
    if t.ndim != 1 or t.shape != signal.shape or len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("Signal and increasing time must match.")
    return np.r_[0., np.cumsum(.5*(signal[1:]**2+signal[:-1]**2)*np.diff(t))]


def resistive_energy(t, voltage, resistance_ohm):
    _finite("resistance", resistance_ohm)
    if resistance_ohm <= 0:
        raise ValueError("Resistance must be positive.")
    return squared_signal_integral(t, voltage)/resistance_ohm


def window_energy_budget(t, voltage, resistance_ohm, events_s, incident_energy_J, support_s):
    """Necessary, not sufficient, passivity bound for an empirical rain window.

    Count the FULL incident energy of every event whose template intersects the
    window, including events before its start. This is a loose upper bound,
    not an energy-balance identity or proof of validation.
    """
    t=np.asarray(t,float)
    events_s,incident_energy_J=np.asarray(events_s,float),np.asarray(incident_energy_J,float)
    _finite("incident energy",incident_energy_J);_finite("events",events_s)
    _finite("support",support_s)
    if events_s.shape!=incident_energy_J.shape or events_s.ndim!=1 or np.any(incident_energy_J<0) or support_s<=0:
        raise ValueError("Event energies must be nonnegative and match event times; support must be positive.")
    energy=float(resistive_energy(t,voltage,resistance_ohm)[-1])
    contributing=(events_s<=t[-1])&(events_s+support_s>=t[0])
    upper=float(incident_energy_J[contributing].sum())
    return {"load_energy_J":energy,"incident_upper_bound_J":upper,
        "within_incident_upper_bound":bool(energy<=upper*(1+1e-6)+1e-15)}


def terminal_speed(diameter_mm):
    """Atlas-type terminal-speed approximation; diameters in mm."""
    d = np.asarray(diameter_mm, float)
    _finite("diameter", d)
    if np.any((d < .5) | (d > 6.0)):
        raise ValueError("Drop diameter is restricted to 0.5–6.0 mm.")
    return 9.65-10.3*np.exp(-.6*d)


@lru_cache(maxsize=2048)
def drop_distribution(rain_rate_mm_h):
    """Speed-weighted truncated MP shape, volume-normalized to target rainfall.

    This deliberately is NOT fixed-N0 Marshall–Palmer: all target rainfall is
    assigned to the declared 0.5–6 mm interval. Normalization is explicit.
    """
    _finite("rain rate", rain_rate_mm_h)
    if rain_rate_mm_h <= 0:
        raise ValueError("Distribution requires positive rain rate.")
    d = np.linspace(.5, 6., 4097)
    lam = 4.1*rain_rate_mm_h**(-.21)
    weight = terminal_speed(d)*np.exp(-lam*(d-d[0]))
    cdf = np.r_[0., np.cumsum(.5*(weight[1:]+weight[:-1])*np.diff(d))]
    norm = cdf[-1]
    volume = np.pi/6*(d*1e-3)**3
    wv = weight*volume
    mean_volume = np.sum(.5*(wv[1:]+wv[:-1])*np.diff(d))/norm
    return d, cdf/norm, float(mean_volume)


def read_weather_csv(data):
    """Piecewise-constant rain, with the last row as the end boundary.

    Supported headers: Time,Rain (hours,mm/h); time_h,rain_mm_h; or
    time_s,rain_mm_h. Blank, nonfinite, unsorted or negative data are rejected.
    """
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    headers = reader.fieldnames or []
    if "time_h" in headers: time_col, factor = "time_h", 3600.
    elif "time_s" in headers: time_col, factor = "time_s", 1.
    elif "Time" in headers: time_col, factor = "Time", 3600.
    else: raise ValueError("CSV requires time_h, time_s, or Time (hours).")
    rain_col = "rain_mm_h" if "rain_mm_h" in headers else "Rain"
    if rain_col not in headers:
        raise ValueError("CSV requires rain_mm_h or Rain (mm/h).")
    try:
        a = np.asarray([(float(row[time_col])*factor, float(row[rain_col])) for row in reader], float)
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("Weather CSV must contain numeric time and rain values.") from exc
    if a.ndim != 2 or len(a) < 2 or len(a) > 2001:
        raise ValueError("Weather CSV requires 2–2001 rows, including the end boundary.")
    _finite("weather CSV", a)
    if a[0, 0] < 0 or np.any(np.diff(a[:,0]) <= 0) or np.any(a[:,1] < 0):
        raise ValueError("Time must be nonnegative and strictly increasing; rain cannot be negative.")
    return a[:,0]-a[0,0], a[:-1,1]


def rain_events(boundaries_s, rates_mm_h, area_m2, seed=0):
    """Independent Poisson arrivals onto a horizontal surface, no wetting law."""
    bounds, rates = np.asarray(boundaries_s, float), np.asarray(rates_mm_h, float)
    _finite("rain boundaries", bounds); _finite("rain rates", rates); _finite("area", area_m2)
    if bounds.ndim != 1 or rates.ndim != 1 or len(bounds) != len(rates)+1 or len(rates) == 0:
        raise ValueError("Provide one more time boundary than interval rain rates.")
    if bounds[0] != 0 or np.any(np.diff(bounds) <= 0) or np.any(rates < 0) or area_m2 <= 0:
        raise ValueError("Require time starting at zero and increasing, rain >= 0, area > 0.")
    rng = np.random.default_rng(seed)
    expected = []
    for dt, rate in zip(np.diff(bounds), rates):
        mean = drop_distribution(float(rate))[2] if rate > 0 else 1.
        expected.append(float(rate/3.6e6*area_m2*dt/mean))
    if sum(expected) > MAX_EVENTS:
        raise ValueError("Expected drop count exceeds 1,000,000; reduce area, time or rainfall.")
    counts = rng.poisson(expected)
    if counts.sum() > MAX_EVENTS:
        raise ValueError("Sampled drop count exceeds 1,000,000; use a shorter scenario.")
    all_t, all_d = [], []
    interval_volume, interval_ke = [], []
    for lo, hi, rate, count in zip(bounds[:-1], bounds[1:], rates, counts):
        if count:
            d, cdf, _ = drop_distribution(float(rate))
            diams = np.interp(rng.random(count), cdf, d)
            times = rng.uniform(lo, hi, count)
            order = np.argsort(times)
            diams, times = diams[order], times[order]
            volume = np.pi/6*(diams*1e-3)**3
            interval_volume.append(float(volume.sum()))
            interval_ke.append(float(np.sum(.5*WATER_DENSITY*volume*terminal_speed(diams)**2)))
            all_t.append(times); all_d.append(diams)
        else:
            interval_volume.append(0.); interval_ke.append(0.)
    times = np.concatenate(all_t) if all_t else np.array([], float)
    diams = np.concatenate(all_d) if all_d else np.array([], float)
    speeds = terminal_speed(diams)
    volumes = np.pi/6*(diams*1e-3)**3
    mass = WATER_DENSITY*volumes
    duration = bounds[-1]
    return {"time_s": times, "diameter_mm": diams, "mass_kg": mass,
        "speed_m_s": speeds, "incident_momentum_Ns": mass*speeds,
        "incident_kinetic_energy_J": .5*mass*speeds**2,
        "interval_counts": counts, "interval_volume_m3": np.array(interval_volume),
        "interval_incident_energy_J": np.array(interval_ke),
        "expected_count": float(sum(expected)),
        "target_volume_m3": float(np.sum(rates/3.6e6*area_m2*np.diff(bounds))),
        "realized_volume_m3": float(volumes.sum()),
        "realized_mean_mm_h": float(volumes.sum()/area_m2/duration*3.6e6),
        "seed": int(seed), "duration_s": float(duration)}


"""Uncalibrated legacy water-film sensitivity model; outputs are not joules."""
import numpy as np

def simulate(hours=16., rain=50., dry=.02, wet=.35, attenuation=.85,
             buildup=.25, drainage=.15, step=.02):
    values=np.array([hours,rain,dry,wet,attenuation,buildup,drainage,step])
    if not np.all(np.isfinite(values)) or hours<=0 or rain<0 or dry<=0 or wet<=0 or not 0<=attenuation<=1 or min(buildup,drainage)<0 or step<=0:
        raise ValueError('Invalid scenario parameters')
    n=int(np.ceil(hours/step))
    if n>200000: raise ValueError('Too many time steps')
    t=np.linspace(0,hours,n+1); dt=hours/n
    w=np.zeros(n+1); baseline=np.zeros(n+1); output=np.zeros(n+1)
    for i in range(n):
        rate=buildup*rain/100 if rain>5 else -drainage
        w[i+1]=np.clip(w[i]+rate*dt,0,1)
        midpoint=np.clip(w[i]+rate*dt/2,0,1)
        z=dry+(wet-dry)*midpoint
        baseline[i+1]=baseline[i]+rain*dt
        output[i+1]=output[i]+rain*dt*(1-attenuation*midpoint)*dry/z
    return t,w,baseline,output

def render_water_scenario(tr):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import json
    st.subheader(tr('水膜：假設情境分析','Water film: assumption sensitivity','水膜：仮定の感度分析'))
    st.info(tr('此頁保留舊程式的經驗公式，用於探索假設。參數未以濕潤實驗校正；濕潤程度不是毫米，輸出不是電能，也未從六份乾燥 CSV 推得。',
        'Legacy empirical sensitivity model. No wet calibration: wetness is not millimetres, output is not energy, and coefficients are not inferred from the six dry CSVs.',
        '旧経験式の感度分析です。湿潤実測による校正なし。湿潤度はmmではなく、出力は電力量ではありません。'))
    a,b=st.columns(2)
    with a:
        hours=st.number_input(tr('時長（小時）','Duration (h)','時間 (h)'),.1,24.,16.,key='wet_hours')
        rain=st.number_input(tr('固定雨強（mm/h）','Constant rain (mm/h)','一定降雨 (mm/h)'),0.,200.,50.,key='wet_rain')
        dry=st.number_input(tr('假設乾燥阻尼比','Assumed dry damping ratio','仮定乾燥減衰比'),.001,.9,.02,format='%.3f',key='wet_dry')
        wet=st.number_input(tr('假設全濕阻尼比','Assumed saturated damping ratio','仮定湿潤減衰比'),.001,.9,.35,format='%.3f',key='wet_damping')
    with b:
        attenuation=st.slider(tr('全濕額外輸出折減係數','Additional saturated output reduction','湿潤時追加低減係数'),0.,1.,.85,key='wet_attenuation')
        buildup=st.number_input(tr('100 mm/h 時濕潤增加率（每小時）','Wetness increase at 100 mm/h (per h)','100 mm/h 時の増加率 (/h)'),0.,5.,.25,key='wet_buildup')
        drainage=st.number_input(tr('雨強 ≤5 時濕潤減少率（每小時）','Wetness decrease at rain ≤5 (per h)','降雨 ≤5 時の減少率 (/h)'),0.,5.,.15,key='wet_drainage')
    st.latex(r'w\in[0,1],\quad\zeta(w)=\zeta_d+(\zeta_w-\zeta_d)w,\quad g(w)=(1-aw)\zeta_d/\zeta(w)')
    st.caption(tr('預設數字取自舊程式，不是文獻量測值。此版依實際時間步長更新，以相同雨強比較乾濕指標；沒有建立存水量守恆、附加質量或機電耦合。',
        'Defaults are legacy assumptions, not published measurements. Updates use elapsed time; both scenarios share rainfall. No water mass balance, added-mass or electromechanical model is claimed.',
        '初期値は旧コードの仮定です。時間刻みを使用。同じ降雨で比較します。水量保存・付加質量・電気機械連成モデルではありません。'))
    t,w,base,out=simulate(hours,rain,dry,wet,attenuation,buildup,drainage)
    fig=go.Figure()
    fig.add_scatter(x=t,y=base,name='Dry assumption')
    fig.add_scatter(x=t,y=out,name='Wet assumption')
    fig.update_layout(xaxis_title='Time (h)',yaxis_title='Cumulative index (arbitrary units)')
    st.plotly_chart(fig,use_container_width=True)
    st.metric(tr('相對基準指標變化','Index change from baseline','基準からの変化'),f'{(out[-1]/base[-1]-1)*100:+.2f}%' if base[-1]>0 else 'N/A')
    st.caption(tr('負值代表下降。這是公式在本次假設下的結果，不能證明水膜必然降低發電量。','Negative means a decrease. This is a consequence of assumptions, not evidence that water always reduces electricity.','負値は減少。仮定の結果であり、発電低下の実証ではありません。'))
    st.download_button('CSV',pd.DataFrame(dict(time_h=t,wetness_dimensionless=w,dry_index=base,wet_index=out)).to_csv(index=False).encode(), 'water_scenario.csv',key='wet_csv')
    settings=dict(model='legacy_empirical_sensitivity_not_electricity',hours=hours,rain=rain,dry=dry,wet=wet,attenuation=attenuation,buildup=buildup,drainage=drainage,step=.02)
    st.download_button('JSON',json.dumps(settings,indent=2),'water_assumptions.json',key='wet_json')
    st.markdown(tr('**文獻支持機制，不支持此頁預設係數：** 水層附加質量、水波作用與撞擊傳遞都值得進一步建模。','**Sources support mechanisms, not these coefficients:** added mass, water motion and impact transfer require further modelling.','**文献は機構の参考であり、この係数の根拠ではありません。**'))
    st.markdown('[Wong et al. (2015): water-layer dynamics](https://journals.sagepub.com/doi/10.1177/1045389X14549871)  \n[Wong et al. (2017): accumulation dynamics](https://journals.sagepub.com/doi/10.1177/1045389X16649702)')


"""Fixed-water, lumped electromechanical impact model. All inputs are assumptions."""
import numpy as np
from scipy.linalg import expm

def _model2_response(m0=1e-4, f0=150., zeta=.02, C=480e-12, R=1e6, theta=1e-5,
             area=3e-4, film=1e-4, participation=.25, extra_c=0.,
             diameter=.003, speed=3., transfer=1., duration=.5, samples=2501):
    vals=np.array([m0,f0,zeta,C,R,theta,area,film,participation,extra_c,diameter,speed,transfer,duration])
    if not np.all(np.isfinite(vals)) or min(m0,f0,C,R,area,diameter,duration)<=0 or min(zeta,film,extra_c,speed)<0 or not 0<=participation<=1 or not 0<=transfer<=1 or not 2<=samples<=100001:
        raise ValueError('Invalid physical parameters')
    md=1000*np.pi*diameter**3/6
    mw=1000*area*film*participation
    m=m0+mw+md  # sticking drop retained at the lumped coordinate
    k=m0*(2*np.pi*f0)**2
    c=2*zeta*np.sqrt(m0*k)+extra_c
    velocity=transfer*md*speed/m
    initial=.5*m*velocity**2
    incident=.5*md*speed**2
    # Energy-scaled state x=(sqrt(k)q,sqrt(m)u,sqrt(C)V).
    A=np.array([[0,np.sqrt(k/m),0],[-np.sqrt(k/m),-c/m,-theta/np.sqrt(m*C)],
                [0,theta/np.sqrt(m*C),-1/(R*C)]])
    t=np.linspace(0,duration,samples);dt=t[1]
    P=expm(A*dt)
    # Exact constant-coefficient integration of xx^T and both dissipation terms.
    B=np.zeros((11,11));B[:9,:9]=np.kron(A,np.eye(3))+np.kron(np.eye(3),A)
    B[9,8]=1/(R*C);B[10,4]=c/m
    Q=expm(B*dt)
    x=np.zeros((samples,3));x[0,1]=1
    e=np.zeros((samples,2));s=np.zeros(11);s[4]=1
    for i in range(1,samples):
        x[i]=P@x[i-1];s=Q@s;e[i]=s[9:]
    energy=2*initial*e
    stored=initial*np.sum(x*x,axis=1)
    return dict(time=t,voltage=x[:,2]*np.sqrt(2*initial/C),
                displacement=x[:,0]*np.sqrt(2*initial/k),load_energy=energy[:,0],
                mechanical_loss=energy[:,1],stored=stored,
                balance_error=stored+energy.sum(axis=1)-initial,
                initial_energy=initial,incident_energy=incident,
                effective_water_mass=mw,total_mass=m,mechanical_frequency=np.sqrt(k/m)/(2*np.pi))

def render_model2(tr):
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import json
    st.subheader(tr('Model 2｜模型優化','Model 2 | Improved model','Model 2｜モデル改善'))
    st.info(tr('固定水膜、單次雨滴撞擊的機電情境模型。以下預設值均為示例，未以你的乾濕資料校正。優化指模型計算方式改善，不代表發電效能已提升。',
        'Fixed-film single-drop electromechanical scenario. Defaults are illustrative and uncalibrated. Improved modelling does not establish improved device performance.',
        '固定水膜・単発雨滴の電気機械シナリオ。初期値は未校正の例です。計算改善は発電性能向上の実証ではありません。'))
    st.markdown(tr('同一滴徑與速度，比較乾燥初始表面和固定水膜。兩者均假設雨滴撞後附著；水膜質量改變振動，電壓由機電方程計算，負載耗能由 V²/R 積分。',
        'Compare a dry initial surface and a fixed film with the same drop and speed. Both retain the impacting drop. Added mass changes vibration; coupled equations determine voltage and V²/R determines resistor dissipation.',
        '同一雨滴と速度で乾燥初期表面と固定水膜を比較。両者とも雨滴は付着。付加質量・連成電圧・抵抗消費を計算します。'))
    a,b,c=st.columns(3)
    with a:
        film=st.slider(tr('假設水膜厚度（mm）','Assumed film thickness (mm)','仮定水膜厚 (mm)'),0.,1.,.1,key='m2_film')
        diameter=st.slider(tr('滴徑（mm）','Drop diameter (mm)','雨滴径 (mm)'),.5,6.,3.,key='m2_drop')
        speed=st.number_input(tr('撞擊速度（m/s）','Impact speed (m/s)','衝突速度 (m/s)'),0.,15.,3.,key='m2_speed')
    with b:
        m0=st.number_input(tr('假設乾燥有效質量（g）','Assumed dry effective mass (g)','仮定乾燥有効質量 (g)'),.01,10.,.1,key='m2_mass')
        f0=st.number_input(tr('假設乾燥機械頻率（Hz）','Assumed dry mechanical frequency (Hz)','仮定乾燥機械周波数 (Hz)'),10.,500.,150.,key='m2_freq')
        zeta=st.number_input(tr('假設乾燥機械阻尼比','Assumed dry mechanical damping','仮定乾燥機械減衰比'),.001,.9,.02,format='%.3f',key='m2_zeta')
    with c:
        area=st.number_input(tr('水膜覆蓋面積（cm²）','Film coverage area (cm²)','水膜面積 (cm²)'),.1,100.,3.,key='m2_area')
        participation=st.slider(tr('水膜質量參與係數','Water mass participation','水膜質量参与係数'),0.,1.,.25,key='m2_part')
        extra=st.number_input(tr('假設額外水膜阻尼（N·s/m）','Assumed additional wet damping (N·s/m)','仮定追加水膜減衰 (N·s/m)'),0.,1.,0.,format='%.4f',key='m2_extra')
    with st.expander(tr('電路、耦合與撞擊假設','Circuit, coupling and impact assumptions','回路・連成・衝突の仮定')):
        C=st.number_input('Cp (pF)',50.,10000.,480.,key='m2_C')
        R=st.number_input('R (MΩ)',.01,100.,1.,key='m2_R')
        theta=st.number_input('θ (µN/V)',0.,1000.,10.,key='m2_theta')
        transfer=st.slider(tr('濕潤撞擊動量倍率（乾燥固定為 1）','Wet momentum factor (dry = 1)','湿潤運動量倍率 (乾燥 = 1)'),0.,1.,1.,key='m2_transfer')
    p=dict(m0=m0/1000,f0=f0,zeta=zeta,C=C*1e-12,R=R*1e6,theta=theta*1e-6,
           area=area*1e-4,participation=participation,diameter=diameter/1000,speed=speed)
    dry=_model2_response(**p,film=0)
    wet=_model2_response(**p,film=film/1000,extra_c=extra if film>0 else 0,transfer=transfer if film>0 else 1)
    fig=go.Figure()
    for label,r in [('Dry assumption',dry),('Wet assumption',wet)]:
        fig.add_scatter(x=r['time'],y=r['voltage'],name=label)
    fig.update_layout(xaxis_title='Time (s)',yaxis_title='Model voltage (V)')
    st.plotly_chart(fig,width='stretch')
    st.dataframe(pd.DataFrame([{'scenario':n,'mechanical_frequency_Hz':r['mechanical_frequency'],
        'peak_model_voltage_V':float(np.max(np.abs(r['voltage']))),'load_energy_in_0.5s_nJ':r['load_energy'][-1]*1e9,
        'energy_balance_error_nJ':float(np.max(np.abs(r['balance_error'])))*1e9}
        for n,r in [('Dry assumption',dry),('Wet assumption',wet)]]),hide_index=True)
    st.caption(tr('此處能量是模型電阻在 0.5 秒內的耗能，未包含整流與儲能損失；不是實測或自然降雨發電量。表中頻率為含附著雨滴的 √(k/m)/(2π)，不是耦合電壓主頻。',
        'Energy is model resistor dissipation within 0.5 s, excluding rectification/storage. Frequency is √(k/m)/(2π) including the retained drop, not the coupled voltage frequency.',
        '0.5秒間のモデル抵抗消費で、実測発電量ではありません。周波数は付着雨滴を含む√(k/m)/(2π)です。'))
    st.latex(r'm=m_0+\beta\rho A h+m_d,\quad u_0=\alpha m_d u_d/m')
    st.latex(r'm\ddot q+c\dot q+kq+\theta V=0,\qquad C_p\dot V+V/R=\theta\dot q')
    st.latex(r'E_0=\tfrac12mu_0^2=E_{stored}(t)+\int_0^t c\dot q^2dt+\int_0^t V^2/R\,dt')
    st.markdown(tr('**適用條件：** 單一集中自由度、小振幅、固定水膜與剛性、線性阻尼、純電阻。質量參與係數 β 需配合振型校正。未描述水波、飛濺、排水或降雨中的水膜演化，未將示波器資料擬合成此模型。',
        '**Scope:** one lumped coordinate, small motion, fixed film/stiffness, linear damping and resistor. β needs modal calibration. No ripples, splash, drainage, film evolution or fit to the uploaded traces.',
        '**条件：** 単一集中自由度・微小振幅・固定水膜・線形減衰・抵抗。βは校正が必要。水波・飛散・排水・水膜成長・実測フィットは含みません。'))
    st.markdown('[Wong et al. (2015): added-water mechanisms](https://journals.sagepub.com/doi/10.1177/1045389X14549871)  \n[Erturk & Inman (2008): electromechanical modelling](https://doi.org/10.1115/1.2890402)')
    st.caption(tr('文獻支持建模方向；此集中模型是簡化實作，不是上述論文的完整模型，預設值不是其校正結果。','Sources motivate the approach; this is a simplified implementation, not a reproduction or parameter calibration.','文献は方針の参考。本実装は簡略化であり、論文の再現・校正ではありません。'))
    st.download_button('Model 2 CSV',pd.DataFrame(dict(time_s=dry['time'],dry_model_V=dry['voltage'],wet_model_V=wet['voltage'],dry_load_J=dry['load_energy'],wet_load_J=wet['load_energy'])).to_csv(index=False).encode(),'model2_response.csv',key='m2_csv')
    cfg=dict(model='fixed_film_lumped_electromechanical',calibrated=False,common=p,wet_film_m=film/1000,
             wet_extra_c=extra if film>0 else 0,wet_transfer=transfer if film>0 else 1,duration_s=.5)
    st.download_button('Model 2 JSON',json.dumps(cfg,indent=2),'model2_assumptions.json',key='m2_json')



import base64
import zlib
_BUNDLED_CSV = {
    '2.CSV': (
        b'c-n;BU+*->btm@w1@=AYKB7@|{!~?8CXp8#z>64b$GPTOj6*;c4T^EHdG#SFub%FH=Ck32Aa-%)O!v(1GpA?Hcltm4@Z&#!`_pgd'
        b'Uyc9ixBvd%{ipB#`G+6>>u>(%yFdQZPygp{=lJ=bpSS)mfBflRe)s*Ke){&0|M=Z+f7$T&KYssD|McCDzxf}3|N9@m`_rF(d;R>+'
        b'ZS&uJ_pjgm;kV<je!lfz{|~?3!{7bzfBpFLZU5*07(f5ZKYaJ&Pv8IJw}1G}|Mti4fBMsJ|KY1mzxmzw|N3j4*XF<ee}4MzkH7zQ'
        b'b*`V6|K{8O@^hWv{QUp^_QxOo%Xh!|`~Uhce|~&_{@4Hf?N8tT@XMb4`Y`7G{{HUg|M~8>eS3s|S^C=_{`CFNxBKJc@&Cske*FIb'
        b'{o#*4{pCUb-E-rg?=OD6#W%l}x-I_y-Q$1y;rHMD_CNpdhkyHj|Nh_oe60WF58wX$G<^G~pMLYZAAb7FasF>l$p8NBzkc_B{QSgy'
        b'{a^fgu)qJ~KmPUK|J&Vu_j7^2|NftUx!b?{_MiXd58waspMLYV-~IUg55ND*!>#AT`CtF=`~Uv_{PG`s+^7HW{g40re}DVqcfb9I'
        b'`B(YNjedOyzx(c|Z@>S;{Zaq!+n>Ju`QN|!^{M&!pYMLT`7d|$-JkF0o9nNx>-@6tOYQf<(n4t=v@o!6RQ#y;QSqbV=P$*-`Q<o&'
        b'_vgd>%gR1nQTeU%TjjU*eX;sd^`+`7cwZ{tI`pmTt?I3PZxt^UFBPxS9+s*XpZHYyRQc4tr>dv+JybkYJXAbXJXCzD_*C(!;*+nr'
        b'sp>=3hpG=%AKLYS;z!UOL3ae*5p+k;9YJ>l-4S$0(4C{=TgAoP5pze(9Wi&lF!y|w94U9C+*v9v<c^R#LhcB;Bjk>dJ3{UVxl<}G'
        b'<4&o#h&v+gh`1x-j)*%V?ufV};*N+rBJPN|BjS#TJ5$9a+>vlc!W{{BB;1j3Tf%J#w<X+`a9hG{3Aeup_j;e$B5sSg&0cQHxGm$h'
        b'jN3A9%ec)}ZVS0B<hGF8LT(GWE#$V4+d^&&xh>?jklR9T3%M=iwvgLGZVS0B<hGF8LT(GWE#$V4+d^&&xh>?jklR9T3%M=iwvgLG'
        b'ZVS0B<d%?ILT(AUCFGWnTS9KBjaxEq$+)F9Zi%=h;+BY8B5sMewNzZfEeW?I+>&rh!Yv86B;1m4OTsM)w<O$>a7)513AZHNl5k7H'
        b'EeW?I+zP#|EfKdw+=}3Rsl1F^GH%JZCF7QiTZ8*9<d%?ILT(AUzEr)XTuZr@axLZhu)3J*Q@n}F%ej_wE$3R!wVZ1?*K)4qT+6wZ'
        b'b1mmu&b6FtIoEQo<y_0TmUAuVTF$kcYdP0)uH{_Ixt4RSPOimVe_`(RYF8)MQm&<3OSzVEE#+FuwK};LaxLUq$hD9wb#is-jIo4V'
        b'3Aqw-rB<$FT&a~S5mzFvL|lou5^*KsO2ie<R!hc}j4K&eGOlD?$+&90x0H}8Ay-1Kgj@-^5^}}w9ZJeo@)a-UO3am*D=}AMuGGzy'
        b'm@6??Vy?toiMbMUCFV-Zm6$6rS7NTjT#C6=E0<C(rCdt6lyWKMQp%;2ODUI9E~Q*bxs-A#<x-7Y3b_<=DdbYfrI1S@mulov#-)r)'
        b'8J99HWn9X*R3n!nF4f4Tgi8sR5-ufNO1KQYx1@+m5tmc%t107B#-)r)8J9d^O(B<qJ1^x@%7v5*DHl>Mq+Ce3;0bGpxe#+9=0eN`'
        b'$4ElXg`A5Wyf1YXA?ZTWg`^8f7m_X{T}Zl+bRp?N(uJf8Nf(kXBwa|lkaQvGLehn#3rQD}E+k#3mkU7`f-cm{g`5jH7jiD-T&R}|'
        b'F&FCPLdu1dn^JB{xhdtQl$%m+s+XHWZVI_6<ff3D>gA@4n=)>ymz(P4ri7akZc4Z*;iiO}5^hSkDdDDsn-XqHxGCYLgqsp>@_coQ'
        b'xGCbMh?^pAinuA_rihy&Zi=`m;--k3B5sPfDdMJxn<8$CxGCc1P(Lwc+>miY#tj)aWZaN(L&gmmH)Pz9aYM!p88>9yka0uC4H-9N'
        b'+>miY#tj)aWZaN(L&gmmH)Pz9aYM!p88>9yka0s@+z@d?#0?QQMBET@L&OacH$>bJaYMun5jRBK5OG7q4G}j)+z@d?#0_<E<Fjx_'
        b'k%yzm!%^hnDDrR=c{qwZ97P_EA`eHAhjWiSycV{KZx!DvzEymy_)_tu;!DMmcR0#B9OWI3@(xFNhoij1QQqMw?{JiNIQP86dqKrZ'
        b'#Z$#o#Z$#o#Z$#Y#Y4qI#Y4qI#ixo-6`v|TReY-WQ1PMSL&b-Ri?}1=j)*%V?ufV};*N+rBJPN|BjS#TJ0k9gxFh0@h&v+gh`1x-'
        b'j)+6p;VA5I6m~cYI~;`_j=~N{VTYrz!%^7bDC}?)b~p+<9EBZ@!VX7ahoi8=QP|-q>~IuzI0`!)g&mH<4o6{!qp-tK*x@Mba1?ep'
        b'3OgKy9nL-M@Os}qBJPN|E#kI_+ahj@xGmzgh*R3(DD80WX@{?FqVhs+3%Og`;ax<^ZG7W!?s<pTO~l+5b6d=9F}KCs7IRz7-Qo`K'
        b'J(P1>&TToj<=mEYTh47cx8>ZHb6d`BIk)B9mUCOqZ8^8)+?I1&&TToj<=mEYTh47cx7EpQF}KCs5_3z;Eit#m+!Aw3%q?|tOUf;E'
        b'a!bf9A-9CwQYW{z{?M0Nxh3P4j9W5p$+#usmW*36ZppZ%R_+#fcpnoPw`AOsaZARnVs#<6a`7fAFXfh$TT<>8d3aZk;)=@4xrG^$'
        b'qtL@q=;0{za1?qt_t3+8fjckhmZV#ft|eVdx?ASq^{`shwWw=R*P^aPU5mOFbuH>z)U~K<wR0`$TGF+oYf0CVt|eVdx|Vb;=~~jY'
        b'nz<HqE$CX%wV-P?b1mmu&0LGQ7IQ7;Zjp!AAC8*2Rx{T^u7zBynQIx>GOlG@%ea<tt!A!8T#L9CaV6qP#Fd)4l5i#AO2U<dD+yN;'
        b't|VMZxRP)s;Yz}lgewVG60Rg%Nw{0!;q_irBCbSSiMZkkYst7<;^BQ0Ay-1Kgj@-^5^^QvZh?pQ<CStH<x0(5Nx718CFM%Wm6R(f'
        b'S5mH|TuHf-awX+T%9WHWDVI_%rCdt6lyWKMQp%;2ODUI9E~Q*bxs-A#<x<L}luId>QZA)jO1V@YmqIRuTnf2VAD1#N)yJiXOA(hM'
        b'E=635xD;`zJ}xC(O1PA8DdAGWrG!ffml7@|TuQi{dSgowmm)4jT#C39aVg?L#D$0p5f>sZL|llt5OE>mg5N!aj0+hTGA?9X$hhEF'
        b'4<Y12$c2y#As0d}gj@)@5ON{pLdb=X3n3RmE`(eNxe#(8<U+`WkP9IfLN0_{2)Ph)A>=~Hg^&v&7i!}|#)XUv85c4xWL(I&DdVP$'
        b'n=)?7xGCeNjGHoU%DAaEZi=`m;--k3B5sPfsWxs(xGCYLgqsp>O1LTEri7akZc4Z*;iiO}5^hSkDdDDsn-XqHxGCYLgqsp>O1Q~y'
        b'9j1tzB5sPfDdKKnhj(=uH)Y(EadYSciYer#kefno3b`TVhL9UVZV0&{<c5$NLT(7TA>@XT8$xafxgq3+kQ+j72)QBThL9UVZV0&{'
        b'<c5$NLT(7TA>@XT8$xafxgq3+kQ+j72)QBThL9UVZV0)dHg3qcA>)RO8!~RFjT<6vh`1r*hT6Cx;f9185)NsHt+c~d+F>j0u$6Y$'
        b'N;_<&9k$XATWN=_w8K`~VJq#hm3G)lJ8Y#Lw$ct;X@{+}!&cg1-_s871r=`<ZxwG9ZxwG9$2SgJX@{+}!&cg1^WqL$afhwA!@kEI'
        b'zPc}!PwjlDdZ>D6-$TVi#ixo-6`v|TReY-WQ1PMSL&b-R3%MiYj*vS-?g+Uf<c^R#LhcB;Bjk>dJ3{UVxg+F`kh|p_o_`-l$Q>bf'
        b'gxnEwN5~x^cZA##a!1G=A&0!fR^DMN@3573*vdO><sG*24qJJLt-Qll-eD{6u$6b%$~$c39k%igTX~1Ayu()BVJq*jm3P?6J8b10'
        b'wl40l6?fS8xWntcXp6XwpUd0|JZuFXwgL}Zfrot$JiHfFT*hq~x3@ZqE#$V4+d^)uliNaW3%Og`;r)1}+?H~;xWoIt#M~BhTg+`S'
        b'x5eBRb6d=9F}KCs7IRz7Z85jS+!k|N%xy8Z#oQKix46UWsTFfu%xy8Z#oQKiTg+`Sx5eBRb6btvl5$Ip+!At2$SonagxnHxOUNxD'
        b'w}jkMBe!JSl5tDMEg83D+>&uijocD(OT;ZTa!bN33AZpHw{?Aot+d0wrybt+RmBz7mvT$WDethAci8v5!&f&^c{#V_+>&!k&Mmcb'
        b'OU|vyeHU~~&@DlyYdmaS<6$fAuoZXM_qfBWNQ=4_buH>z)U~K<QP-lbMO}-!7IiJ^TGX|uYf;ytu0>sox)yaU>RQybsB2NzqOR4<'
        b'wWMoF*J|ck&0MRQYcbbiuGP%7lxsC}E#z9rwUBEe*Fvs^Tno8YGk1$SyvUGoE#q3vT#L9CaV_Fn&0I;il5i#AO2U<dD+yN;t|VMZ'
        b'xRP)s;cjt<*V{pfxDs(C;!4Doh%274mW(SIS2FIFcX$^OawX(S$d!;QwsR%rO3Ia#D=Ak}uB2Q^xsq}v<x0wxlq)HBi#xoJqL?c&'
        b'S8C--%$1lcF;`-)#9WEF5_2i$Qp}~8OEH&XF2!7mxfF9L=2Fb1m`gQsDdkeirIbr4mr^d($fb}=A(uif)ySocOBt6kE@fP*kxLPm'
        b'A}&Q-intVUDdJMZrHISWDO<|8lyNEJQk`6ixD;_I;!?z=h)bTW=2%=&c_EiVE`(eNxe#(8<U+{l0uNh}hpouN)&(B65)Ydfc-V?O'
        b'Y(*ZnA`hF9hpouNR^(wT@~{<o*or)CMIN>y4_lFkt;oYx<Y6oFuoZdOiacyZ9=0M6TakyY$ir6TVJq^m6?xc-JZwcCwjvK(k%z6w'
        b'!&c;BEAp@vdDx0PY(*ZnA`e@UhpouNR^(wT@~{<o*or)CMIN>y4_lFkt;oYx<Y6oFuoZdOiacyZ9=0M6TakyY$ir6TVJq^m6?xc-'
        b'JZwcCwjvK(k%z6w!&c;B-y;t%!>5Rw9G06rU!6j33b`rdrjVQb@?lE3DdldlhxdJnIprR<at~X%hkegIyzfiUO+hyW-4t|F&`m)%'
        b'1>F>MQ_xL8Hw4`+_V7GbL(mODHw4`fbVJY$K{o{55OhP(4M8^q-4Jv`&<#O11l<sHx7fq$i4t@}&<#O11l>?8H{{$c_V9YRYUPHM'
        b'8&Ymaxgq6-TDc+QhL9U-<%WzKYUPHA8zOF~l^YUnNVp;4hJ+guZb-Pn6V@T(hKL*ZeswAMuoQe)3O+0aAC`g->mGb~KdhtbN7WJ1'
        b'Uy44gd-UObldbYw<u}W3m0vpaOV#lO#8UcUDgCgNeppIBtb6+5UA<MkRlQWbRK0ZMOT|mYQ^ixoQ^ixoQ^iBYL&ZbIL&ZbIr;1M%'
        b'pDI38e5&|R@uA{F#fORy6hCtA$hjluj+{Gk?v{Oc{^1-kcf=gB4@=pHrR>8}_F*aeu#|mRy4J%|_F*aeu#|mR%04V*AC|HYOWB8|'
        b'?88#_VJZ8tlzmvrJ}hM)ma-2^*@va<!&3HPDf_ULeOSsqEM*^-vJXqyho$VpQubjf`>>RKSjs*uWgnKZ4@=pHrR>8}_F*aeuym=1'
        b'rRc-br5=`&59^+Mc)ina5x3dO>0%E{;fJO0!@7qb-uETtwv^jaZcDi><+haDQf^DRE#<b9+fr^zxh>_kl-p8nOSvuOwv^jaZcDi>'
        b'<+haDQf^DRE#<b9+fr^zxh>_kl-p8nOSvuOwv^jaZcDi><+haDQf^DRCFPcsTT*VRk6S_x(TAnz!&3BNDf+M!eOQV<EJYudq7O^a'
        b'ho$JlQuJXd`mhvzSc*O@MIV-;59=O%c)cAg5w}F#5^+n!EfKdw+!Aq1#4QoGMBEZ_OT;Y^w?y0$ak|RGQtV+V_OKLtSh~o=QtDwT'
        b'^{|wBSh~o=Qs`kR^sp3qSh~i;Qs!YP^RRf0ho#WNQs`kR^sp3qSPDHXg&vkd4@;qkrO?At=wT`JuoQY&3Oy`^9+pB6OQDCQ(8E&b'
        b'VJY;m6na<+JuHPDmO>Bf9(s7au4G)xxR!A(<66eGjB6R!GOlIZE%fl(bG30T;#$PDh-(qoBCbVTi?|kXE#g|lwTNpGS0b)NT&ayK'
        b'30D%XBwR_jl5i#AO2U<dD+yN;uK1lpiMSGRCE`lNm53`5SFO(oO2!qxb0{HKLau~dsgWxoS3<6Y+%5C)-VrHRQm&+2Nx718rB1G-'
        b'TuHf-awX+T%9WHWDOXaiq+Ch4l5!>GO3I~_ODUI9E~Q*bxs-A#<x<L}luId>QZA)jO1V@YmqIRuTnf1qaw+6e$ff$YlyRv(E=635'
        b'xD;_I;!?z=h)WTd>f=(vrG!ffml7@|TuQi<a4F$Z!sQh1DdJMZrHD%rmm*G)hjouUe6^zT93crQ7g8>yTu8Z)av|kH%7v5*DHl>M'
        b'q+Ce3ka8jALdu1d3n>>;E~H#YxsY-p<wDAZlnW^rQZA%iNV$-5A>~5Kg_H{^7g8>yTu8Z)av|kH%7v5*^>HEOLdb=X3n3RmF4V_`'
        b'`nV8rp+0U(xGCYLgqsp>O1LTEri7akZc4Z*;iiO}5^hSkDdDDsn-XqHxGCYLgqsp>O1LTECeKx;h?^pAinuA_rih!RPYI@sn=)?7'
        b'xGCeNj8ooWDetiEd58CcstdU(<ff3DLT(DVDdeV*n?i01xhdqPkefno2)QBThL9UVZV0&{<c5$NLT(7TA>@XT8$xafxgq3+kQ+j7'
        b'2)QBThL9UV?v{3V{tXTxH`K-r88>9yka0uC4H-Ao#tjiSMBET@L&OacH`K-r2{$C%kZ?o74GA|S+>mfX!VL*GB;1g2L&6OSHzXV`'
        b'^3cjUw6YGZtV1j7(8@ZrF7nWdI<%q=t&2Rgk`AqlJhXxit)N5eA`h*cLo4Ud$~m+y^3aMov|<jem_sY((7MP&-%}2cj-?fIXvG{_'
        b'F^5*np%rsz#T;5OhgQs?6?16C99l7lR?MLlb7;jJS}})K%%K%?XvG{_F^5*np%rsz#T;5OhgQs?6?16C99l7lzQ-J%uiYc!j)*%V'
        b'?ufV};*N+rBJPN|BjS#TJ0k9gxFh0@h&v+gh`1x-j)*%V?ufV};t+Fa#T;5OhgQs?6?16C99l7lR?MOAF^3n`N5mZwcSPI~aYw`*'
        b'5qCt~5phSv9T9g#+!1j{#2pcLMBEW^N5mZww?*7m7q>AT*Sf?*E9cO<#6#;653Q6#E9KC-#6v6O&<Z)ULJqBvLo4La3OTew4y}+w'
        b'E9B4$IkZ9!t&l@2<j@K^v_cN8kV7lv&<Z)ULJqBvLo4La3OTew4y}+wE9B4$IkZ9!t&l@2<j@K^v_cN8kV7lv&<Z)ULJqBvLo4La'
        b'3OTew4y}+wE9B4$IkZ9!t&l@2<j@K^v_cN8kV7lv&<Z)ULJqBvLo4La3OTew4y}+w-$M?sH;^UbmWW#-Zi%=h;+BY8B5sMeCE}Kd'
        b'TOw|WxFzD2h+AyqmW*36ZppYM<Cct5%Au8V=zGfHeG@6Ct2?xE4y~L+E9cP4IrKf}@UAZBmYiF1uH{_Ixt4P+=UUFSoNGDPa<1iE'
        b'%ej_wE$3R!wVZ1?cgs1vp4(O{*K)4qT+6wZb1mmu&b6FtIoF?mVX1X>hgQ&`6?A9?9a=$$R?wjpbZ7+~T0w_a(4lp8hgQ&`6?A9?'
        b'9a=$$R?wjpbZ7+~T0w_a(4iG{XayZwL5Eh*p%rv!1sz&JhgQ&`6?A9?9a=$$R?wjpbZ7+~T0w{A)g4+%hi1~D?@5REin_THawX(S'
        b'$d!;QAy-1Kgj@-^5^|+xu7q3(xe{_E<VwhukSifqLau~d3Aqw-CFDxTm5?hTS3<6YTnV`nawX(S$d!;QAy+~!g<J}`6mlu#Qplx{'
        b'OCgs+E`?kQxfF7#J}zZks*g+caVg<a!li^u36~NsC0t6llyE8GQo^N#O9__}E+t$_xRh`y;chvH*PB#|xD;_I;xhEMnldhBT*|nV'
        b'aVg_c#-)r)8J99HWn9X*lyNEJZYhV?;SC`dLN0_{2)Ph)A>=~Hg^&v&7eX$CTnM=kav|hG$OS)?3n>>;E~H#YxsY-p<wDAZlnW^r'
        b'QZA%iNV$-5A>~5Kh5EP<av|hG$c6g2kZ~d7LdJ!R3mF$OE@WKDxR7xn<3h%Tj0+hz)yGW{H$~hOaZ|)i5jREL6me6;O%XRm+!S$B'
        b'#7z-5Mcfo|Q+?c&a8trf2{$F&lyFnRO$j$8+>~%r!c7S`CES#7Q^HLNHznMZa7sC}QVy+@L*G*lU)`6=^J|AG<ff3DLT(DVDdeV*'
        b'n?i01xhdqPkQ+j72)SF%;du~;kQ+j72)QBThL9UVZV0&{<c5$NLT(7TA>@XT8$xafxgq3+kQ+j72)QBThL9UVZV0(in8_SMZV0&{'
        b'<c5$NLT(7TA>@YIxFO?)j2mj>hKL(#<Hpnp+YoU>#0?QQMBET@L&OacH$)tQ4yB+&Dd<oNI+TJArJzG8=uiqe)II3%c&188hf>m^'
        b'cuj}8M;*Re(az&*hf>&~6m}?u9ZF$`QrMvsb|{4%N@0gm*r9Yuhf>y|lyxX&9ZFe;Qr4lAbtq*WN?C_e)}fSjC}kZ=S%*^Ap_Fwf'
        b'WgSXchf>y|lyxX&9ZFe;Qr4lAbtq*WN?C_e)}fSjC}kZ=mvks)9ZFe;Qr4lAbtq*WN?C_e)}fSjC}kZ=S%*^Ap_FwfWgSXchf>y|'
        b'lyxX&9ZFe;Qr4lAbtq*WN?C_e)}fSjC}kZ=S%*^Ap_FwfWgSYFbSOm~N>PVW)S(o0C`BDgQHN60p%ir}MIA~}hf>s`6m=*?9ZFG$'
        b'Qq-Xobtpw0>K=7?wHaH)-Lei}-K2Dr;7zdeTgYu8w}sq3tS;rYl-p8nOSvuOwv^jaZcDi><+haDQf^DRE#<b9+fr^zxh>_kl-p8n'
        b'OSvuOwv^jaZcDi><+hZ&r5#=`Q7N~j+?H}%%55pPrQDWsTgq)Ix24>ca$Cx6DYw<f-O>)v!@ktVEg84e$1M@JMBEZ_OT;bpaZAE2'
        b'3AZHNl5k7HEeW?I+>&rh!Yv86B;1m4OTsM)w<O$>a7)513AZHNl5k7HEeWTvLn-V~3Okg-4yCX|DeO@9u)}*n#YNl_aZAK45w}F#'
        b'5^+n!EfKdw+!Aq1#I^dk7I7`&TEw-8YZ2EXu0>pnxE66O;#$PDh-(qoBJLJ+c>YDTh-(qoBCbVTi?|kXx17W4;flBxaV_F-L5EV#'
        b'p_Fqd<s3>mhf>a=lyfNM97;KdQqG~2b13B;N;!v8&Y_fZDCHbVIfqitp_Fqd<s3>mhf>a=lyfNM97;Kdy5}5TZ)+vuO2n0jD-l;B'
        b'u0&jkxDs(C;!4Doh$|6SBCgcM)#7oNaV6tQ#+8gK8CPoKO2*w{4(}bQ!xfd6awX+T%9WHWDOXaiq+Ch4l5!>GO3Ia#D=Ak}uB2Q^'
        b'xsq}v<x0wxlq)G$Qm&+2O1V@YmqIRuTnf1qaw+6e$fb}=A(uifg<J}`6mlu#QplzHxRh}z<5I?@j7u4pGA?CYs*g+caVg<a!li^u'
        b'36~NsC0t6llyE8GQo^N#O9__}E+t$_xRh`y;Znk-gi8sR5-ufNO1PA8A>l&8g@g+UcS||E-nT=<g@_9g7a}f1T!^?3aUtSD#D$0p'
        b'5f>sZL|llt5OE>mLd1oL3lSG0E<{|2xDat6;zGoQhzk)HA}&N+h`11OA>u;Bg@_9g7a}f1T!^?3aUtSvDTfyTA}&N+h`11OQ^ZXX'
        b'H$~hOaZ^p)lyFnRO$j$8+>~%r!c7S`CEVm_+!S$B#7&;6P8m04+%4zuzAqs+h1?W!Q^-vrH-+3S=<t5LQf^AQDdnb=n^JB{xhdtQ'
        b'l)D8T-g_wKrkI;zZi=}n=BAjNVs47LDdwg+xhdwRn44m5in$@?hL{^-Ziu-d=7yLXVs41JA?Ajd8)9yVxgq9;8o43mhLjs><c5$N'
        b'LT;#$8!~RlxFO?)j2kj;$he_KZiu)c;)aMDB5sJdA>xLJ8zOF~ksA_jNVp;4hJ+guZb-Nx;f9185^hMiA>oFE8xjsNhg8fV6>~_%'
        b'98xidRLmh2b4bM;QrB}xr5sWzht%~PQrB}xWgJr1b4W!TQW1w##32=NNJSh{5r<U7Ar)~*MI2HQhg8HN6>&&K98wX7RKy__aY#iR'
        b'QW1w##32=NNJSh{5r<U7Ar)~*MI2HQhg8HN6>&&K98wX7RKy__aY#iRQW1w##32=NNJSh{5r<U7Ar)~*MI2HQhg8HN6>&&K98wX7'
        b'RKy__aY#iRQW1w##32=NNJSh{5r<U7Ar)~*MI2HQhg8HN6>&&K98wX7RKy__aY#iRQW1w##32=NNJSh{5r<U7Ar)~*MI2HQhg8HN'
        b'6>&&K9P%D<c$_?^G7hPXLn`Bt$~dGl4tdWwydRc~Q^+9|a!6g!A(e4RUC$vEaY#iRQW1w##36M(hg8BLm2gNU98w8~yeAyqhgZsN'
        b'DYvEEmU3IlZ7H{<+?H}%%55pPrQDWsTgq)Ix24>ca$Cx6DYvEEmU3IlZ7Fw)IJ{myQf^DRE#<b9+v?-CklR9T3%M=iwvgNE<F<_3'
        b'>f@G(TOw|WxFzD2h+86ViMS==mWW#-Zi%=h;+BY8>f@G#TM}+bxFzA1gj*7BNw_89mV{dpZb`T$;g*D35^j~=S(b=fB5sMeCE}Kd'
        b'TOw|WxTQvJiMS==bUBAq#vzq)NM#)Io^kl9i1uB`Eg`pr+!At2$hDAbA=g5#g<K1{7IH1*TFAAKYa!P{u7z9+xfXIQ<XXtJkZU2='
        b'Lav2e3%M3@E#z9rwUBEe*Fvs^T&s<18P_tdWn9a+mT@iPTE?}EYZ=!vuGPl1h-(qoBCgfOwS;R4*AlKJ+%4qr^0h@=iMSGRCE`lN'
        b'm53`5S0b)NT#2|6aV6qP#FdCE5vOZ8<UQr^)r!gsxe{_E<Zda4cXcUOQm&+2Njaq)QYnX2$|03<NTnQ7DTh?bA(e7Sr5sWzhg8ZT'
        b'm2ya>98xKVRLUWha!92dQYnX2$|03<NTnQ7DTh?bA(e7Sr5sWzhg8ZTm2ya>98xKVRLUWha!92dQYnX2$|03<NTnQ7DTh?bA(e7S'
        b'r5sWzhg8ZTm2ya>98xKVRLUWha!92dQYnX2$|03<NTnQ7DTh?bA(e7Sr5sWzhg8ZTm2ya>98xKVyr&#qZwD#jQpDxdTTAkLhg;C$'
        b's}<FsdSAaq9ll!8{>!<Pb1COS&V`%{ITvy+<Xp(PkaHpDLeAZy4zI%-`*20&C0$6mkaQvGLehn#3rQD}E+kz@x{!1s=|a+lqzg$G'
        b'k}f1&NV<@8A?ZTWg`^8f7m_X{U8t7}K^KB9)XRl>xlk_`QZCfXg?hP=aUtVE#)XWVGH%MaDdVP$n=)?7xGCeNjGHoUs+XJU<)(z2'
        b'5^hSk$urg|;--k3B5sPfDdMJxn<8$CxGCbMh?^pAinuA_Cck=^GVT_2c;6%!SBfjzdoee~+zghNb5qVue*G{7-4t|F&`m)%1>Iyf'
        b'HznQVAjuSUQ`8MnH$>eKbwkt*Q8z^05OqV;4N*5l-4Jy{)D2NLMBPv`HzeJVbVJe&NjD_jkaR=R4M{g7-H>!c(hW&BB;8OmHw4{K'
        b'GdI-C4K;H^$_*(uq})(5H-y{}azoABka0uC4H-9N+>miY&D;=iL(SZfa0onv0uQ0UL)-%o4_`xxhfv}nlz0dw9zuzSP~st!cnBpP'
        b'f|qp&MIJ(thfw4p6nO|m9zv0aP~;&Lc?d-wLXn41<RKJ!h<oJWeH2U8OVvx&OVvxeo+_Ryo+_Ryo+=(H9x5Ix9x5IxK2?0G_*C(!'
        b';#0+kiVqbZDn3;F3v$mF+!1m|$RYC($~=TJ524IMDDx1?JcKe2anC%w_FTpx^AO5Bgfb7I%tI*i5XwA+G7q85Ln!kQ$~=TJ524IM'
        b'DDx1?JcKe2q0B=l^AO5Bgfb7I%tI*i5XwA+G7q85Ln!kQ_sqlV?cj*GBjS#TJ0k9gxHHww92s|H+>vod#wqj=3Oxiv524UQDD)8b'
        b'(8E{v6}ky_{+4=p-<O=*a&F7HeYo?2ZVS3C=(eESf^G}CE$FtO+k$Qjx-ICopxc6O3%V`nwxHXBZVS3C=(eESf^G}CE$FtO+k$Qj'
        b'x~*1j%egJ*ww&8?Zp*o?R_+#ic>P6*xh>|lnA>XQwv^jCUtL0O3Av?KZppYM<Ca>vCE}KdTOw}hbahF%CE=EYTM}+bxFzA1gj*7B'
        b'Nw_89mV{dpZb`T$;g*D35^hPjCE=EYTM}+bxFzA1gj*7BNw_89mV{dpZb`T$;g*D35^hO2g&snohfwGt6nY4S9zvmqQ0O5PdI*Ic'
        b'LZOFH=pht(2!$R(p@&fDAryKDg&snohfwGt6nY4S9zvmqQ0O5PdI*IcLZOFH=pht(2!$R(p@&fDAryKDg&snohfwGt6nY4S9zvmq'
        b'Q0O5PdI*IcLZOFH=pht(2!$R(p@&fDAryKDg&snohfwGt6nY4S9^xK)cxl-pu0>pnxDs(C;!4Doh$|6SBCbSSiMSGRCE`lNm53`5'
        b'S0YZKhtQ=RLYaq9<{^}M2xT5ZS9S<R9zv0aP~;&Ld5C-D;k}^ZGOlFYE%ER^?n17FTnV`nawX(S$d!;QAy-1Kgj@-^5^^QvO30Ov'
        b'D<M~E<4VSrj4K(JGA?CY%D9wqDdSScrHo4%mohG8T*|nVaVg_c#--Z06mcoyQpBZ*OA(hME=635xD;_I;!?z=h)WTdA}-a&rG!ff'
        b'mlEz4d3Z6JA}&Q-intVUDdJMZrHD%rmm)4jobnE#OFM+(4xzY1DDDu7JA~p6p}0dR?hyC5!)p&i$c2y#As0d}gj@)@5ON{pLdb=X'
        b'3n3RmE`(eNxe#(8<U+`WkP9IfLN0_{2)Ph)A>=~Hg^&v&7eX$CToAbsav|hG$c2y#As0d}gj@)@5ON{pLVaAwxR7xn<3h%TjGHoU'
        b'%D5@xruw)k;--k3B5sPfDdMJxn<8$CxGCbMh?^pAE`6UoW!#i;Q^rjhH)Y(Eaksd``|);gMdhX3lyXzbO({2}obnE#YdeIl?GQ>k'
        b'gwhV7v_t694xz9^DC`gnJA}dxp|C?J><|h&gu)J?YdeIp4xwv1grW|is6!~~5Q;j4q7I>`Ln!JHiaLa%4xy++DC!W3I)tJQp{PSB'
        b'>JW-LgrW|is6!~~5Q;j4q7I>`Ln!JHiaLa%4xy++DC!W3I)tJQp{PSB>JW-LgrW|is6!~~5Q;j4q7I>`Ln!JHiaLa%4xy++DC!W3'
        b'I)tJQp{PSB>Jay+!wcRa;)aMDA`V%HsjS0P)?w=64pUKwsi?zL)M4JE4v%9pm35f+tix9~8Cuz@zgd5){?^`amR~BrbnI90zEpnc'
        b';N$0(rUDOBfrqKU!_+k%rt%I`d55XI!@TDm-aAsNo~oXzo;>oYeNR;nRS#7U?R%(r@KrZeeX9CY^{MJpyFOHWsQ6Ivq2jXc$hsrz'
        b'j;y;y9-c?{$hsrz5P6u2JWNF%rY`d^6?vHV$ir(t<lK>Sh&)V19;PA>Q;~<M$ir0RVJh-46?vG7JWNF%rXmkhk%y_s!&Ky9D)KND'
        b'd6<ekOhq21A`erMhpEWJRODeY@-P*7n2J11MIPoo^6)Mq<Bp6wGVV+XdxYE>+<7T?q?|GjQ<;aU%)``$9;PA>Q;~<M3q4FF9;Olx'
        b'Q;CPE3q4E)9;N~hQ-O!c3q4FF9;OlxQ;CPE#KTnLVJh)3m3Wv+JWM4XrV<ZRiHE7g!&Ks7D)BItc$i8&OeG$s5)V^}hpEKFRN`SO'
        b'@i3Kmm`XfMB_5^{4^xSUsl>xn;$iAS4^xSUsl>xn;$bTBFqL?iN<2&@9;OlxQ;CPE#KTnLVJh)3m3Wv+JWM4XrV<ZRiHE7g!&Ks7'
        b'D)BItc$i8&OeG$s5)V^}hpEKFyeA%BA}<lQMBEZ_tM#tBWZaT*x6H#=_oebeZV9<1<d%?ILT(AUCFB;Hxh3V6lv`46Nx3EEmXupk'
        b'Zb`W%<(8CNQf^7PCFNSmwUlcq*HW&fTuZr@axLXr%C(ehDc4f2rCdw7mU1oSTFSMQYbn=KuGPo2kZU2=Lax=vwTx>S*XrY1#I=ZP'
        b'5!WKFMO=%x7I7`&T76tgxR!7&;abAAglh@c60Rj&OSqPBE#X?iwS+4PR}!uyTuHc+a3$eN!j*(830M5?p+sDXxDs(C;!4Doh$|6S'
        b'{N|x#+%5F*zKM`4Ay-1Kgj@-^5^^QvO30OvD<M}xu7q3(xk`OPQBtm?TuHf-awX+T%9WHWDOXaiq+Ch4l5!>GO3Ia#tIt2RG?jXo'
        b'N<B=a9;Q+cQ>llk)Wf`|9$pVu%B7S`^>Me<!>fjnOCgs+E`?kQxfF7#J}zZks*g(%mm)4jT#C39aVg?b#HEN!5tkw^MO=!w6mh9O'
        b'?iPA@*>_7le6^zTGVT_8`0Bp2^Fl6#oG$b*6?~ZY;KOTSaOcHbin$bX$uAy4&fSs^U)`j1lh6WoJ_KE;oeM!1f-cz3h1$80bRp?N'
        b'(uJf8Nf(kXBwa|lkaQvGLehn#3rQD}E+kz@x{!1s=|a+lqzg$Gk}f1&sFw@%av|qJ&fSs^uNSqL3o#dBF4W6~lnW^r>g7Vng^&v&'
        b'7dm5|GH$Aun<8$CxGCbMh?^pAinuA_rh2(4;iiO}5^hSkDdDDsn-XqHxGCYLgqsp>O1LTEri7akZc4Z*;iiO}5^k2>)~1NlWgez-'
        b'4^z2^socX<?qMqTFqM0l%00|`?%}l{<L2b93%M!erjVOLZVI_6<ff3DLT(7TA>@XT8$xafxgq3+kQ+j72)QBThL9UVZV0&{<c5$N'
        b'LT(7TA>@XTyX799uiYW!hL9UVZV0&{<c5$NLT(7TA>@YIxFO?)+PER&hKL&?Ziu)c;)aMDB5sJdA>xMGxFO+&gc}lWNVp;4hJ+gu'
        b'4zY*p&zXSuc_H2lOADoi(89pNQSqbVN5zkd9~Iv!zEymGDgJoByZ$_x^m#?)m&&iD`&!!hrJY}@-m2c(`Bw2(@mBFV+QU-y;-N2<'
        b'uc4cyc0N@;wezX!shtlM4;2p;4;2p;pDI38e5&|V@u}iN#fORy6(1@-J{Lc7?#Q_#=Z>5^a_-2vBj=8sJ9p<Euj(V{&K<hfP4?nS'
        b'afJ_H53MXdf~5nnbO4r){?h)pj()3nt9Yw;t9Yq+@z|HDm#UYlr>duR{U+V>QQf3_EetH&q^sw`(n4wBOY!qu7+APTHy=BAlkT-p'
        b'w}rSbgcinap>7M~XyIOb-4|Ym{ma7ru-}W1`(gi5W8bBVTaA5}E^l|W@6x>%;<oU(t8rU6w}o?m&hK}1?$7!C;hxe$+!pS|Z_w4F'
        b'_&!?rQoL_Rci)+NErb?Sd>!n^x*zt(LVbK{_no-;T=@K_w(rD!^IBP4DczaMw_b8L<eo2wx*zz*Exr`rkK<eSzw~?Yn{n6U7B}Or'
        b'$Lr)~+-sq33-?Er_u}{cxEc3+xHseaQR89UW1;XD5`lvl19vr5{AS#G+||4v_Q%3~&+mt1>adUdVZZOkSXwC57%$zwEPOuf>uy}%'
        b'_G8_RtLMUJX4c)f*TTTUIUj9UcjKOq>So-!9g=l7Zrz^2bvN$0a5L_;aDUG4kLqFE{ZT!PyYJxs{DNC|<MO%i`K@I=jeFc8K3cM#'
        b'#@%nxM+^7j_dV~zqw3Ht1{Uhz;ogLc=Pf?B6YDA57kAfPxOgm}i(7Z$zIol`*1tci+>Cp#46WRhd#~Ic!F)Wnn{(gH$055x_g)$I'
        b'mHTsiqwc-(I0p9vJs;;V_d4@&4)eIT+aq|s0&d=YlaG75f%nbx&*CQDH~IM6yOH-zJs$PVyl?LA^c}r#?thnkSMQtq-+15Id#}WO'
        b'<#BI(1P8~U@AN&6b>HoKJU{(9m3`c1e13lVaW4CO1ov~<=cC?_WAIqHJ%T%${WzChk4JDnm)(#3xqooCpSOAZH9l_h_-njBw)OaH'
        b'ygze&xN>_0*P~A3mDqO+AAgN~$MEr2-~0ftc~;vxt8Jatw$5r>9=6VETc6*xKELZ1hWEn~M^Ez|!{^H>9xuVi3LxkQhWV&7AAbb*'
        b'+syTuuO^25!tik$aNG|JpSKxHf9_8VAGZnpQ4bu0dSMv%mGzm)dSO`imH5nLJurOU=8oZO1;?Ns7~bD5>w)3@vyXaVcwf;6d!?^1'
        b'il6J%-NM&R1{M|%z*?VQ?)8wcp0|n5Le)dUeB7qb=a*EyPJ8b!f-XHdjX#8XNcdbS^d<F>a6TS&6<2~QLo0c>vh>)pv_Ba-IypWA'
        b'QV$TH$2#~dMCqA};L6a7KrDh-4-j7~p+`N3R-Pc<AHi5U)}!?NjNl4h2l)W;PD?&OyuXj<1H{(~ILZfz@mM*BE1x;aS9AEdN$4g6'
        b'3!CNjHAg-;eD3BB9hyC~viKCN$Dd-{SBfja6}|}iaAk3&xDs3$dTdV+A1gct#bfXU@qR>$&q5wOr@6So*J0{)n0g)J3*NkI_^c77'
        b'h0wyl!c_68KS6xKn_u^ZUy6``!DBy&&>%vCe}hj3@4xRU2IIltYo)jnT*2R9ycn$a<>)`Jpecz*U-+tnzks+)__}KE`uLet+#QUs'
        b'FEae~cNB3~FuuM-?bqLZ#ofU9^#xA9{?;$_Rc|~1eBPvw4oWYs6z@$Bpa+idjlc7{xEc=q<8#9GzG~K|BmDTBFy7Z(`na%gSE=F|'
        b'c(^&RgZQ>L#9oNKxU=`VtAUr$)QfH|?BFaMtNHnZ>A0)+To^j+V`)KuZbOHCjE@WR?%n+Q=WahPlomn@Dt?X@mKN&c6E*M7y%q)*'
        b'_WAhK&Zlzg{mW?|?<(+c_rM`ppa0J1UAcPw)X~RR`+R;A@>m#H=%WP{@6tkOVPK(-7F4`cytL=|<ZnKDyH~Cgz1@8DcJ+05^}}1j'
        b'<X0h+Uxm!a$?mgG6mJu}4IjZgy3Jg?4IjbeF_`GF=A+A-Uk}xP=;Y_?WH;VJ+4*s0+*j%~)EoWtN<MG%=ZV+PE5()I%6P0i=J2?p'
        b'M{xKE4j;jJJ%V_R!F*m>TzNi%dJP$TUKv<fuSf9y68Fo>{VTEiBY1vc8obTGZTj^H*6Zt&&nr3x&129!2E}KgUXNhBKFs^Pa(@Kp'
        b'@$u~a5j@YV&im+m`8<Lvw@2_gN7cgkyb$;D&n|9hp|lXHf#Q?<KDqDn>sR2z+)u}NK92r#&hXcTUtFvASL64U=d=9lN`Ac(|DBbN'
        b'74w&SJCBuLt|@Wt+F#uywa{AFTDY|EOF;MMm9JmPzx)czmxbG5`Q>N!zAV&zVQb;~ybzD#abLI<zr9@IUi-0qWBc`Sd)^D*>gK)f'
        b'V|!!!#`f#u_PW=-J)Xk;7WTKWzx6oMx5E86(sf@*EwmQ47VtE$NAYuCNG-G$wiZx)YtOg#d~475qj=mGaM%x??}O+2;Q2myogKW+'
        b'F1)@kyuPnL*R8L&Z*0FlZvQ1A_shcjeBhVNAhtKQ@4svxca+BUdHbB${-y3Wj{>u`u(j}|^i#K(T1YLl7F0Y`JXJik=czqU70(L`'
        b'g`-}H?TzhtwhPa8<5kvpl{H>vy|MlJynUhWLfwV^UD)4+{oNB!z}Azz^(1dS$y-nIQSqaf_tDGyY%S>JeX01RJ-@W)m-hVnyytPp'
        b'?6r_u`1pheI16wV;4JQ(J-4IoMBRzHfx3aZfx3aZfx3yhiMolpiMpw$xbzg4p5kgPY%QpGt9Yw;t9Yw;tN6kpUpV9ohkW6XFFe&7'
        b'b@xX@<1X3rzu)-pug}}h#CGiOLEVG82m5<re=pR%Q1`-dzCPB?J6_N2pI=*fx9hdAwQyl!PA#bTRPm|eQ^iBYKO>evEP+@8v3#^*'
        b'=W$Yb+v~^q{k;`?E^IAas8M*jZ;sXDzyDJDc|Tt&KMLOvyM0EMcf%h4{f)5a_Wt~MKDT0d_v^7=3;Vn9Brm<d7M|sW!h3#xpc@Bz'
        b'<3Qj1I`2;uJ{`D&1ATCyZ+<<uqwdYG$K$`SzZVYlg}<f?$NBo`b>U31?tax{`^V#4cfIEQlag_L-aaR`C$^*R4X%2B0`zIW0{a_@'
        b'?H`YGL9M#0^|+s#T93z>IL_JFj{VK+^Y%jBLft~$+MoBUu)mGt-1+&qHI8%RIDZDK09FC4x`Xw2oD2KAu)iC1H(pm8`@6BfIBzVS'
        b'H`e34@ww1i*jiBWOT{l0zf}BEahx}{&Kq03S3i2M$49E@z53C6J#I0l7E%i;KDQQBJXAbXJXAbXJXAbYJXJhZJXJhZygoO7{Wy_)'
        b'F0>Z57A`Ebinoflinofl_I&BEFCBLDVtu#mwV*$>&${(px99f8_Koe==k1@*Ci`yNb9-Yu>K@d+u)i1f_rm^OI?_OG3AOb=?eVP-'
        b'P+LN63AH8EmQY(lZRvz^>4b9WgmOJldoHvVRQ&Tf<-!MJ3m=Rvd@#1~!PvTc_Sjw@U*8L#>@R$>zwpWa!YBLdF4=QC>Ne^&>Ne^w'
        b'J;zJW@zQg=^c*j}z_yBS72kSwZx!DvepLLZ_|cvp6+bxY7oP14&-R69`@*yR`LnT&&&Kwhvgh{3_KoeRJ5hI{E<P38_*{SEbN!9a'
        b'^|u7u5^PJb{b<V`3pur*;;G`P;;G`L;)O$Ac&ZCWyl})DN4!zDQTOxbVjGZcd_}qO73Ich=EiB}#%bop$6_0e*hV9^(THs{V*Ae5'
        b'b35uD)IF$sQ1?RJOE2n6FX{`GKjU@Oh#j3>9(}TZ;Maj)2Ywy+b?~+10bd7v9q@JViT=T7W9N?7<2VHBCh8{YCh8{kH&M4xxAY7b'
        b'3Kt6FE6Jm;BoEv=`bP5T8_A<@B#(NqqZ7&V=)s-~TMHK!_S6D)eB+RByvFv&&q2=}u;=!~_WpeT2Xzl#V;5dy7xwqU{$6;A;Y9L6'
        b'^L3&5x-cSk-SK*EzdmoD6Wc!?=<ANxb35uz)D6`A{K5VOye@pOe*v!xcwNBjy5sd68N~j6{$T&Y2m2R3*uU_>{&lzOxgE#3vA>Pu'
        b'+&IpS<J>sT3w0OjF4SE(&I|jy@w(cmyYc*P?C(b1gZ({tT^;N%&K56-bwR8PVqFmHf>;+CE?tp9mvGQEA9RrjT~$Jtp?F-k;+xm-'
        b'?a~4&2P%KQSfsAk@wmXp`v$1|`J$J)>WD5=^0=<a`;Jh#RJpX%rQ=<yT?Y%T;;rJX;;rJX;!DLBj`_kdUn;#+dgFz*QF&{xw@Pn3'
        b'-A6lpbi@z#`l$3l=}V<AmA+8=f+!b`_vfo&>GD~+-qzy+UGD;)_xW}Wu-6S{XaS|?!gf>+RTgHq>#MyNwAaw=^F`lul{j5i?r|Zy'
        b'*LKt`)GgF4)GgF4)NRyl)NRyl)NRyVs0+X*0Gk5r^Q9SeEyu^jAKxhwy|y-X8Ge1fgyrL!m#+dxmF3rjUsHZf`E|QI=zBr6C0IbP'
        b'+f`2AHz?d-Vf*L34s>IIZY<zez^{N`fnF>Eb<v7_zGSYhv8#*n>MFjE%l^Ke0N@qCD}Yx3uK-@5E-au|K(Bya0lflx1@sE&70@f7'
        b'*XIji>&n@>)V8k8t&4f<>fgFt_~QcOukEOd#_RJn&vlV>T~%F|Vb^urA6I&RZU0<1K~|#aN{E#ZD<M`wtb|yp^Tz~P39=GoCCEyU'
        b'l^`oYR)VYqS&4Qlb?%rTD?wI*tOQw!ek&nXLac;X39%AlrA`?WWF^Q-kd<n<QfG__v=V3~&`O|{TZN%kLal^a3AGYxrMj&ITM4!j'
        b'Y$ez#uvK8Iz*d2+0$T;P3Tzb`t%6zwwF+t#)GDY|P^+L;L9K#X1+@x&RspR7S_QNUXcf>ZpjAMtfK~ym0$PPOt6)~atb$pEHmlHP'
        b'6~rotRS>HnR-w%*fK>pi09FC40$7DMtKe0^tAbYruNq!8ylQyW@T%ce>!Ym(SPifmVD-dXQbVkUSPiinVs)snKBeyrEtD2e8)`Mw'
        b'YN*vvt95qRV5`AagRKTz4YnF=HP~vf)nKc^R)ehuTMf1vY&F<wu+?Cz!B&H<23rlb8f-P#YOvK{tHIU+TMKM0u(iO}0$U4h)`D6K'
        b'YAvX>&}J>5wSd+FS_^0`v{?&n)<T=L&}J=wwE)%vSPNh+fVI$OEqJxy)q+<GUM+aF;MIax3tlaFwcyo)R|{S(eH6a{)&f`yU@d^P'
        b'0M-InOP|B9gSX+OQ^TbX;up|bKx+Z5Jy3Z=tqrv{)Y?#ML#++9Hq_csYeTIKwKmk+`VMh}tqry|*xKr}Hr(29Ys0M#w>I3`aBIV@'
        b'4YxMj+Hh;btqr#}+}dz!!>tXsHr(29Ys0OLK5K)m4YoGe+F)ygtqry|`mBvUYopKF=(9G++8}F#tb;!5K&%6?4#YYT>p-jnu@1yK'
        b'5bHp!1F;UoI_R?wz&Zfy0IY*P>%gl6uMWIA@an*;1FsIeI`Hbis{^kNygKmez^enV4!k-G???w=9e{NJ)={H%Al89c2VxzKY#oqw'
        b'K-K|S2V@<PbwJhuSqEetkadBqOS29a%(`IK1+y-gb-}C)W?eArf>{^Lx?t7?vo4r*!K@2rT`=o{Sr^Q@VAl1;tmpFxvo4r*!K@2r'
        b'T`=o{Sr^Q@VAch*u7&-DSr>Y&3uIj&>jGI9daMg#T@dSnSQmP%3t(LU>jGF8z`6j|1+XrFbpfpFBUt}-DWi|u1KSJR7q+ABMBRzH'
        b'6Llx*e*OeL?t}90h0p>@2TF%ZC-yn9&!0bmkNc$j>jpU3g~Elxg*`6RE!1t)ZPablZPablU8uYC6fYEBD7;X3qwvNKZ`9qWyY<pK'
        b'D11=(pzuNAgB`w5_d?wZbuZKfTN7+eur<Nf1X~kqO|Uh=)&yG<Y<(VykNZgcYdh+~tO>I|kHZJX;RECFfpPf2IDB9nJ}?d+7>5sx'
        b'!w1IU1LN?4arnSEd|(_tFb*FWhYyUy$9)|B`3FBC)`VCSVoiuOAr_|L1Jm$vpN4<k0Ax*&H9^({S@Wnc%)%^u+-KomH(0vC=Yt(k'
        b'E1*_Dt$<nqwPK>@39uEjbOTg|TLHHMZUx*5xD{|K;8wt`fLj5#0&WG|3b++;E8teZt$<qrw*qbj+zPlAaO?9Rd|(hhFbE$QgbxhD'
        b'2L|B-gYbbt_`o21U=Thq2p<@P4-CQw2H^vP@PR@2z#x2J5I!&n9~guW48jKn;RA#4fkF7dAbemDKJJ6?ukEM{uM%D*yh?bL@G9X|'
        b'!mET=39k}f-NxWwS7!rQS)m)07M2#U&y9Db1X>BS5@=x-J}?U(n1v6_!pD6U{<R%@47GCWV1um$Td8I%;Z`1123&dcEJLn@TnV`n'
        b'a;2KB1YHHX3Un3dD$rG+t3X$Qt^!>Jx(aj^=qk`vpsPSvfvy5w1-c4!73eCoS_QcZtyZDcDzsV!whFCQL9K#X1+@yTRspR7S_QNU'
        b'Xcf>ZpjBwK3T73|DzsV!vI?zML99ZnRRF61RspO6ScO)r;8nw`hF1-*8eTQLYMmT5z-oZi0ILC31FQyE4X`@!&d?C6Ayz}IhFA@;'
        b'8e%oXYKYYkt07iHtcF+(u^M7E#A=Au5UU|pL#&2a4Y3+xHN<L&)ex&8Rzs|YSPiinVl~8Sh}96QAyz}IhFA+?Er_)s)`D0IVl8x7'
        b'3t%mPwE)%vSPNh+fVBYDLWi~B)q+<GUM+aF;MIax3tlaFwcyo)R|{S(c(vfwf>#S(EqJxy)q+<GUM+aF;MIax3tlaFwcyo)R|{S('
        b'c(vfwf>#S(EqJxy)w=MJ!va_fU~Pc40oH8<{#kgBr3F+5SsP?+khMY923Z?qZIHD=)&^M{WNnbOLDmLY8)R*ewL#VfSsP?+khMY9'
        b'23Z?qZIHD=)&^M{WNnbOLDmLY8)R*ewL#VfSsP?+khMY923Z?qZIHD=)&^M{WNoxq8)9vUbs*M(SO+cEL5p?Z)qz(BULAOK;MIXw'
        b'2VNa`b>P*}IpG0V2Vfn5bpX}@SO;L8!rAwMSVy0~ACPrG)&W@uWSv%Fm~~**fmsJ;9hh}s)`3|EW*wMyH0p4G)&W|Wf)7l=2d3Zy'
        b'Q}BT)_`no=U<y7k1s|A#4@|)arr_f~1^?P!*uJnGb)nV;wJxZ2L9GjFT~O<SS{Ky1&}Cgf>jGLA(7J%u1+*@pb)n0;VAh2$>jGI9'
        b'$htt*1+p%5Sr^2*Al3!3ZbR^ozv~NPT@dSnSQo^)Al8L0>jGF8z`D?7UGVCHR~Njx;ME1ME_ijps|#Lz9)eE{!6$~`6GQNcA^5})'
        b'd}0VbF$A9&f=>*=Cw?qG@ni9+5%|Oid}0JXH3FadvG~LceB#IA6C?165%|OieCns-6EpCM8TiBud}0PZF$15Nflti9CuZOiGw_LD'
        b'h|l{7{Odq06kaI2QFx>9#tv`P-Kcv|_n_`U-GjOZbuZMtQ1?RJ3w43k1X>elO`tV_)&yD;XicCsfz||C6KGAK^?3w7F#?|$flrLU'
        b'Cr02CBk+k4_{0c&Vgx=h0-qRxPmI7PM&J`8@OdABe?l-J)`VCSVoiuOA=ZRgpGV*mBk+k4_{0c&Vgx?#Bk-^7s0*+rz{2mrCnn%i'
        b'KL?-qIrzi`eBLMEU)xa_WKED2AS*yt47?}BSXw}3pcOzXfL2TuhFSr&0%`@+icn#&6<{mCR)DPlTLHELYz5c~uoYk{z*eBo3aAxO'
        b'E1*_Dt$<nqwE}7d)C#B-P%EHTK&^mU0kr~Z1=_3tT7fnzU{=7afLQ^v0%irwN|==}E74{p+N^|F39%AlCB#a!SqZQbU?spxfR$*o'
        b'5?&>|%EY^20;~jB39u4iCBVV}eB!6z6Z7whpMp<}zbD4u6XWlBAAf&thgb=*5@O+(;1lETiShTu55Xs<-xJgCd7plNZU<QjvJzw^'
        b'$V!lvAS*#uf~*8t39<@g704>JSp~BSW);jTm{l;VU{=Abf>{N#3T73|DwtI;tI%T=$SROkAge%DfviH0RS>HnRza+SSOu{PVim+H'
        b'h*c1)AXcHrDu7i0s{mF3tO8gCunJ%mz$$=M0ILo@o-K$~5UU_oL9BvU4Y3+xHJYpjSPifmU^T#MfYku20agR723QTST4#j~v09(M'
        b'H^{=Tz$b>^^FI9kx&hE?pw&PNKLei_e@~3RC&u3s<L`;__r&;nV*EWZ{+<|rPmI4O#@`d;?}_pE#Q1w+{5>)Lo)~{mjK3$w-xK5S'
        b'iShTu_<LgfJu&{C7=KTUzbD4u6XWlR@%O~|dt&@OG5($ye@~3RC&u3s<L`;__r&;nV*EWZ{+<|rPmI4O#@`d;?}_pE#Q1w+{5>)L'
        b'o)~{mjK3$w-xK5SiShTu_<P>R-=9z}fVBYD0$2-REr7KE)&f`yU@d^P0M^=gOIi?XL97L_7Q|W*Ye6jh2z=s4;1dJzi2?Z3kHF`B'
        b'0{*>FxTmEXpfuFlP-{c24YfAZ+E8mltqrv{)Y=+#*kEgetqry|*xF!ggRKp=HrU!=YlE!~wl>(>U~8kz+E8mltqrv{)Y?#ML#++9'
        b'Hq_c1o%9A<8-3P>S{rI@sI{TihFTkHZS+|iXl<aif!0Bvb<k%Wkaf^!9f)-x)`3_DVjYNeAl89c2VxzFbs*M3pLGD%0ayoM9e{NJ'
        b'7DnI`Bk+meflo}p=Y0bHRRCn&M&RE!KxLS9VAg?I2WB0Zb@UPZ0a^!W9iVlD)&W`vXdR$+fYt$8N3#wGY8|L`pw@v}2WlOtbwRBQ'
        b'YTahwANzGdtqW>hQ0szP7u33-)&;dLsC7ZD3u;|Z>w;Pr)ViS71+^}ybwRBQZPo>}E}(S*tqX0|1+y-+Sr^E<K-LAaE|7JBtP5ma'
        b'XtOSebwR8PVqG{nya3h(ur7dg0jvvPUFfndcy+<63tnCD>Vj9FXW$Dn@P!%p!VG-fXW;KZ7lz;qL-2(m_`(o;VF<o31Ya0}ulo@E'
        b'Ydh)&>V7^kEX}|dX5dS|178?|FATvKhTsdo17Dbdulo%A>j4!Cw|2Qvxly^X$Bn`Zd%RG0=}<2eUMRd#c%$%E;f=x@g%1iJ6h1i8'
        b'2XznXUZ{Ja?uEJ+>cXuFw<g@0aO?9Jd|?c}Fa}>3gD;H17slWVWAKGB_`(=`VGO=7245J1FO0z##^4KM@P#q>!WevE48AZ1Ul@Zg'
        b'jKLSi;0wP5Uzma~Ou-kX;0sgmg(>*L6ntR{zAyz}n1U}%!560B3sdlQpMrlvHUZWISQB7PfHeWu1X%OJd(wnh6Jkw>H6d0E6pqgy'
        b'!WVuAzVJKng(>*b@4y$v;0t5$g`a`1`xN}E0MrVo6;LanRzR(QS^>2JY6a8^s1;BvpjJSwfLZ~y0%`@+3aAxOE1*_Dt$<nqwE}7d'
        b')C#B-P%EHTpvwxN6+kP1RsgL4S^=~IT~@%XfLQ^v0%j$;tVEZU5Gx^8LaaoWl>jRNRsyU<mzD4;;Z?$`gjWf#Qs;&Vuo7S;z)FCX'
        b'04o7j0;~jB39u4iCBRC6l>jRN)@=y>b)X?uLac;X39%AlCB#ad7$(R{kd+`SLDp>w{#6)eCCo~gl`tz|R>G`=Sp~BSW);jTm{l;V'
        b'U{=Abf>{N#3T73|DwtI;t6)~atU`}fAge%Dfvf^q1+ofc704=(RUoTCR)MSnSp~8RWEFa>f>;Hy3St$+Du`7OtI%T=z$$=M0IL92'
        b'p~otCRq(3dRl%!*R|T&MUNyXGc-8Q#;Z?({)>&Z#te&MCpt4R28)S8aZh(D`r5j+U)o699G}yWg!oP2T(r~NcR>Q4^TMf4wZZ+KM'
        b'*3%5Q8gMn>YQWWis{vO7t_EBUxEgRZ;A+6tfU5yl1Fi;K4Y(R`HQ;K%)qtx3*8*G%a4o>K0M`Os3ys!-TMLcW0$U4>)`D6Kjn)EM'
        b'3ys!-SqqKULZh`H)<UDT0M-In3t%mPwE)%vSPNh+fVI$QEqJxy)q+<GUM+aF;MIaxw>kJHxC>w{fVBYD0$2-REr7LjYPcZQf>=wR'
        b'!DAA>FbQAxN%;2-K9}A=YXhwfv^LP%Kx+f74YanJtqrv{)Y?#ML#^8&{Hrk7+F)ygtqry|*xF!ggRKp=HrU!=YlE!~wl>(>U~7Y|'
        b'4YoGe+F)ygtqry|*xF!gqs`h-YeTIKwKmk+P-{c2jW%lotqrs`+N_N>YlEzfHfuwy1F;UoIuPqXtOKzQ#5xe`pv^h}>j10+unxdF'
        b'XtR#a4G+LN0P6s(1F#OjIsoectOKwPz&Zfy0IUPBFa}>3gD;H17slZ0J_i3P0J09q!W?{I4!$r4Uzmfh`yBjhJN6i69hh}s)`3|E'
        b'W*wMyVAg?I2WB0Zbzs&7vo4r*!K@2rT`=o{Sr^Q@VAch*E|_(}tP5scFzZ5(b%Cr4WL+TZ0$CTxx<J+ivM!KyfvgK;T_EcMSr>Y&'
        b'+Z6o6hYMm|5bHvZbpfmkU|s03E_ijps|#LT@alqB7reUQ)djCEcy+<63tnCD>Vj7nyt?4k=PCHUPr={cNE&1CjWPJfPr)~);QKxW'
        b'|0+CD`15&TV-CJC2j7^3Z_L5>eGdLrI8ZoHI8iuJIB}>ye*oY2G5FU)>0p2UHTcFPeB;;P`#uQ&x}(NUH+H&FxKX%quovns)Lp2%'
        b'P<Ns3M%|6N8+AA8Zqz-fdr<eF?m^vyx)<tRsC%LAg}Q+2^B8<%48Ac2-xz~$jKMd?;2UG`jWPJf7<^+4zA*;h7=!Qo82sb!aRRLg'
        b'v?kD+Kx+c6&tve7G5E$9d}9p0F$UilgKvz%H^$%_WAKeJ_{JD~V+_7A2HzNiZ;Zh=#^4)c@QpF}#u$9x$KapPPk=Q6)&yAiHTcFH'
        b'd}9v2F$dq8gKx~i_k9ljeFId6S)pbtfK~vl09v;>_*VgSTem^@_X0}Kr5j+U0apO7&y(<tN%+Pjd}9*6F$v$8gl|m3Hzwg5lkkm6'
        b'_{JoBV-mhG3E!B6Z%o2BCgB^C@Qq3M#w2`W6237B-<X7NOu{!N;Tx0ijY;^%Bz$8MzA*{kn1pXk!Z#-28<X&jN%+Pjd}9*6F$v$8'
        b'gl|m3Hzwg5lkkm6_{JoBV-mhG3E!B6Z%o2BCgB^C@Qq3M#w2`W6237B-}pWF#vpuS5WX=8-x!2%48k`C;TwbSeIJB>zA+@gN`RFB'
        b'D*;vltOQsIuo7S;z)FCX04o7j0;~jB39u4i<-vP@Lac;X39<5^>qwB5AS*#uf~?yh{IhVutb$nuvkGPv%qo~wFsooz!K{K=1+xlf'
        b'70fD_RWPeyR>7=-Sp~BSW);jTm{l;V&|?+IDv(tmt3XzPtO8jDvI=As$SROkAge%Dfvf^q1+ofc6?&|KSOu{PVikI<0$2sG3Sbq$'
        b'D)d+luNq!8yy}7X;09O?uo_@Bz-oYnLHNcXd}|QCF$mupgl`POw+7)`gYb<(_{J~7w|)`6F$mungl`POHwNJwgYb<(_{Jc7;|Jjz'
        b'bMTEh_{JQ3-{;_82MBI8+-kVhaI4`~!>xu}4YwL@HQZ{r)o`ofR>Q4^TMf4wZZ+ImaBIP>1-BOZtOd3f*jiv~fvp9$7W%9OwHErU'
        b'1+*6WtOc_c`mBXMYeB39u@?HQ1+W&tS^#SStOc+Zz*+!n0jveE7Qk8y?+pu$)&f`yU@d^P0M=~~{`G($)`D0IVl9ZZAl8Cd>)>s8'
        b'fvlxZ;V}u{n1pZqCVXQMzA*^j7=&*O!Z!xt`#uQ&+<^_XHq_csYeTIKwKmk+P-{c24YfAZ+E8mltqrv{)Y?#ML#++9Hq_csYeTIK'
        b'wKmk+P-{c24YfAftPQj_(Aq$21Fa3THoB}0vu<<n&%YO#wPDtVSsP|;bXgl@ZFE^1UDgIz8(?jKbpX}@SO;JofOXJi9e8!%)zO63'
        b'0ayoM9e{NJ)&W=tU>$&U0M-Fm2Vfn5bpY1Uc-Dbf2VxzFbs*M(SO;Pqh;<;=fmla<)&W@uWF3%oK-K|S2V@;hI~?>`2WB0Zbzs(k'
        b'SqEkvn03@<9iVl9)&W`vXk9?-0$LZ)x`5UNv@W1^0j&#7)&;XJn03Ld3uav~>w;Mq%(`IK1+y-gb-}C)W?g8qE|7JBtP5maAnO8I'
        b'7s$Fm)&;UIkaeNSx**mCu`Y;pL97d6T@dR+lXU^C3t(LU>jGF8z`6j||5w+&Wx1^+Nfe!u5D#<r`A@9AsvC*o{v~hHgp{;1$a@wv'
        b'0Bdl_8hADEYT(uH5I(pae9*xM9emKi2OWIW!3Q0Da69<CGx+ZdVR_+-7ls#x7p{0=cVl;BcVl;BcVqWp_h9#6_h9#6_hR>A_hR>A'
        b'_hL8L4R(XwU^mz;>=t$lyM^7tZezEx+t_XFHg*TQgWbXIU>9ON5bJ?h55#&P)&sGA7w|y=9~AIG0Us3bK>;5W@WGAXg91J%;DZ7_'
        b'DByzvJ}BUW0zN3<g91J%;DZ7_DByzvJ}BUW0zN3<g91J%;DZ7_DByzvJ}BUW0zU5o{_hXe2Vgw_>j78~z(N6^cLD!>0?Q!l0a?#z'
        b'8D>2&>w#G);DZ7_DByzvKDZNn-U0mQaqI%E1+*5>de!fr3x-+?YAvX>e*ddGLH{1~??L|_^zT9c9`x@){~q-3LH{1~??L|_^zT9c'
        b'9`x@){~q-3LH{1~??L|_^zT9c9`x@){~q-3LH{1~??L|_^zT9c9`x@){~q-3LH{1~??L|_^zT9c9`x@){~q-3LH{1~??L|_^zT9c'
        b'9`x@){~q-3LH{1~??L|_^zT9c9`x@){~q-3LH{1~@4=1WgZw?n--G-;$lrtfJ;>jq8^H(td(giJ{d>^A2mO1{zX$z$(7y-$dvG84'
        b'Ab$_?_aJ`{ZUZ0G??L?@)bBz49^3{#h~I<wJ&50f_&tc<gZMp&--Gx)h~I<wJ&50f_&tc<gZMp&--Gx)h~I<wJ&50f_&tc<gZMp&'
        b'--Gx)h~I<wJ&50f_&tc<gZMp&--Gx)h~I<wJ&50f_&tc<gZMp&--Gx)h~I<wJ&50f_&tc<gZMp&--Gx)h~I<wJ&50f_&tc<gWJI8'
        b'-M;_*fp-Dc1y~ngU4V4~)&*D>U|oQ90oDarDBq*<Jt*I!TfhgmfDg*|pnMO?_n>?a%J-mr56btTd=JX^pnMO?_uvljLHHho?|Fys'
        b'pWgtWb%E9eS{G<tpml-P1zHzqU7&S=)&*J@XkDOnfz}0D0a^iC0a^iC!67R!D=;fCD=;fCD=;fCD=;fCD=;fKWCe$;K&;@96@V3h'
        b'6@V3h6@V3h6@V3h6@V3h6@V3h6@V3h6@V2SvI4IHuL7?EuL7?EuL7?EuL7?EuaLdxoxOj)0K*f#(C_8aNALA->4~E!q9?EowF+t#'
        b')GDY|P^+L;L9K#X1+@xl71S!IRZy#-Rza<TS_QQVY8BKfs8vv_pjJVxf?5T&3ThS9DyUUZtDsgvt%6zwwF+t#)GDY|P^+L;L9K#X'
        b'g;Q1mtpZwwQ&z#O#wn|D%4(dl8mFv=R}HTkUNyXGc-8Q#;Z?({hF1-*8eTQLYIxP~s^L|`tA<w%uNq!8ylQyW@T%ce!>fi@4X;|C'
        b'y*I#mb?@IVz%s;Yh}96QAyz}IhFIOYVvyA!t3g(StOi*PvKnMH$ZC)^AZtL@fUE&o1F{BW4agdhH6Uw1)_|-5Sp%{LWDUp~kToD{'
        b'K-Pe)0a*jG24oG$8jv+0Ye3e3tN~dAvIb-gj#vY+21l#`SOc&IU=6?;fHeSX0M_7$HSlWS)xfKPR|BsGUJbk&c=fw`FLdvP?!C~x'
        b'7rOUC_g?Sr{qs0>AM8HZeXzT*yRf^kyRiGa$0}s+h3vhMy%)0gLiS$B-V51#A$u=m?}hBWkiFMCd;ffcE`~3LFNTBR;DUqQU^mz;'
        b'>=t$lyM^7tZezEx+t_XFHg*TQgWbXIU>9ON5bJ?h55#&P)&sE~i1k3M2Vy-C>w#Dg#Cjms1F;^6^}BnoclZAHFX{ua9*FfotOsH}'
        b'5bJ?h55#&P)&sE~i1j;sFNE)f@VyYe7sB^K_+AL#3*mbqd@qFWrSQEFz8AvxLik<?-|HQ|e;&s!#CjkW?)+Z3^LrtCuXpzTeWBmW'
        b'3uY~twP4nQSqo+@n1$QE7rOUC_g?7U3wM1lWbcLSy^y^Z?)qM+-V4=xp?WV=?}h5UP`wwb_d@kvsNM_Jd!c$SRPTlAy->Xus`o<m'
        b'UZ~y+)qA0OFI4Y^>b+3C7pnI{^<Jpn3)Op}dM{M(h3dUfy%(zYLiJv#-V4=xp?WV=?}h5UP`wwb_d@kvsNM_Jd!c$SRPTlAy->Xu'
        b's`o<mUZ~y+)qA0OuXpwS7wA{^{(S<=5NkuM4Y4-F+7N3)tPQcY-WP6=wL#V^eE+-<%-S$(!>pZXK+8~TL#++9Hq_csYeTIKwKmk+'
        b'P-{c24YfAZ+E8mltqrvf)H+b>K&=C{4%9kO>p-mowGPxeQ0qXg1GNs+I#BCCtpl|VPFV-1tOK(S%sMdZz^nta4$L|*>%gpoQ`P}l'
        b'2V@<bvJS*L5bHp!gHzVQDeJ(i1FsIeI`Hbis{^kNygKmez^enV4!k<>>cFc5uMWJr@an>=3$HG`Li%1v-wWw`A$>2T?}hZekiM60'
        b'`(9|@OSgTmcl`c&LLX;cpmlZ9x=`yvtqZj-)VfgX9$hoox?t;qtqZm;*t%frf~^a-F4($Y>w>Kdwl3JZVC#ad3$`xUx?t;qtqZm;'
        b'*t%frf~^a-F4($YD_|>ND_|>ND>!BaY6WTqY6WTqY6WTq$E^JPw^s}Cdm(-=#P5apy%4__;`c)QUWnfd@p~bDFU0SK_`MLn7vlFq'
        b'{9cIP3-Nm)elNuDh4{S?zZc^7Li}Ec-wW}3A$~8!?}hli5Wg4V_d@($h~MiSzkeRbF1!l73cM<Kh4Q^{%lE=9-wWM)p?fcM?}hHY'
        b'-rf7>7YeZoVim+H9J2~y6~rotRS>HnRza+SSOu{PVim+Hh*c1)AXY)Ff>;Hy3St$+Du`7Os~}cEtb$ktu?k`p#43nY5UU_oL9BvU'
        b'1+fZZ6~rotRS>HnRzs}D39A8C1FQyE4X_$uHNa||uo_-9ylQyW@T%ce!>fi@4X+wrHN0wgg*(0%%J)L~UMSxS<$IxgFO=_v^1V>L'
        b'*SmcGTv&9)Age)EgRBNwuk!sf46_<$HOy+5)iA4JRyRIxXrR@t?={qFsMS!bp;kk!fm#E#25Jq|8mKi;YoOLZt$|ttwFYVp)EcNY'
        b'P-~#pK&^pV1GNTf4b&Q(vIb}k&>EmMKx=^30IdO91GEOGtidU3K-Pe)0a*jG2B)lnSOc*JVhzL^h&2#vaLO8hH8^Dryc&2l@M_@I'
        b'@AkdXzBk(UM*H6H_WccUBYtnh?~VAq5x=+Y`QE7C8})m?>-WzKt)nM`SHkkf^2Q}^T=K^5#_qxH(G?#I9}FK1UkqPd@Wt-M?!|7f'
        b'8|((V!EUfy*e&c9b_=_O-NtTXx3SyUZR`$q2fKsa`Sq`#HS+gH{@%#n8~J-9e{baPjr_flzc=#tM*iN&-y8XRBY$t??~VMuk-s-?'
        b'``+*R{VyC3#Cjms1F;^6^*euW<nN9Ay^+5+^7lsm-pJn@`FkUOZ{+Wd{JoLCH}dyJ{@%#n`<=gk9>*@gdH~h~upWSgyS_K__kQQ^'
        b'pD*~`zc>2#M*rTp?R&rT_rC)RXf2?%fYt(93urB%wSd-o@V#D8YeB6AwHDM`P-{W01+^B`S~zAcptXS30$K}bEugi4)&g1!Xf2?%'
        b'fYt(93urB%wSd+FS_^0`ptXS30$K}bEugi4)&g1!Xf2?%aLQURYvGjj>fitVeS}yGVl9ZZaLU>MYXhteur|Qj0BZxR4X`#&SsPw$'
        b'c(vixhF2S2ZFsfe)rMCaULk*P<nN9Ay^+87JAeNSK&%b1HpF`M@1FsXwL#VfS?J#z{d=Q-?|1+HeIYCZtqrs`(AuS8sI{TiZhfy?'
        b'zsqoI!>tXsHr^d>xb-UFKffTrwE@=wTnBI+z;yuE0bB=g9l&(}*8yAya2>#P0M`LruLS<j_x}K{1Go<0dL{6GUmtEAxOL#x!9nYQ'
        b'tpm0W*g8099jJAn)`40F2dx9N4$wL{XdRe!VAg?I2WA}{v<}ERAnV|ub#Tx+0P6s(1F#OjIsoh7pmpHYfmat^U3hij)rD6VUR`)~'
        b';nmeU!waxp3H;{^Al8Lg7h+u;wNSwKyMX`u1^Ptm>acZz)&*Mk;OFp#S{G_vsP!t~KLcRvf~^a-F4($Y>w>Kdwyw@v7j9j+b>Y^9'
        b'TNiF!xOL&yg<BVHUAT4O)`eRaZe6%_;nsy)7j6Y^1#Sgy1#Sgy1#Sgy1?Q}Qt>By$s1>Lcs1>LcoU;P7f^$~zfmT3PKvr<h3d9P;'
        b'3d9P|Spir9SOHi8SOHi8SOHiOSOHi8SOHi8SOHi8SOHkkd&B6xVL(<uR`fZ1U{+vO^yV-?EBYM1pjJVxf?5T&3ThS9DyUUZs}wvG'
        b'*ebA9V5`7Zfvo~t1-1%o71%1URbZ>YR)MVoTLrcXY!%oluvK8Iz*d2+0$T;P3U3b!Y8BKfs8vv_pjJVxf?9=RR^gacFsooz;h0q*'
        b't3X!am{ky~aLj6e)c~sjRs*aCSPifmU^T#M9J3l;HN0wg)$pp}Rl}=>R}HUPZw(t@HNa}UHEf905UU|pL#&2a4Y3+xHN<L&g%G|G'
        b'!Z$+rMhM>s;TyMrZ*=gD4!+UBH#+!62jA%68+U+jWblnUz&9%RMg`xf;2RZuqk?Z#@Qn(-QNcGV_(lcasNfqFe4~PIRPc=ozEQz9'
        b'D)>eP->Bdl6?~(DZ&dJ&3cgXn_q&4s^Y?Uc#2SD#0BZo&0IUI61F!~Q4Zs?JH8^4oyc&2l@M_@Iz^j2*1Fr^N4ZIq7HSlWS)xfKP'
        b'R|BsGUJbk&c=fx2A5`#z3Vz-d{QC#k;1=+M3Vu+*4=VV1SMZ<V2g47B*FU__!t&Dc!t(FW;OCvef1g--;{2Z2nBKVPqvfOJqpLoc'
        b'KDz3w?Tc^t#YNxfg>coupRr&#7!EEu*e&c9b_=_O-NJ5Tx3SyUZR|F72fKsa!R}xecs;=D0bajr_(2UnsNn}S{Gf&()bN8Eeo(^?'
        b'YWP77Kd9jcHT<B4AJp)Jd%+K4_(2Rmh~Wn@{2+!O#PEX{eh|YCV)#J}KZxN6G5jEgAH?v37=94L4`TR13_pnB2QmC0h9AW6gBX4g'
        b'!w+Kk!L8s2E&SkC@bgaLKaXP<UOn*YfmaJ&t$+9-ya3h$SPNh+fVBYD0$2-REr7KE)&f`yU@d^P0M-In3t%mPwE)%vSPNh+ow63h'
        b'S`ceNtOc<a#99z*L97L_7Q|W*YeB39u@=Ny5bIUK|NTBgtOc<a#9EEB>;hQ}WG#@jK-L0T3uG;jwLsPaSsP?+9I-aU+7N3)tPQa?'
        b'#M%&RL#z$4HpJQxYeTHP@LpwutgZKk8)j{o^=je2FN9^FwQ<VYFl)oC4YM}P+AwRwtPQg^%-S$(!>kRnHq63};0GoApoAaX2!0U4'
        b'4?_6Cjo=42f}eK=|G5C5wSm?KS{rC>ptXV423iMb9iVl9)&W`vXdR$+fYt$82WTCjb%53ZS_fzy9J3D0Iyh#%GWb7#eh$bwIA$G)'
        b'bs*M(SO;Pqh;<;=fmjD(9f)-x)`3_DVjYNeAl89c2dAt9unxdF0P6s(1F#OjIsoh7kaghIfma7!9e8yHP#%DF0M-Rq7hqk0bph4|'
        b'SQlVjfOP@Z1y~ngU4V4~)&*D>U|oQ90oKJC>jJC`ur9#50PE^~;e}WiVqJ)JA=ZUh7h+wAbs^S;SQlblh;<>>g;*D2U5Ird)`eIX'
        b'VqLv2d=>DY$1x1DF37qd>w>I{Bi4mj7h(ls1!4ta1!4ta1!4ta1!4ta1!4ta1!4ta1!4ta1!4ta1!4ta1!4ta1xKs^tN^S4tmuCh'
        b'24V$b1!4ta1!4ta1!5(5`yG%Kkd?yEhQO@AtiY_`logz^0<vBO{O1=6vjVdMvjVdMvjVdUW);jTm{l;VU{=Abf>{N#3T73|DwtI;'
        b's}wpE&?=x+K&yaO0j&aB1+)ri70@c6RY0qNR^gOYFsooz!K{K=1+xlf70fD}vI=As$SRz&3St$+Du`7Os~}cEtimCy09FC40$4SB'
        b'X8!lkS9AaS_~*YbJU%}k?|%;--+27u@xtTSeX#pr_rdOi-3Pl1y9>Juy9>JuyBoV3yZirk{{q?@!yCf~!w16$7ksdLuzRt4v3s$5'
        b'v3s!_>;}8RZm=8d7Iq7}h26q#VYeUrL1_#(h8x3;;lTwDb|F?ntcF+vu?Aud#2Sb-5NjaTK&*jS1F;5T4a6FVH4tkc)<CR*SOc+U'
        b'V;5izz#4!x0BZo&0IUI61F!~Q4Zs?JH2`Y>)&Q&lSOc&IU=6?;fHeSX0M-DkY3#zQfmZ{s=KsC=3*rE*0aydD<|kPH0S(>yp8'
    ),
    '3.CSV': (
        b'c-n;BUC%Vxbspw(1O5*@4@ao=wfC;WB$D9(GGbVYb3C>t$^aB8P>d7j*Jp;LIaU2Wm%{^s?EQAHs_K5OUe&$su7C5p-~ZG1fB1F$'
        b')%8F9`d|O6fA_;b{qFbw^;duU!*Bob5C7+{<NE18Ki&Gj{Pqw3{MSGJ%OAe~?SKB^*FWFz_rL$~AOG=(-~Z}={D*(|{SW{0FTcKj'
        b'`p@I$zy0CA{P3G!Uw`$}z5nKa_~Rq|-S7Ud-~V*q|M}nMPygpX{_y)h{P>^0|IM%dx8MHwhkyC?fBfdAU;Xuu|K*Q$K5qV-|K|@s'
        b'{PrLIxEkxH<-h&@KmAnaS3mu)fB*a6{ih#(_4ohvpZ~OdfBL`w;rl=Q_`9F??2nrf&&T__pZ@EIU(aI;|FZP&fA=px{&c_JzPA70'
        b'{_gib{=dKb?H~TK(SP^8@lTJJf4s+ce=PM_{QrBzfA_n8_~F<8;dj6J^Z)wS|LUh>{m;Mo{->wm`+xbvum1XXfB4IB{%=pn|Ni}d'
        b'`QiWg>52RHU;Oc4|M=Vg{5OC9=ZF2_rviWf<3IiUuz&acKmGG>e*En}{_5ZV@cSQs_YZ&B+<tG)|N5JM`0ww}&;Q`defl>){{El-'
        b'@9%&A!>|8i{8jyYr$27OU;prj?|=NwbF2UQ`#*gD)BpeOk5A1{|M}tPyZ`(^Km6%&zPtbGzR%AKKiB?TSXyW;q!unL92Gw*epLLZ'
        b'_)+n#;#<YHif<L)D!x>FsrXXyrQ%D)N5x0QN5x0QN5xykTg6+&Tg6+&OT|mYOT|mYOT|;gQ^ixoQ^ixoL&ZbIL&ZbI<1fX(`<b*K'
        b'{v_JZE0?NYs(z{ZrRtZepI?moW(DPskUPI1_swJB&>tyxq}-8m=dk+W-f!-`pgV%@2)ZNaj-We&?g+Xg=#HQ}g6;^qBj}ExJHrDn'
        b'>5im3lI}>lBk7K$JCg25x+Ce1q&t%CNV+5Gj-)%1?nt^L>5im3lJ10x3%Vodj-We&?p!Fo)yr)$x5eC6FSn)KmU3IlZ7H|a%Wd^?'
        b'TfN*Caa+V~5w}I$7I9m|Z4tLc+!k?L#BC9`Mcf{}xojG@W!#o=TgGh}x0}`La7E>%+?H}%%I#!zF}KCs7IRz7Z85jS+!k|N%xy8Z'
        b'FYdaW+j4Hpxy^=d3A!cdmY`dLZV9?2=$4>cf^G@ACFmBLx%G>5->j&-s9U0LiMl1~mZ)2zZi%`j>XxWmqHd|1TRLrBf^G@ACFqu*'
        b'TWaQ(nz<$BmY7>&Zi%_2W^PHjCFPcsTT*UGxh3V6lv`46Nx7wFZV9<1<d%?IYUW19jf@)^H!^Ny+{m~&Qf7qQ2)Pk*qi$~0&CM^w'
        b'eQuVF8yPn;ZWi}l$c>O2AvYu7N6L+q8#Z(!=0?nom>V%SVs6;a4g0w%9(zeQl5Qm3NV<`9lYHHYx)F6F>PFO!s2fo?qHaXph`JGV'
        b'BkD%fwWw=R*P^aPU5mOFb**-;C0$FpmUJ!YTGF+exfXOS=vvUVnz@#9E$3R!wVJsWb1mjt&0I^lmU1oSTFSMQYc+GNX0FxDwTNpG'
        b'*CMV(T&I9-b#pD^TEw-8YZ2EXu0>pnxE66O;#$P@r8m~*Ico{I5^{AnA1iei?0g-)vzC}EF<1N&qWC353Az$=CFn}fm7ps@SAwnt'
        b'T`lgrq$^2RlCC6ONxG7BCF!d5_g<o|L|uux5_Ki&O4OC8D^XXXuGGzyq$^2RlCC6ONxG7BCFx4im82_4SCX#O%$1tClyfQPQq5e7'
        b'xfFA$W-g^%O1V@smulux#-)r)8J99HWn9X*<XLM9xl}ioGA?CY%D9wqDdSScrHo4&m)%@Zc_EiVE_u$HQZ6fb7nK)tDdtklrI<@G'
        b'mtro(T#C6Ab1CLh%%zx1F_&U4#axKF5OX2sLd=Dj3o#dBF2r1jxe#+9=0ePcm<ur%VlKp7sFe#T7g8>yTu8Z)av|kH%7v5*DHl3t'
        b'4K;Eh<3h%Tj0-h#A>u-fTu8W(a3SGB!i9tj2^SJBBwR?ikZ>X4Lc)cF3keqzE+pI~;VubxNw`bGT@vn+aF>L;B-|z8E(v!@xJ$xa'
        b'67Jf1L%&4aCE_j-cP(`@myEk)+$G~K8F$IJOU7L??vinrjJw##T|(~S*AJJJyQJJzEHCCRF?Wf%OUzwj?h<pCn7hQ>CFU+McZs=6'
        b'%w1yc5_6ZByTsh}HFHM+h@$|+Q2^p70C5z6I0`@<1t5+B5Jv%sqX5KF0OBYBaTI_!3P2nMAdUhMM*)bV0K`!M;wS)d6o5DiKpX`i'
        b'jsg%z0f?gj#8Cj^C;)L3fH(?390eeb0uV<5h@$|+Q2^p70C5z6I0`@<1t5+B5Jv%sqX5Kt1|VLzpJxK%n-!JE#N1H?;wS=f6oEL5'
        b'K%8d;;(eDR<&Km)Qtn8(Bjt{iJ5ugQIprVDGym{$7cqBs^RZN3&K)^-<lK>SN6sBNcjVlWbC3MP=dKI7Bj}ExJA&>Ax+Cb0pgV%@'
        b'2)ZNaj-We&?g+Xg=#HQ}g6;^qBj}ExL;m3?|8SIlILbd9<sXjn59gVG_<Sj<m)lZqOSvuOwv^jaZcDi><+haDN4vh&%xx*RrQB9C'
        b'w}spma$Cr4A-9FxUfgvlx24>ca$DWp7IIt2Z6UYu<-<_|;wS-eo(YJL1u3_s+?H}%%55pPrQA;Lx|rKyZi~4s=C+vIVs4AME#|hE'
        b'+hT5uxvgeyOSvWGmXupkZb`W%<(8CNQf^7PrCx3cxh3S5kXu4-3ArWYmXKRQZV9=iPHxG#CF7QiTQY9RxFzG3j9Y5tmWW#-Zi%=h'
        b';+BY8B5sMeCE}KdTWaK%gj*7BNw_89mV{dpZb`T$;g*D35^hPjCE=EY8wocOZf>2ojEEZ%HzICC+=#dlaU<eJ#Epm>5jP@kMBIot'
        b'MIX*H`tZ$)$}gR&kCc04AHKPZ_Fl}*u)Lfb_HrZWM$nC*8$maMZi>fV(v75>)aO&fZf<1V$hwhrBkM-ijjS74H?nSI-N?F;btCIW'
        b'*0ro_S=Z|3TGX|uYf;ytu0>sox)yaU>RQybsB2NzqOL_<tD9?eb1mpv-CWDLRyWsTuEkudn`<f8>gHO=wUBEe*Fvs^T&tUF8P_td'
        b')y=iKxt4G(;abAAgnMKkK74Hv*CMV(T#L9CagXrB=Vr;cmT@KH9@&SFA|>QX$d!;QAy<dhrCdq5;`a|F=4!LNoGUq3a<1fD$+?nq'
        b'CFe@cm7FU%S8}f8T*<kTb0z0W&Xt@iIahM7<Xp+Ql5-{JO3sy>D>+wkuH;<Fxl$)rVy?toiMbMUCFV+<TuHe{^5Okgn^G>-$)%7>'
        b'A(uifg<PtWOBt6kF4f7Uh)Z>HDdAGWrG!ffml7@|TuQi<a4F$Z!li^u36~NsC0t6llyE8GQo?2FgfT^2intVUDdJMZCC^q<#-)r)'
        b'8J99HWn9X*lyNEJQpTl>OSW<;<Wj9%2)Ph)!H?!b%7v5*DHl>Mq+Ce3ka8jALdu1d3n>>;E~H#YxsY-p<wDAZlnW^rQZA%iNV$-5'
        b'A>~5Kg_H{^7wY3e$c2y#As0d}gj}eP3mF$OE@WKDxR7xn<3h%Tj0+hTGA`7|g@_9gcZs-5#9iv+E(v!@xJ$xa67G_4mxQ|{+$G^I'
        b'33qW+?h<jAh`U7G#cv-j8Fz7v<l?sv*Bq{B_r=^L<}NXJiMdP6U1IJMbBaD3MIVl$59b+u_*~$j7j&1Py9C`O=q^Eb3A#(rU4rfs'
        b'beEvJ1l=X*E<tw*x=YX@`mhy!*or=EMIW}J4_ncPt?0v6^kFOduoZpSiau;bAGV?oThWKD=)+d@VJrHu6@A!>K5RuFwxSPP(TA<*'
        b'!&dZREBde%eb|aVY(*coq7Pfqhpp(tR`g*j`mhy!*or=EMIW}J4_ncPt?0v6^kFOduoZpSiau;z`(f+a4_m>9txG>_<sSAk_wafH'
        b'*$O`FXYk>@a75e@apzX`Bjb*YJNVV5t>nXgCLg}JOL9f^H|xu}Bj=8sJ96&Gxr2UgEBmmOeb~xAY-Jy|vJYF=e%OjWY(*coq7Pfq'
        b'hpp(tR`g*j`mhy!*or=EMIW}J4_ncPt?0v6^kFOduoZpSiau;bAGV?oThWKD=)+d@VJrHu6@A!>K5Sk3VJrEtm3-JrK5Qi)wvrE9'
        b'$%n1v!&dTPEBUZ>>4&Z0!&dNNEBLS#eAo&;Yy}^-f)884hpphl)}<e|at~X%hppVhR_<Xd_pp_F*vdU@UHD-u_OPF^hmW^|E#h?H'
        b'hpphlR`6jf_^=gx*a|*u1|PP94_m>9t>D9c1|L3)NVzTLwv^jaZcDi><+haDQcf3s*or>vXY}ElyR`0-TEM<9Ik)88l5>mA+!Az4'
        b'&@Dl?1l<yJOVBMrw*=i1bW6}JLAM0m5_C(@EkU;g-4b+5&@Dl?)XFV6x75llF}K9r5_3z;Eit#$$}K6kq}-BnORd}za!bf9A-9Cw'
        b'5^_t(Eg`pr+)^tyGH%q$jffi&HzICC+=#dlaidmlB-}{2k#Hm7M#7DR8wocOPSJ<0=)=}!AGVSYTUUM93O;NFA2x#zTfv8|t3GVy'
        b'9=381n^%3<3O;NFAGWUfu$6n*$~|md^I<dguoZjQial(_9=2i+Td{|&*uz%rVJr5q6?@o<J#57uwqg%kv4^eL!&dBJEB3Gzd)SIS'
        b'Y{ee7Vh>xfhppJdR_tLb_OKOu*or-D#U8d|4_mQ^t=Pj>>|rbRuoZjQial(_9=2i+Td{|&*uz%rVJr5q6?@o<J#57uwqg%kv4^eL'
        b'!&dBJEB3Gzd)SISY{ee7Vh>xfhppJd)+HadQV;u?diW5zMO=%x7I7`&TEw-8t6L}PCF4rQm5eJHS2C_-T*<hSaV6tQ#+8gK8CNo{'
        b'WZWb2@Oj*YTnV`nawX(S$d!;QAy-1Kgxn+Z@VV<!uB2Q^xsq}v<x0wxlq)G$Qm&+2Nx718CFM%Wm6R(fS5mIj$CZ#PAy-1Kgj@-^'
        b'5^||NE@fQGxRh}z<5I?@j7u4pGA?CY%D9wqDdSScrHo7UaVg?b#O2c4YRb5jaVg_c#-)r)8JBA0QpBZ*OA(hME=635xD;_I;!?z='
        b'h)WTdA}&Q-intVUDdJMZrHD%rmzUnxQpTl>OBt6kE@WKDxR7xn<3h%Tj0+hTGA?9X$heSkA>%^Eg^UXs7cwqn+#~YvewBpUxDat6'
        b';zGoQhzk)HA}&N+h`11OA>u;Bg@_9g7a}f1T!^?3aUtSD#D$0p5f>sZL|llt;JIqZxKJAxA}&N+h`11Omx#MW+$G{J5qF8WOT=9w'
        b'?h<jAh`U7GCE_j-r_94v=3y)Iu$6h(y4b^3<YDVt4_nuI*t*ukR^DMN@35bFhtFdo<Srq13AsziU7R4fq}(OtE-80OIRzfJ0uTEc'
        b'c=)`FoV(=QCFd?VcgeX+&RufuQY&|fxl7DlV(t=imzYE1VJY#jlz3Q5JS-(1mJ$z3iHD`c!&2g5De<tBcvwn2EF~V65)Vs>ho!{B'
        b'QsQAL@vxM5SV}xBB_5U%4@-%MrNqNh;$bQAu#|XMN<1tj9+nahONoc2#KThJVJY#jlz3Q5JgjHp;YHk1<Y6iDuoQV%iaabu9+n~x'
        b'OOc1gi#;r59u_b5uoQY&y4b@~=3((-59=9v_+|y=kC;1R?ufY~<_>;tY3XtgOP70Cyxham<sOzI4~v(3Sjs#sWgeC?4~v<HrOd-p'
        b'=3y!Gu#|aN$~-Jy?_nwQuoQY&3Oy`^9+pB6OQDCQ(8E&bVJY;m6na<+JuHPDmO>9pp@*f=!&2yBDfF-udRPiQEQKDHLJv!!ho#WN'
        b'Qs`kR^sp3qSPDHXg&vkd4@;qkrO?At=wT`JuoQY&3Oy`^9+pB6OQDCQ(8E&bVJY;m6na<+JuHPDmO>9pp@*f=!&2yBDfF-udRPiQ'
        b'EQKD{GxYGG<rZ;U#BC9`TW@V!#%&q5W!#o=TgGh}w`JUxagWTy=N<~VE#!9SeRWH@E#<b9+fr^zxh>_kl-p8nNx3EEmXupkZb`W%'
        b'<(8CNQf^7PCFPcsTT*UGxh3V6lv`46Nx3EEmXupkZb`W%<(8CNQf^7Pr9N&6xh3S5kX!2GmW*36ZppYM<CctDGH%JZCF7R*xFzD2'
        b'h+FF8mV{dpZb`T$;g*D35^hPjCE-TGjf5KsHxh0n+(@{Qa3kU7=zVQO+=#dlaU<eJ#Epm>5jP@kMBIqD5pg5pM#K%zRcCnMh1{@}'
        b'8!0zZZkpxA+=#hH=Hc_P<lM-)k#i&GM$V0#8#y;}Zsgpsl^a1ff^G!e2)Yq;Bj`rZji75m*MhDET?@JvbS>yw(6yjzwQ?=zTF$kc'
        b'YdP0)uH{_Ixt4P+=N_Sl_e-?pT+6wZbFEgc)ylP$YqfGM<XXtJTDg{SE#q3PT#L9CaV_Fn#I=ZP5!WKFMO=%xRx8&Mt|i<f^YEc%'
        b'i?|kXE#g|lwTLSbS0b)NT#2|x=HX-4OU9LqD;ZZZuK3kM?ZXw7mvSZLO3EqruoQb(iajjF9+qMcOR<Ng*uzrnVJY^o6nj{@)WcHh'
        b'VJY>nlzLc7JuIajmQoK(sfVT1!&2&DDfO_FdRR(5ETtZnQV&b1ho#iRQtDwT^{|wBSV}!Cr5=`24@;?srPRYx>R~DMu#|dON<A#4'
        b'9+pxMOR0yY)WcHhVJY>nlzLc7JuIajmQoK(sfYDUJ$w+CA}&Q-intVUDdJMZrHD%rmm)5CzM3*FWn9X*lyNEJlBcUF<Wk6`kV}q|'
        b'=xPs3$%pk!K74Z*y|JgDOF@@{E;&t-k}f4(O1hMEkL<%|bx{|hE<|03x)60C>O$0os0&dSqAo;Th`JDUA?iZZg{TXivxcM#Nf(kX'
        b'Bwa|lkaQvGLehn#3pH~g=t9tipbJ44f-cm|g`5jH7jiDt%!Qh{kaD4BF4WA0j0+hTGA`82g@_9g7a}f1T&S4~33o}jOTt|e?vikq'
        b'gu5i%CE+d!cS*QQ!d(*X+InBTMBF9f9@&S_$C7auhvlx-Tv2@~cS*TR%3V_Kl5&@nyQJJD<t{0ANx6%YBv&1-sJxuJ<lH6aE;)C}'
        b'xl7Jna_*9Imz=xg+$HBOId{pqOU_+#?vittoJ06w6n+?mA4cJaQTSmLei(%xM&XB1_+b=&7=<53;fGQ9VHAECg&#)Yhf(-p6n+?m'
        b'A4cJaQTSmLei(%xM&XB1_+b=&7=<53;fGQ9VHAECg&#)Yhf(-p6n+?mA4cJa(d8aS*@sc~VU&FsWgkY_hf(%nlzkXo?qQzMhu0g('
        b'=z0&M=))-bFp55mF846H+`}mNFbY15f)DcyK78!@k#R@H9T|6I+>vod#vK`VWZaQ)N5&m|FgMTK!{=R8UC13FcZA##a!1G=A$Nq_'
        b'5pqY!9U*sw+!1m|$Q>bfgxnEwN5~x^cZA##a!1HLQV$=mD<OA;+!1m|$RYJGN<EBH52Mt>DD^N(J&aNhqtwGF^)O03j8YGy)WazC'
        b'FiJg)QV*ll!zlGIN<EBH52Mt>JW~%Jl5O>ITf}Xgtd4RIquj$N_b|#mjB*d7+`}mMFuK^oDE2UlJ&a-xqia2kQV*ll!zlGIN<GXo'
        b'_3-&j3ArugwvgLGZmW~qLT(GWE#$V4+d^&&xh>?jklR9T3%PZxp;%IGNx3EEmXupkZXv=tiam^C52M(_DE2UlJ&a-xqu9eJ_ArV)'
        b'jA9R?*uyCHFp52lVh^L(!zlJJiam^C52M(_DE2UlJ&a-xqu9eJ_ArV)jA9R?*uyCHFp52lVh^L(!zlJJiam^C52M(_DE2UlJ&a-x'
        b'qu9eJ_ArV)jA9R?*uyCHFp52lVh^L(!zlJJiam^C52M(_=t2*p)WazCFiJg)QV*ll!zlGIN<EBH52Mt>JW~%JZwDjdM#PPXQ|w_F'
        b'dl+8oVRWg7dBz^Txr=sR%#D~EF{jwWDE2UlJ<K!q@cCGRZUo&U_wdbQsl22cNvGh$DEKf6K8%77qu|3R_%I4SjDioN;KL~RFbY15'
        b'f)AtM!zlPL3O<a252N72DEKf6K8%77qu|3R_%I4SjDioN;KL~RFbY15uJ$krK8%77qu|3R_%I4SjDioN;KL~RFbY15f)AtM!zlPL'
        b'3O<a252N72DEKf6K8%77qu|3R_%I4S%rp4#Az7=NYZ2EXu0>pnxE66O;#$P@r8k$7aV6tQ#+8gK8CNo{WL(L(l5q+?j4t;u$~}y7'
        b'52M_}DEBbW+{5RBiVL|CawX(S$d!;QAy-1Kgj@-^5^^QvO30OvD<M}xu7q3(xe{_E<VwhukSifqLax-um5eJHS2C_-T*<hSaV6tQ'
        b'#+BN*6mcoyQpBZ*OA(hME=635xD;`zHZCPxO1PA8DdAGWrG!ffml7^{#+oAT5qtQ&i;PPdmohG8T*|nVaVg_c#-)r)8J99HWn9X*'
        b'lyOQuj4t;u&(Om+cTssMmr^dJTuM1*9_E>O_*{{5Dd$qorJPIla>1`3Lehn#3rQD}E+kz@x{!1s=|a+lqzg$Gk}f1&NV<@8A?ZTW'
        b'g?hOVbfI1@<Xp(PkaHpDLe7Pp3pp2ZF63OuxsY=q=R(efoD21GA?8B8T%@20F&APk#9WBEP%{^5=0eDYkP9IfLN3(IU25hYv4@YN'
        b'c*(d+#$7V*l5rQmVtoW3zFARup0Zy1a0SQyl5>}wyX4#@=ahUHUH4%Wd>92EM!|<s@L?2u7zH0j!G}@sVHA8A1s_JihtX9ZMpu0p'
        b'#U4hnhf(Zd6nhxO9!9Z;QS4z9dl<zYMzM!c>|qpp7{wk&v4>IYVHA59#U4hnhf(aI6?<sK9$K-7R_vh_duYWTTCs;#?4cEVXvH2{'
        b'v4>Xdp%r^*#U5I*hgR&N6?<sK9$K-7R_vh_duYWTTCs;#?4cEVXvH2{v4>Xdp%r^*#U5I*hgR&N6?<sK9{L%3c)dwAFZ$35KD2@l'
        b'{R}>Q^H{7O+WS!XQ2Ai_(6PU`_m|3FDu3zVU)uR2=#HQ}g6<K1_@)SU{z$qb>5im3lI}>lBk7K$JCg25x+Ce1q&t%CNV+5Gj-)%1'
        b'?nt^L>5im3lI}>lBk7K$JCg25x+Ce1q&t%CNIGO6TG@wI_Mw%1Xk{N-SA1w?A6nUmR`#KleQ0GLTG@wI_Mw%1Xk{N-*@u2+A3j{#'
        b'GH%Pbt!8eExGmzgh}$A=i?}V~_U5k3xGm!z>4$F~OXY>!7IIt2Z6WsvKYVVMlzXHfzPXFai@7c4wwT*uZi~4s=C+vI#phklZ8^7#'
        b'&%2=8f^G}CE$FtO+k$Qjx-ICopxc6O3%V`nwxE0DA3hJSq}!5ishL}XZV9?2=$4>cf^G@ACFqu*TY_$>ms@gf$+@LoZi%@i=9ZXS'
        b'Vs44KrCx4Hxusri3ArWYmXKRQZV9<1<d%?ILT;&-TQY9RxTRihiMS==mWW#-Zi%=h;+BY8B5sMeCE}KdTO#fefB2AWshL|>@-8Sp'
        b'GHztt$heVlBjZNKjf@)^H!^Ny+_0IOy|~g`QGGEtVs6CTh`AASBj!fTjhGuTH)3wY+=#gmb0g+P%#D~EF*jmv#N3Fv5pyHvM$C<v'
        b'8!<OxZp7S(xe;?C=0?nom>YF+BjrYo+z7cAaxLUq$hDAbA=g5#g<K1{7IH1*T8&(*k!v+_E#X?iwS;R4*AlKJTuZo?aLv=z7I7`&'
        b'dUUqjGOnBDg<K1{7IMuK)|PTD<yy)$zk+Bn*J7@-`B=#n?Y*FDLDzz=`3*!%x|Vb;=~~h?`?(f%CF)AlJrWQfMM~C{tSebpvaV!Z'
        b'$-0ttCF@Gom8>gSSF)~TUCFwVbtUUc)|IR)b#o=^O4OCQxsr6HZmtAf3Az$=rEad|T*<kTbER&s#9WEFQa4xX=1RzwkSifqLax-!'
        b'm5eJHS2C_-T&bHY5tkw^MO><zO9__}E+t$_xRh`y;Znk-giD^Wrie=smm*GAerRPNS{aDel^<FWh}M-K`k8?E+$<rNLN3|QrIbr4'
        b'mr^b}SwuT8=5+0cRtln@DTr_GqVj?+1zifd6m%)*QqW~^-z8m2x|DP&=~B|Aq)SPck}f1&NV<@8A?ZTWg`^8f7m_X{T}Zl+bfI1@'
        b'1YHQa5OkqlF63OuxsY=q=R(efdbtpDA?8B8Tu8Z)av|kHy<7;n5OSejE@WJ&mkSXWA}&N+h`11OA>u;Bg?hP=a3SGB!i9tj2^SJB'
        b'c)l7U?h<jAh`V^cdU0az5^|T2yM&w)5dBO*e6ynR{0`!hbC;aE<eV=4(9a0OH!CVH=`KlkNxDnYUF_&CQFn>DOVnMW?h<vEsJlen'
        b'CF(9wcd?(lWZfm}E?IZUx=YqwvhI>~m#n*F-6iWTS$E00OV(Yo?vizvth?0BAp}thL6kxer4U3Z1W^h>ltK`t5JV{iQ3^qnLJ*}8'
        b'L@5MO3PF@Y5Ty`ADFjgpL6kxer4U3Z1W^h>ltK`t5JV{iQ3^qnLJ*}8L@5MO3PF@Y5Ty`AJwp&L;z}uqdZr-0xr@rD$|uXG%BOZ7'
        b'A&624qMjj$Z|<V<!MzXd{iX63%U>#gY40!X{gHG>(j7^6B;Ap8N75ZhcO>1BbVt%1Np~dOk#tAW9Z7d2-H~)h(j7^6B;Ap8N75Zh'
        b'cO>1BbVt%1Np~dOk#tAW9Z81}L@5MO3PF@Y5Ty`ADFjgpL6kxer4U3Z1W^h>ltK`t5JV{iQ3^qnLJ*}8L@5MO3PF@E08z?7lrj*d'
        b'3`8jdQOZD+G7zN<L@5JN%0QGZ08xrSlp+wt2t+9YQHns6A`rz3K$J2Nr3*lmA`qnuK$H>?r36Gh6A+)<E9AD2+d^&&xh>?jklR9T'
        b'3%M=iwvgLGZVS1ceAcDhmU3IlZ7H{<+?H}%%55pPrQDWs`{JLQm|J3QiMb`_mY7>&Zi%@i=9ZXSVs44KCFYixTVighky}!3Nx3EE'
        b'mKwPw<d%?ILT(AUCFGWnTWaK%j9W5p$+#usmW*36ZppYM<CctDGH%JZCF7PFxh3M3h+Fv0;8Fyl6oDv3AW9L4QUs!O1&C4tqI3m_'
        b'QUIb9fT(8x;^UZ%jGH@o7nK)sb68%=jg%WHH&Sk-+(@~RawFwN%8is8DK}DXq})ikk#ZyDbP0%30-}_FC?z0D35ZexqMiwe&&`r^'
        b'Bj-lWjhq`fH*#*|+{n2{1ma@{<=m*18!<OxZp7S(xe;?C=0?nom>V(IVy@N6wUldhaxLUq$hA7TmT@iPTE?}EYZ=!vu4P=yxR!A('
        b'<66eGjB6R!GOlG@%eYo2*CMV(oB|M~07NMOQ3^nmF8@%<Ka}zh^~^tfZm*1M8P_tdWn8P5YZ=!vu4UXK{qVVmLav2e3%M3@E#z9r'
        b'wU8?zS3<6YT(Om_Tkorn^usqRsxRkC&Xt@iIahM7<Xp+Ql5-{JO3sy>D>+wkuH;<Fxsr1w=St3%oGZ0*CFV-Zm6$6rS7NTjT&a^Q'
        b'DOXaiq+Ch4l5!>GO3Ia#D|K=u<VwhukSifqLau~d3Aq$<DdbYfr8>D(CztBvQo^N#O9__}F1LJ55tkw^MO=!w6mcoyQpBZ*OA(hM'
        b'E=635xD;_I;vU(D&*RQsE`?kQxfF6K<Wk5zvJan|CFN4erIbr4mr^dJTuQl=aw+9f%B7S`DVI_%rCdt6lyWKMLdu1d3n>>;E~H#Y'
        b'xsY-p<wDAZlneE7A>=~Hg^&v&7eX$CTnM=kav|hG$c2y#As0d}gj}eP3-xg!;zGoQhzk)HA}-X&g@lXJc}s}65OE>mLc|3}<wC}V'
        b'j0+hTGA?9X$heSkap}D!`2A{dQtlFS7r%W_{-JaQh*JEap7Dpz$Kuxym#Di$-6iTSQFm=tmvtA1NiJb`3A;<!UBd1Xc9*cbgxw|V'
        b'E@5{GyGz(z!tN4wm$18p-6iZUVRs3;OW0k)?ovB<$+}C{U9#?yb(h+?OVnMW?h<vEsJlenCF(9wcd4Db)XpLMkjg%!vJa{3Ln`}_'
        b'%08sB52@@!D*KSiKBTe_sq8~4`;f{$q_PjG>_aO1kjg%!vJa{3Ln`}_%08sB52@@!D*KSiKBTe_sq8~4`;f{$<TLy5ddtaY_~DzC'
        b'#TC^r)i2gh?R{$RF))`(Kjbt0@Ual89;zPN_fYZRzF(?-@z7r?f2q8nJA&>Ax^wHuA4zv4-H~)h(j7^6B;Ap8N75ZhcO>1BbVt%1'
        b'Np~dOk#tAW9Z7d2-H~)h(j7^6B;Ap8N75ZhcO)IM52@@!D*KSiKBTe_sq8~4`;f{$q_PjG>_aO1kjg%!vJa{3Ln`}_%08sB52@@!'
        b'D*KSiKBTe_sq8~4`;f{$q_PjG>_aO1kjg%!vJa{3Ln`}_%08sB52@@!KC=&>3o0(*wuIXfZcDf=;kJa+#UE19hg9?-b@7K(@*$Oc'
        b'NF^Uq$%j<(A(ec{XY%25*Ja$6aht8&7IIt2Z6UXX+!k_M$Za9Fh1?c$TgYu8w}qT8{*b!(Ln``^iaw;G52@%wD*BL$KBS@#spvy0'
        b'`jCn~q@oY0=tC;{kcvK}q7SL)Ln``^iaw;G52@%wD*BL$KBS@#spvy0`jCn~q@oY0=tC;{kcvK}q7SL)Ln``^iaw;G52@%wD*BL$'
        b'KBS@#spvy0`jEQ#Ln`@@&*a00GE2rS8MkEIl5tCo+!Aq1#Epm>5jP@kMBIqD5pg5pM#PPX8xc1mZbaOOxDjz9;zq=ch#L_%B5p+7'
        b'h`3=RH!^Ny+{n0*aU<hK#*K^{88<R+WZcNOk#QsAM#hbd8yPn;Ze-lZxRG(AHf}`Rh`14PBjQHHjffi&*CMV(T#L9CaV_Fn#I=ZP'
        b'5!WKFMO=%x7I7`&TEw-8YZ2EXu0>pnxE66O;#yr?OSqPBE#X?iwS;R4*AlKJTuZo?a4q3l!nK5J3D>E2mKJd>;#$PDh-(qoY~xzS'
        b'wT$aa-}khTYa!P{u7z9)xe{_E<VwhukSifqLavSud`Y>IawX+T%9WHWDOXaiq+Ch4l5!>GO3Ia#D=Ak}uB2Q^xsq}v<x0wxlq)G$'
        b'Qm&+2Nx718CFM%WmHN06awX(S$d!;QAy?|-O2(CpEA??D;!4Doh$|77A}&Q-intVUDdJMZrHD)QaVg<a!li^u36~NsC0t6llyE8G'
        b'Qo^N#O9__}E+yO}^YHP`lE~<%QV*#sJ|sgA`3yaLv!cD1a>)^r6mu!&Qp}~8d!!ydtFxO+L6?Fq1zifd6m%)*QqZNKOF@@{E(Ki*'
        b'x)gLF=t9tipbJ44f-VGI2)Yn-A?QNTg`f*T7lJMXT?o1mbRp<M(1oB2K^KB91YM|=3$=0~=0ePcTDedw7eX$CT&R@`85c4xWL(I&'
        b'P%9T|<wC-RgbN855-ucMNVt%2A>l&8g@g+U7ZNTc+$G^I33o|2UGO25ct~CFA$7rr<OLs6frnJ!Ar*K?UG5>3cSv3CAr*H>#T`;{'
        b'hg94l6?aI*9a3?JRNNsIcSyw@QgMe=+#wZrNW~peafejgAr*H>#T`;{hg94l6?aI*9a3?JRNNsIcSyw@QgMe=+#wZrNW~peafejg'
        b'Ar*H>#T`;{hg94l6n6;49YS%3P~0IDcL>ED;u&{%LKccUgyIgNxI-xJ5Q;m5;trv>Ln!VLiaUhj4xzY1DDDu7JA~p6p}0dR?huMQ'
        b'gyIgNxI-xJ5Q;m5;trupJ%ldx5DYtn!VaOZL+DBmp{zqF>kzuoLp-AnpWB<N9;zOyj_)0U7kUV$9pagG_+~{LaB2519eX)<<lK>S'
        b'N6sBNcjVkT+I6HILTQIk+98y72&ElDX@^kSA(VCqr5!?Phfvxfly(TE9YSe`P}(7sb_k^%LTQIk+98y72&ElDX@^kSA(VCqr5!?P'
        b'hfvxfly(TE9YSe`P}(7sb_k^%LTQIk+98y72&ElDX@^kSA(VCqr5!?Phfvxfly(TE9YSe`P}(7sb_k^%LTQIk+98y72&ElDX@^kS'
        b'A(VCqr5!?Phfvxfo@s{<+_#9^B5sSgE#kI_Q`{jGcL>EDLUD&s+#wWq2*n*jafeXcAryBAUFac{b_k^%LKk`ng&jg+hfvrd6m|%O'
        b'9YSG;P}m_9b_j(XLSctc*dY{l2!<U(VTVxIAry89g&jg+hfvrd6m|%O9YSG;P}m_9b_j(XLSctc*dY{l2!$O&VTVxIAry89g&jg+'
        b'hfvrd6m|%O9YSG;P}m_9b_j(XLSctc*dY{l2!$O&VTVxIAry89g&jg+hfvrd6m|%O9YSG;P}m_9b_j(XLYH|6WgS9Uhfvlbo>_+v'
        b'$(H)KCE}KddxRZ6ij0h#yLp$?T~uDkjgT85H$rZN+z7eZI`SjsW+(5W@?vhp+$<h>IX7}{<lM-)k#i&GM$V0#8#y;}Zsgp^xsh`t'
        b'=SI$roEteea&F|@$hnboBj-lWjhq`fH*#*|+{n36D>q_p#N3FvRwvg|uGPu4kZU2=Lav2e3%M3@E#z9rwK}<$ajj0SMO=%x7ICdk'
        b't|eScxR!7&;abAAglh@c60Rj&OSqPBE#bQKrrsj1MO=%x7I7`&TEw-8Yku?4GOlG@%ea<tE#q3owTx>S*D|gz?z)gGAy-1Kgq*JP'
        b'5K26R5)YxoL+C0Gp}<2Z@DK_-gaQwtz(XkT5YNEF=P?m;CFV-Zm6$6rS7NTjT#307b0y|V%$1lcF;`-)#9XP7D=Ak}uB2Q^xsq}v'
        b'<x0wx8o3g3CFDxTm5?hTS3<6YTnV`na-~MDWL(O)lyNEJQjJ`SxD;_I;!?z=h)WTdA}&Q-intVUsYWg(TuQi<a4F$Z!li^u36~Ns'
        b'C0t6llyJ&Bgf8+BiaUhj4#BuXDDDu7JH#{Y@XceXypT&FmqISt%B7S`DVI_%rCdt6lyWKMQp%;2ODUI9E~H#YxsY-p<wDAZlnW^r'
        b'QZA%iNV$-5A>~5Kg_H{^7g8>yTu8Z)av|kH%7v5*DHl>Mq+Ce3P#+gUE`(eNxlkV$GA?9XsE-Q~7a}f1T!^?(9~TlXBwR?ikZ>X4'
        b'Lc)cF3keqzE+kw?xJ$xa67G_4mxQ|{+$G^I33o}jOTt|e?%H}wzxcJoCF3p`cgeU*#$7V*;<pZ$kh_H3CFCw4cL}*m$X!D25^|T2'
        b'yM){&<Srq13AsziT|({>a+i?1gxn?ME+Kabxl71hLhcfBmyo-J+$H2LA$JM6OUPY94ta<BPjP_!X(2xsmKItIsf7y*N5zkd9~D0;'
        b'epGy`_*U_);#<YHiZ2ykD!x>FsrXXyQSnjnQSnjn`K9<v<oi!w1ATRuR`pi(R`u4dmx`B)mx`B)mx`x~r;4YFr;4YF$I%lKsvfEy'
        b'svfF-Y1c0mzf}BE@$2j2N5&l)cVyg=aYx1-8F!w>y`GRG<c^R#LhcB;^F;2W`d0O=>RZ*9sxKY*rQ#3eKDKwM`l$M-`l$M7*GI)$'
        b'#aqQ&#aqQ&#Y@FY#Y@FY#Y@Fg#Z$#o#Z$#o#iiVND)+h4T;Yo@F0EYLf{R;l=>S~ZfD0S2rQMcxTiR`Dx24^dc3U0Y7Is_MZDF^C'
        b'-Bw4ppVqw(;={U+1r>i-_rAx&x_&RT7E%ir79P}{_rfp5<F&aD>OK}y3)f>I9}5rae*WhFvG6Ruo(ms`{g;K*!u42q)~M%Uf7W<V'
        b'SFeY9P*<=0cu-gG8rNgtS>r)n{q(YYEIgm{XN`P5=g%7Xe9m7rp3nKS#)G<f?O-h3qjite!i9y0a_`N(77oePcs{PgtK>o4^F_U$'
        b'!oBQUPvSmTzGh}Ujr*u_eYsuhX<WS?>U{mIucvYIeyCrHulK?)#rJDrKmXF7H6F&D$AMo@<L=kyK8*WVxbRTxgPC6z)?+_j_jvC4'
        b'YvK8l>*vCSg*jSy7JqKl!?^cWJ&fD$1?~C6xchb3AI9CUt?K7te?Ehqy2pj*qx76VjC;RIN_$><!99$7AH;`o9}7C{`7Hk2+*I+r'
        b'u(=Q8_NzuLEi`IGVyi-XejU7aAI81^oUi^;-t{!DU(fk<eM!bVje9yePvV|$M)MTz`SvwW;p$!E!UBYwr*QeIalU>>pQmu~+T8PX'
        b'J26k;;`LDb>qctS#f{Eb=PBHG@4GyR`|e%kVcd7;z4ADLc<;;eto38%dagVV<-@tpmGxYC9?bbXh52RWbq4X=g7+tM&pHq6zI#9a'
        b'5AD8tzhoZVefNI(J-qww{Zf8__g%m4Q?F;?xu*~EzU$YcJ<R)VUR(V@?{npPt~|Hk!QOZ4wFM9NzFV&?c)<7FdL5nQ79@{Byq<;U'
        b'7R1uAj^=&xS?9TJap{OYPi5!zXxGxwc@X&Be(mY=Tz0?fTpxc~e_rWVr*|HE+OJOUe&45`E3f+`AMN`3i1w4h*I#45KEHc<&`%6s'
        b'y0p$|Tj#W`bK2H9ZR_*9mWHi!+SccHt<UfJMdABtS$O*UMd9<s)GrF>b?|3>{b$oJ3hVu7sV8}^ufgjVh55QqudiY47lrwLwB&tW'
        b'7(chV&e!Dji^BJwJ}G=39Z}dH6h2q-aAoQF$rr}YXZuOvdOzBwV;!$Q!smTref@Ub{8X>`sb1^ri`Lf{t*<XyAL+HezG!`YQ7;77'
        b'dj-e8UI;$lK<kCz^X<5ZV7(Bm*A~3L#CY8Ye+BhK@YTlDQ^6Ou^<?nXNY(3$jE|M(O0v$SZCgkCvmimeH2he}um9fbrQ!Qn&(ZTU'
        b'i!05Q<jVE6EA`PBzE&1jxCQUV@OgBa$DodmTk(}xd=@J8oEBe+$yZ{cA*h#z{a%4ZrLQ>ZiQ&h>(!$s0N2Tv^OAwYIEPlCBua5A&'
        b'^H=JCUDwy(=L^H<;m;R_&!;S37(V}~^M&EZ%Jub2BVQQiw^y0}dZoFNT)DJuZw%MxRiM8-2Es5uF#P5UyI(v%$+P5qVfeA~^*dr}'
        b'fF++4eq7h^XyNOR2=eLR$3kjB#aqRDX+eA5+VjTe6Zvsw{90LhwyO0E7oYn#fcbg7ldo46SNQ77<O;Vy?O?vz!S`QA@GmfEKyV8#'
        b'ZUGv?d@}fcc3$6q<mYb`f4dM*1D}87@g(r$9$y!aCxHF&8>C;pm_y(22IpCV=fk1%;n4YTJnj4ZaXBqCc5#3GO=mpadmqN71HZoh'
        b'rZb-CeJrFFF4UNV&7FlOceLk|I98))y$@>i!d7+TAa-I0`@$iqqXixI(qS(ZFFog}!=Bpn)E`@>7A`D=4twZ1kH%JoUhSbhzm68f'
        b'%%$i2kH*y(#s6sBw+nw7+s*lDp<X-qr>WLIFZ6q^^p}PAd}C_i`g-9!i+?+7_UDCfN3p)xs`D(~-wg14v9O=Tzn%X0ashk3wdea;'
        b'{QYhCqxi=c=!+}OmE_7}13tcHIb2y>X|5z!v<1yAXl_As3z}Qd+=6;-!TZE$apk!M@j5k2u3T8j*A~P#P5IX=&6VWJh5ZR0gWxfU'
        b'*A{#$>{nY5JO+RK3(fgivwp8EuJoT)>f4q4@2pI&jGrIvyxPp4uM=_a-rwA%J{JCRkI}-`!u_$(zZ8$3SN?Kb9t)*~(Zbfk{aX0T'
        b'R^><W$3kggwD2x|{j%^Z{&=y}t9U;bN(;~8`CQmqxW8WLXYq&3{jB}EeQ^86?e~|r59$u;j&^yp%cEUhI^Ijid%cRk7WUWA>C%(E'
        b'U&UVwrG@$W1-11`+bVuk{HXYO6@M*kE!<xpt6PWt)?vT@RJ^}=2Jb%=@6Q*>{ioXB-2QVE?&pQiv%8<efw+DCa{Ht1&k0ie`}^zL'
        b'Bffr|DBM1H9Mlce&0p$%_bQO3h0(&+!i|M`6z|tUX<@XmwQyshRlHTaRlK$5tvw$d_R(P<JmG^UeDH)XyucS;;0rJCwQ>9X_3axk'
        b'@QoMvM%|6N8+8x%_h5eyj`PuTeDoaOdX8^B$G85pZWX`3E}l=vp0Cw>LiTZw(Zbe(ibrTc#Y4rv_AKF9K0SNiUbua5JL)Ft7V6fQ'
        b'FYkOp_Wt*%Td3Qp+t}Ym-PV)bdXh&^^5{t(J;|fuqvA`&mx?bHUn;&-e5?4zLEkv&8&CGelYQ`HAJjdldr<eF?j2vhh;P)rQTImO'
        b'`^&n8W~|VR)swLI?O#8|g>I|_SP8H~KUPn_KJIX1Ayqq7J5l><!wQBK46CPM@7qzgP`6OGQMXaIQMXaIzibWaDcJjV)E(5FjmN?M'
        b'F6{3@-G#af`@2wgqwdxVYoqYii+bxtee|L}s(nTaDt@rrH+K8R;l5S*#_|4|uLfWJYQg&X3CptuJFkVUg&PYIs1fz`M#Y!?c0U2@'
        b'&rgH?eEX~LgRgi#{zI?#zaQL=x_N)SorQy4U%$Zmbt?H@*jl);(5T%w){SG`*yqNvegO8q9mjfbtRH~AZ@<6Zo)uqj|0>)*Umu6B'
        b'&mkLtH2`azL-y0J*Y<B5=Z*b6IL;5e-v1uQ`JnDW-5baG#{S+o&Nq(p*Ko}fuGi<JSK;=-?O*QiJk@&L{y^(}d*OD}jg8x}zh9p;'
        b'4o(^e4cDOI8Z=x3vjz><py8UQSg-x9`|H~q&u^n{qi$n=8~ck>#?dL`=#+7E$~Zb@9Gx<bP8mn1jH6S=(JABTlyTI1&C{-r1r^^a'
        b'e!lJnM$Ol}ny>dgwia$I+$w&n_^sl%_8gyrt=Feu?}e{hu=Q%e-V0j`Hx|%>t=CE9>#Lv`EvR_hxJRmZYR^-9o+_T&^U`5QGq%)>'
        b'Eu2d(oJ%gWVhgk_(6*kmy|$-O_v>@X^_1;>JL(ShcTjg>e;4+5=|C?OUi#bGdXl$ZVOzC9+7f9?=acJ2+N=1{%loMKQSlqc{MM`c'
        b'#*=;Hn19`uZJbbUd>*#(dDzC6llw_ozP^b@-LJo#-1snT<HP)ozHFl}+vv+S`m(KWCbzzs+!Aa{ur0y1PA9icC->`{$@jw7FYf*N'
        b'X7YUx6>k+E6(8;SsQ9S(;Q3y7z89YFh39+Wb+%A<qwYrCjk+6kH|ieLJ*az7_n_|4lYHw*zV#&EdXjHF$!N-snzEy>CXc?FJb>)r'
        b'bN!>w^$$K3JCN)^vIEHuBs-AoK(d1~%!4z`gEP#7GtBc;>~(w!bqjSHbsKdXbsKdXbq94vFRj6W9uyuFUV3FMRbHr!lgk6WjvBF}'
        b'bISw24*WXs>%gxAzYhF5@ay0c{euSVNUwti?4SWVXuxhXU^g1DTX5a_M)HPSH(Ib;Ur64XsJQiw<gITcZ+#<q>l?{iGh#RVy48r?'
        b'YQ%0eVmCe{yFu6u!fp_DznZXD;nq&Kj(BUY2ek*Uv-y(G8{bUcPsLuhfBnJ!jSu#37<S`>{TqbcXuxhXU^f7}0oV<|ZUA-zup5Bg'
        b'PrzRLdvKg@9OoOy`Ns2mV}EZ9|9yQC23@s5mwC{2A#@=LU3o&6s(4-7;yqRS%f&F>xBqhWjE~z<_m}H)eEfUVP3&)Be+zXB`&&59'
        b'g}RNpjk=Axjk=AxgSvydgSvydgSrcK7wUd}(H&i;N7wbyg@APBAYE$cb#bDP+fnzR?m^vyx(9U+>fRR)+}D>h)`g{BSEBm7!`G$1'
        b'zSxznj-|_K>H1r*3wFIXD1cS~tpHk&tB!pZQ02U|fYOQ5iG7A!0k;Bf1>6d_6>uxyR=}-*TLHHMZUx*5xD{|K;8wt`fLj5#zP=!$'
        b'uArz(F21hO`0>JmTLHHMZUx*5xD{|K;8wt`fLmW*j#Jm~)CE6vg-~5m^mPr=kIW0yN~o1kE755s&`O|{K<n#^!M?5@`>|t*!XPU_'
        b'R)Va@MRPv`Nhn;YJG2&-7I3gz&obOfxRr1#;a0+}$Ay?b!&xW{xE>dW{&=9JT?SnVx*nIf{(K<pGg_|4Ww$?bXQ&Lja%-P~*W+sA'
        b'pLftJ416W{O7NB7E5TQSuLNI(o~xi&L9c>d1-%M-74$0TRnV&fb%9p_uL52LybAqR!LCBTRp_@0{Z;|4LcdjTtKe3_t%6$xw+j7M'
        b'fvo~tg?_8hZ`HzIa6zp?!&N}5fL5X3DwtI;tI%&1$SU+(g?_6JlrD%>5UU_oL9EhgVu7p%Sq-upWHrcYkkx@V#E#SgDg&(sTCKCh'
        b'hFT4^8frDvYN*xfxEgFV*lMuVV5`AagRKTz4YnF=HP~vf)nKc^R)eiZuhmeip;kk!hFT4^8frDvYN*vvtD#mytwx*GK&ydP1FZ&H'
        b'jW(-c*1)WRSp%~MX3h2W$Kr!FYtUv5#2Sb-5Npt84Zs?JH2`Y>)&Q(Qn>Fxi;MKsZfmZ{s23`%k8hADEYT(tttASSouLfQXyc&2l'
        b'@M_@Iz^j2*1Fr^N4ZIq7H99dIfHi8f24W4w8i+LzYarG@tfkMj)|FalJkrt~P#S10ptXS30$K}bEugi4)&g1!Xf2?%fYt(93urB%'
        b'wSd+FS_^0`ptXS30$K}bEugi4)&g1!Xf2?%fYt(93urB%wSd+FS_^0`ptXS30$K}A)`D3JP1XWg3uG;jwLsPaSqo%sG+7%>)&^J`'
        b'P1c528(wXAwc*u<R~ue!c(vixhF2S2ZFsfe)rMCaUTu8}zX8?;SQ}t%fVEqNA=ZXi8)EIj$HIG58D?#mwe@+{23i|v?S;?6H`Llt'
        b'YeTIKwKmk+P-{c24YfAZ+E8mltqrv{)Y?#ML#+d~4%9kO>p-mowGPxeQ0qXg1GNs+I#BCCtpl|V)H+b>K&=C{4%9m6vJTKXK<fal'
        b'1GEmhtOK(S%sMdZpvyYwvJS*L5bL1JIsoectb;D=z^enV4!k<>>cFc5uMWIA@an*;a{%-KSO;L8yY)b*46$yAbwjKhV%-qyhFCYm'
        b'x*^sLv2KWUL#!KO-4N@BSU1GFA=X_u1;0Vo4YF=ET6f{Kbpx#%Xx%{T23j}Jx`EaWv~Hkv1Faiq-9YOGS~t+Tfz}PQZlHAots7|F'
        b'K<frtH_*C))(y07pmn3kx?$E0vu-q5H^{obB<s(YEBf;G#O;mS7j8%0K;2)+`uNdS)D6@P)J@b)>~ErOqHdyYp>CmWp>CmWp>CsY'
        b'qi%osUYMANPt3z7=HV0b@QHc&#5{ar9zHP-pO}Zw=REw!?Wns@_v^XR#5{ar9zHP-pU-*tj{*mK{PjohiHZ2cM0`Fc;y-`oSo?gd'
        b'^o`OtN`tKcTLHELYz5c~uoYnI>w);hKzu$2;y-RjU9c5kE5KHOtpHmAwgPMg*b1-}U@O2@fUU2m;S<yFiD~%6G<;$jJ~0iSn1)YG'
        b'!zZTU6VvdCY52r6d}10tpVROkx1%o53ZV7%FnnSdJ~0fR7=}*_!zYH}6T|R{Vfe%_d}0_rF$|y2Vfc^R6Sp^RU$`B0;Z?$`gjWf#'
        b'5?&>|N_ds<D&bYatAtkxuM%D*yvo8`VFIiKSP8HaU?spxfRz9%0agO61Xu~Ma^amnAyz`Hgjfl&5@IF9N{E#ZD<M`wtb|zk^_)Xu'
        b'06sAQpBR8o48SJ_;1dJzi2?Y;0DNKqJ~05F7=TX<z$XUa69e#x0r<oKd}07TF#w+!fKLp-CkEgX1MrCf_{0EwVgNod0G}9uPYl2('
        b'2H+C|@QDHV!~lF^06sAQpBR8o48SJ_;PW{E|8YC&0;~d91+WTW6~HQhRRF61RspO6ScMj=;8p3QumDy8tO8gCuo_@Bz-oZi0IM&&'
        b'_hbG&pY!h@SEgy*0i|IU=HC<Z?}_>Me9pgrJ`gIGDub;q?K9kJxYcm0;a2Mttp;2TxEgRZ;A+6tfU5yl1Fi;K4Y(R`HQ;K%)qtx3'
        b'R|BpFTn)Gya5dm+z}0}O(P%Z?YPi*KtKrtbt$|wuw+3zv8m&R2HBf8NXbsRBpfzZ;24)S+8Z=r1vIdRTK&(NdH2`Y>)&Q&lSc67u'
        b';MKsZfmZ{s23`%k8hADEYT(tttASSouLfQXyc&2l@M_@IV+8)=5^eebejwIBtbtggR%`V6`@pP$Sp%~c%vvz(F#`W_B{`tAfYt(9'
        b'3urB%wSd+FS_^0`ptXS30$K}bEugi4)&g1!Xf2?%fYt(93urB%wSd+FS_^0`ptXS30$K}bEugi4)&g1!Xf2?%fYt(93urB%wa{cO'
        b'G+7H|Es(W9)<ToD&}1!uwE)%vSQ}t%fVI(NZFsfe)rMCaUTt`_;njv$8(wXAwc*u<R~ue!c(n`fNE={nfVBbE23Q+lZGg1_)&^J`'
        b'U~Pc40Tu?}^Em+ju>i8RK7!vcYs0J!vo_4yFl)oC#{m4th61e(v^LP%Kx+f74YW4U+CXaqtqrs`&^kct0IdVG4$wM4>j13-v<}cZ'
        b'K<fal1GEm%Iza0Htpl_U&^l<c4$L|*>%go7vkuHUFzdjq1G5gyIxy?Ntb-=&pvgKA>!8Ux0P6s(1F#OjIsoectb-=&Xf*2ptOKwP'
        b'z&Zfy0IUPB4#2_$d}0DVF#(^LfKN=oCnn$%6Yz-%_{0Q!Vgf#&6Y%dHzCqRvvTl%dgRI90{6}GZ0)GRo8))4?>jqkwfltlACuZOi'
        b'Gw_KS_{0o+K4;)R3d5}%ZryO}hFdq>y5ZIhw{Ey~!>t=`-EixMTQ}Ug;ns~l>jqml*t)^i4YqEub%U)NY~ARyZm9M35PV?>zAyw|'
        b'7=kYh!54<$3q$aQ--Is=!54<$3q$aQA^5@&d|?Q_Fa%#1f-elg7lz;qL-2(m_`(o;J%`}mmo5K#3cm28@P#4x!Vr962)-}`Ul@Wf'
        b'48a$M;OjXA|FOegpBom&;0t5$g)#WTufi9m;OjXB|FJ<Eg*OTx6h0_?u)_y+59;2id!z1+x?lg=Utt8ko+I$@I|S4Us1;BPzY1R%'
        b'f-elg7lz;qL-2(m_`(o;VF<o31Ya0}FATvKhTscB@P#4x!Vr962)-}`Ul@Wf48a$M;0r_Wg(3LD5PV?>zAyw|7=kYh!54<$3q$aQ'
        b'A^5@&d|?Q_Fa%#1f-elg7lz;qL-2(m_`(o;VF<o31Ya0}FATvKhTscB@P#4x!Vr962)-}`Ul@Wf48a$M;0r_Wg(3LD5PV?>zAyw|'
        b'7=kYh!54<$3q$aQA^5@&d|?Q_Fa%%tE%?F=d|?K@FauwhfiKL!7iQoKGw_8O_<GL3zu)i@U?spxby*3q5@IF9N{E#ZE7fHs$V!lv'
        b'AS*#uf~*8t39=GoCCEyUl^`oYR)MSnS#{w+7tAV{RWPeyR>7=-Sp~BSW);jTm{l;VU{=Abf>{N#3T73|DwtI;t6)~atb$nuvkGPv'
        b'%qsL)1+ofc6?&|KSOu{PVim+Hh*jvZ3Sbq$Du7i0s{mF3tO8gCunJ%mz$$=M0ILC31FS}m)$poyQrG~i0TyQ9>p27exqw{;Sq-up'
        b'WMKxro-^<t3qY%ZRs*drRfbv(wH`C@9}fh!8f-P#YOvK{tHD--tp-~Swi;|T*lMuVOHVP}YPi*KtKn9|t%h3-w;FCW+-kVhaI4`~'
        b'!>xu}4Yvkv4cr>IHE?UtXARgIur=tj27T55tpQpCv?jsjpwSvMS_88NW(~|5G+G0)24oEyt$|nru?Aud#2Sb-5NjaTK&*jS1F;5T'
        b'4a6EWS_7~KV2w@<2VxDx!Weu#$KXHifIS9T1F|p$U-%{X!Vr962)-}`Ul@Wf48a$E2fp+>@P#4x!Vr962)-}`Ul@Wf48a$M;OjXA'
        b'|FHvTwHDA?Kx+Z51+*5>T0m<7tp&6e&{{xi0j&kJ7SLKiYXPkVv=-1>Kx+Z51+*5>T0m<7tp&6ex~v7W7R*}cvKGi%AZvlF1+o^p'
        b'tOc<a#99z*L97k2HpJQxYop8B0BZxR4X`%A+5l?<tPQX>x~vVaHoV&KYQw7yuQt5e@M^=W4X-x5+VE<_tKB$F+5l?<Ec_09X$-#b'
        b'JMe`m_<Bykf835;23Z?qZIHD=)&^M{WNnbO)n;v&wPDtVSsP|;n6+=b`EQ`Lfz}3E2WTCjb%53ZS_fzypml)O0a^!W9iVl9)&W`v'
        b'XdR$+fYt$82WTCjb%53ZS_fzypml)O0a^!5)`3|EW*szH2Tj(2SO;Pqh;<;=fmjD(9f)-x)`3_DVjYNe&}1EebpX}@SO;JofOXJh'
        b'9e8!%)qz(xyt?7l4X<u^b;GM0UfuBOhF3Sdx&v?iH^909)(x<3fOP{b{0w|y1iml=U(XTvkK0igV%-qyhFCYmx*^sLv2KWUL#!KO'
        b'-4N@BSU1GFA=VACZisb5tQ%t85bK6mH^jOj)(x?4h;>7(8)DrM>xNi2#JVBY*Yod<`S-^Bdt?5+G5_9}e{amcH|F0P^Y4xM_s0Bt'
        b'WB$D{|K6B?Z_K|p=HDCh?~VEQ#{7F@{=G5(-uMyt#`t?<{Jk;$-WY#xjK4R=-y7rajq&%!_<Q3w;9Jx0jp_G(PQQQN0hKo@Z|w2c'
        b'pT75V{QYwQl@BT(R6f|{gJXSA_eR|tb#LwQjlw`HfK~vl09paG0%!%$3ZNB0D}Yu2tpHj9v;t@a&<daxKr4V&0IdL80ki^W1<(qh'
        b'6+kP1RsgL4S^=~IXa&#;p!N0idt>^&G5y|{es4^_H>Tek)9;Px_kK>lzyII^Vg<wsh!qekAXY%EfLLFTzc<F;8{_Yd@%P5~dt>~)'
        b'G5+2de{YPxH^$!^<L`~}_r~~pWBk1_{@xgWZ;Zb;#@`#`?~U>I#`t?<{Jk;$-WY#xjKBAD{Qdohk-fBl%5&ikFe_nJ!mNZ@3A0j-'
        b')?)zva{-m1Rzj_WS_!ohY9-pN1X>BS5@;pRN}!cMD}h!5tpr*Lv<hey&?=x+K&yaO0j&aB1+)ri70@c6RY0qNRspR7S_QNUXcf>Z'
        b'pjAMtfK~ymLYGzOvI=Asx~zg&1+fZZ6~rotRp_z`T~@)Xf>#Bv3SJewDtJ}!s^C?@tAbYruL@ojyefEA@T%Zd!K;Q>4X+wrHN0wg'
        b')$pp}Rl_Sxzc;4e8`JNN>G#I;du#f=@eA;c;rD(HzkfUs$ZC+)Age)EgRBNw4YC?!HOOj^)gY@uR)eetSq-upWHrcYkkufoK~{sT'
        b'23ZZV8e}!dYLL|+t3g(StOi*PvKnMH$QqC}AZtL@fUE&o1F{BW4agd_SOc*JVhzL^h&5=j24D@q8njpguLfQXyc&2l@M_@Iz^j2*'
        b'1Fr^N4ZIq7HSlWS)xfKPR|BtR;RB@sSOc&IV2wU|ABZ&&YarG@tbtequ?Aud#2S6}J|JsA)&f}zWG#@jK-L0T3uG;jwLsPaSqo$>'
        b'kcGkb#^8Hn@Vznk-WYsu48Au8-y4JPjluWE;Co~6y)pRS7<_LGzBdNn8-wqS!S}}Cdt>muG5FpXd~XcCHwNDugYS*O_r~CRWAME('
        b'_}&<NZw$US2HzWl?~TFt#^8Hn@Vznk-WYsu48Au8-y4JPjluWE;Co~6y)pRS7<_LGzBdNn8-wqS!S}}Cdt>muG5FpXd~XcCHwNDu'
        b'gYS*O_r~CRWAME(_}&<NZw$US2HzXM{@$2-Z_K?ne*L|lWA7igqwc~d{#&2CZ-})a)`nOcVr_`EA=ZXi8)9vUwISAqSQ}z(h_xZs'
        b'hFBY7ZHToY)`3_DVjYNeAl89c2VxzFbs*M(SO;Pqh;<;=fmjD(9f)-x)`3_DVjYNeAl89c2VxzFbs*M(SO;Pqh;<;=fmjD(9duX+'
        b'9oB(Y2VNa`b>P*3R|j4lcy-{_fma7!9e8!%)nn@Y4gCRF2Vfn5b?&7CU+?jK;XUaFSvSbKLDmhjZjg0@tQ%zA`t1FNSvSnOVb%?^'
        b'ZkTn$tQ%(C`sn=zS~t+Tfz}PQZlHB*#NmcoH`KbJ)(y38sC7fF8*1H9>xNo4)ViV84Yh8lbwjNiYTZ!lhFUk&x}nw$wQi_&qszL1'
        b')(y07pmn3m`g-zxF!?^1d>>4{4<_FSlkbDc_rc`*VDf!1`97F@A56XvCf^5>?}N$r!Q}g3@_jJ*KA3zTOui2$-v^WLgUR>7<ojUq'
        b'eV&u=Pv8cl?}O3z!RY&7^nEb;J{WzU=ji*#?Wns@ccJdSa3D4cZxr4rymh1x_V}RiLE*DhK$UM){yGso`0e+>==)&wee~1sgW30a'
        b'&c1)%Vc`y_47UPq1>6d_6>uxyR=}-*TLHHMZUx*5xD{|K;8wt`fLj5#0&WG|`g-_%F#J9kejg0K4~E|d!|#LP_rdV{VEBD7{5}|d'
        b'9}K?_hTjLn?}Op@!B4*rX5R<1?}OR*!R-4yXWu{m;9=I+!|#LP_rdV{VEBD7{5}|d9}K?_hTjLn?}Op@!SMTF_<b<^J{W!<48ISC'
        b'-v`6*gW>nV@cUr+eK7n!7=9lNzYm7r2gC349DaYl5EEi0#7cEq39=ru@1J)-Wtf#PD`8f`tb|#)(0C-!N}%<aegAkMsFhGFp;khz'
        b'gjxx;5^8-t{5}|d9}K?_hTjLn?}Op@!SMTF_<b<^J{W!<48ISC-v`6*gW>nV@cUr+eK7n!7=9lNzYm7r2gC1!;rGGt`(XHeF#J9k'
        b'ejg0K4~E|d!|#LP_rdV{VEBD7{5}|d9}K?_hTjLn?}Op@!SMTF_<b<^J{W!<48ISC-v`6*gW>nV@cTT6-``L^rr$poP#Izs#43nY'
        b'5UU_oL9B*Y4Y3+xHN<L&)ex&87JmJGp7ZY?cR*o~)gY@uR)Z}3_WNM`eem1w^PGPFd>~W?S`D-sXf@Dkpw&RDfmQ>p23ifY8fZ1p'
        b'YM|9XtASPntp-{Rv>IqN&}yL7K&#PYHJYpjSq-upWHrcYkkufoLDqn*0a*jG24oG$8jv+0YtUp3#2Sb-5NjaTK&*jSgC=VL)&Q(Q'
        b'lQr;a;MKsZfmZ{s23`%k8hADEYT(tttASUu@ZszLtN~a9um)faz#4!x0BZo&0IUI61F!~Q4Zs?GmW3aFAI!fGe)xSb{yrFgAB?{b'
        b'#@`3y?}PF8d5*uocWl9|1+x~+S}<$DtOc_c%vvyO!K?+d7R*{OYr(7qvlh%+Fl)iA1+x~+S}<$DtOc_c%vvyO!K{TIYk{l<vKGi%'
        b'AZvlF1+o^%S|DqItOc?b$XXz4fvkldYoW(l0BZrP1+W&t+5l?<tPQX>z}o1sHoV&KYQw7yuQt5e@M^=W4X-x5+VE<_s|~L<yxQ<;'
        b'!>bLiHoV&KYQw9o58pSy+WPQ)LoEF6`(Ofo@VoDW-+dp<zYpf$=Q;oWu|Y6v!>kRnHq6>EYs0J!vo_4yFl)oC4YM}P+AwRwtPQgc'
        b'%sMdZz^nta4$L|*>%go7vkuHUFzdjq1G5gyI*Gre1GEm%Iza0Htpl_U&^kct0IdVG4$wMivJT8TXtEB-I%u*E#5xe`K&%6?4#YZW'
        b'vJSvH0P6s(1F#OjIsoectOKwPz&Zfy0IUPB4!}A9>j119U_Hj)-+z!d#JVBYt<%CAWZfX^23a@A!Y{uM2H*z+@Ph&Pc@DsT7EooN'
        b'g%S9{2>f6KelP+*&k^{K0-AKd5d2^WelP?-7=j-R!O#C!*S#gTvE%?0+|sQ=;kEyb&EJ?}Fn`IAz;4+okUmjH1fPiD6A^qOf=@*7'
        b'i3mOs!6zd4L<FCR;1dyiB7#pu@QDaM5y2-S_(TMsh~N_ud?JERMDU3SJ`uqeBKSfCUx?rf5qu$nFGTQ#2)+=(7b5sV1Yd~Y3lV%F'
        b'f-gkyg$TY7!51R<LIhuk;0qCaA%ZVN@P!Dz5W$xs_(B9<h~Ntme0@gn-ve;Pg~5fvzmFv?WblOyzHt6~p@J_|@P!J#aQb^8f-gky'
        b'g$TZI`g`H@_d)_+IQ_j)z!wVm`Yho8{$LKwIxy?NtOK(S%sMdZz^rrdC+h&M1GEm%Iza0Htpl_U&^kct0IdVG4$wM4>j13-v<}cZ'
        b'K<jq_Unt-U1$?1^FBI^F0=`hd7Yg`70beNK3k7_kfG-sAg#x}%z!wVmLIGbW;0pzOp@1)({$2>+3jur~fG-5_g#f+~z!w7eLI7U~'
        b';0pnKA%HIg@Pz=r5Wp7#_(A|*2;d6=d?A1@1n`9bz7W6{0{B7zUkKm}0emTdFCG70DBugHzZU}d!r||Q{=Lw@7y9=?|6b_d3;lbk'
        b'e=qd!h5o&8_IrKi?_UF2lY!O+S{G<tpml-P1zHzqU7&S=)&*J@Xa&#;pcOzXfK~vl09paG0%!%$3ZNB0D}Yu2tpHj9v;t@a&<dax'
        b'Kr4V&0Ik516<D$YWCh3ykQE>+KvsaP09gUD0%Qfq3M^Rxu>xWR#0o4~fh8;8Rluu&R{^gAUIn}gcopy};8nn@fL9M*J$Uut)q__L'
        b'UOjmAk-z`_1@i#b16U7WJ%IH9)<^#S^@48c2~7U|pRX1I_`>n;h5o%h`}eN_K<fdm2ejVPB}1(TwI0-ZQ0qah2els5dQj^@tp~Lp'
        b')Ot|sL9GY19@KhJ>p`stwI0-ZQ0qah2els5dQj^@tp~LdY9-W4sFhGFp;khz#FmvnD}h!5tpr*Lv=UoZV#`X9mDsWpVkN{%h?Ur~'
        b'5@033N`RHvvJze;yh?cWQNaHNF#%QrtOQsIuo7S;z)FCX04o7j0;~jB39u4iCBRC6l>jRNRsyU9SP8HaU?spRfK>pi09FC40$2sG'
        b'3Sbq$Du7i0>mz^v`xm1iRza+SSOu{PVim+Hh*c1)AXY)Ff>;Hy3St$+Du`7OtFU1ez$$=M0IL920jvU81+WTW6~HQhRRF61RspO6'
        b'SOu^OU=_eBfK>pi09FC40$2^O8elcRYJk-Ms{vL6tOi&Ouo_@BHmrtM4X+wrHN0wg)$pp}Rl}=>R}HTkUNyXGc-8Q#;Z?({hF1-*'
        b'8eTQLYIxP~s^L|`tA<w%uUenHH^6Fu^%1{+eItm~5UU|pL#&2a4Y3+xHN<L&H4tkc)<CR*SOc*JVvRn@8jv+0Ye3e3tN~dAvIb-g'
        b'$QqC}AZtL@fUE&o1F{BW4agdhH6Uw1)_|-5Sp%{LWDUp~kToD{K-Pe)0a*jG1}oM;tbtgA6>9+20IUI61F!~Q4Zs?JH2`Y>)&i^r'
        b'SPQTgU@gG<Xy5<BwAis0U@gE}fVBW?0oDSn1z1Sm3+a1(rte>mV{nuP;EF-kf~*Bu3$hkuEy!AswIFLj)`F}BSqrijWG%>AkhLId'
        b'LDqt-1z8KS7Gy2RT9CCMYeCk6tOZ#MvKD0h?%o^Sd!u`AbnlJsz0tily7xx+-ss*N-Fu^ZZ*=dC?!D2yH@f#m_ulB<8{K=OdvA2_'
        b'jqbhCy*Ik|M)%(6-do*!qkC_3?~U%gark>9dv9d#jqJT~_<N&zZ&dG%>b+6DH>&qW_1>u78`XQGdT&(kjq1Hoz4vGJ{`G@j3|<Ug'
        b'42D<-VjYNeAl89c2VxzFb$-7qY@GYvsNNgZd!u@9RPT-Iy-~e4s`p0q-l*Oi)qA6QZ&dG%>b+6DH>&qW_1>u78`XQGdT&(kjq1Ho'
        b'y*H}&M)lsP-W%0>qk3;t?~UrcQN1^+_eS;JsNNgZd!u@9RPT-Iy-~e4s`p0q-l*Oi)qA6QZ&dI7S-t-S<wC3ru`a~A*s(6ax&Z3}'
        b'tP8L%z`6kI0;~(LZs1qag;*D2U5Ird)`eK;-WvzMH?sFe_TI?e8`*m!dv9d#jqJUVy*IM=M)uyH+56YynA`dZ!>kLlF3h?x>%y!H'
        b'vo6fKFzdps3$rfFx-jd)tP8W^Kiud5T7fMqU{=7afLQ^v0%irw3YZlzD_~Z@tbkbovjR(2fUE#n0kQ&Q1;`34Spl&EVg<wsh!qek'
        b'AXY%EfLH;s0%8Tk3WyaDE3jk*zzTpB04o4i0IUF50k8sK1;7e`6#(m_djEn3Vg<x{*s>nLdI0Mqd;j}_9>jVO>p`psu^z;F5bHs#'
        b'kL>;H0zlRSSr24AtyvFdJ(%@i)`MB^X)w@wK<fdm2ecm0dO+&|tp~Io(0V}Y0j&qL9?*I~>jA9?v>woUK<fdm2ecm0dO+&|tp~Io'
        b'(0V}YVas|j>%pvqSqZZeW+j%a1X&5P5@aRFN|2QxD?wIb$x4Wo5Gx^8V#!K?l>jRNRsyWVl9li(;Z?$`gjWf#5?&>|N_ds<D&bYa'
        b'tAtkxuM%D*yh?bL@G9X|!mET=39k}fCA>;_mGCOzRl=);R|&5QUKPA5cvbMK;8nq^f>#Bv3SJewDtJ{|u?k=nz$$=M0IL920jvU8'
        b'1+WTW6~HQhRRF61RspO6SOu^OU=_eBfK>pi09FC40$2sG3Sbq$Du7i0s{mF3tO8gCunJ%mz$$=M0IL920jvgC4X_$uHNa|s)c~vY'
        b'Z-ou98e%oX>WjBXjTNgQRzs|YSPiinVl~8Sh}96QAy(_7_Xb%FvKnMH$ik`bjqbhCy*Ik|M)%(6-W%O}qkC_3@BP`me_a^RYM|9X'
        b's~agvL#>8d4Ye9-HPmXT)ljRURzt0ZS_8EPY7NvHs5MY)pw?i`8lW{mYk<~Z%^H|BFl%7ez^s8;1G5HZ4a^#tH85+iWevz0kToD{'
        b'K-OT(8i+LzYarG@tbtequ?Aud#2Rc_1F!~Q4Zs>~Sp%;IUd`Y$@Bvr@um)faz#4${(Y=4YAc(d4=fVY93$j+9zAwyLn6)r#5AO8>'
        b'Eu8z_pY8ki6IV~*qOUF*aIHRpU&ytPYa!Q$E*f+#=vvUVpld<bg02N!3%V9`E$CX%wV-Q3*MhDEU8`klVb{X0g<T7~7IrP{TI^a2'
        b'x)yXT=vvUVplh*f{qElf{rjMQAN22o{(aED5Bm2(|32v72mSk?e;@SkgZ_QczYqHNLH|DJ-v|Bspno6q?}Pq*(7zA*_d)+Y=-=nF'
        b'fB*Y<Vs2t?Vs2t?Vs2q>{rsgg=-&tZ`=Eax^zVcIebB!T`u9QqKA-*j*M$uR|K7F+1^l3ZADsI>2;c_+{2+iIo%=qY1^o9DzbBvn'
        b'p1z>;gl5C71Gf&``iS7a2mHR<2XGz0bpY1^TnBI+z;yuE0bB=g9l&(}*8yAya2>#P0M`Lr2XGz0bpY1^TnBLduHXk1{Gfs#RPcie'
        b'eo(;=D)>PKKd9gb75t!rA5`#z3Vu+*4=VUU1wW|Z2NnFFf*(}yg9?6d_WK}$A4Kqj2!0U34<h(M1V4!22NC=rf*(Zig9v^Q!4J-U'
        b'A2jfT27b`M4;uLSY~cTbegW16SQlVjfc4S9e+`CM7h+wAbs^S;SQlblh;<>>g;*D2eKhc2H=>`Oz+{+pVb+CN7iL|Ubz#<pSr=wq'
        b'm~~;+g;@c!0%irw3YZlzD_~Z@tbkbovjS!X%nFzlFe_kIz^s5-0kZ;T1<VSV71*%?WCh3ykQE>+uww<p3WyaDD<D=ttbkYnu|6XB'
        b'Kb{1{3Wychu>w0*z^i~)0j~mHh29zlzzTpB0PCZI{~7?X0%8TkdJyYDtOv0k#Cj0xK`b2pKB(Xa75t!rpU(>ZYXHo8Fzdmr2eTf`'
        b'dNAw3tOv6m%z7~E!K??f9?W_$>%puCvmVTPFzdmr2eTf`dNAu@%X%>D!K??f9?W_$>%puCvmVTP*s&hSdLZk8tcM-zL97R{9>hwB'
        b'l@Kc-Rzj?VSP8KbJ5~a$1Xu~M5<6DHtAtkxuM%D*yh?bL@G9X|!mET=39k}f<--St39u4iCBRC6l>jRNRsyU9SXo#)5@IF9N{E$u'
        b'Uzi{(K~{pS1X&*${MTUZSqZcfXeH1}?O6%6a`mSRY?VHPFSu23tKe3_t%6$xw+e0*+$y+LaI4@}!L5Q@1-B}6qXAa|t^!;IxC(F;'
        b';3~jXShNam72GPgRdB1|R>7@;TLrfYi&lZH0$T;P3X4`jt%6zwwF+t#7OldfRWPeyR>7>oqE#TPuxJ&;YAjj}uo_@Bz-oZi0ILC3'
        b'1FQyE4X_$uHNa|s)c~sjR%6j>c-8Q#;Z?({hF1-*`r(&w1FQyEtq<TEVl~9-EPW$P23ZZVTB}yWtcF<)vl?bK%xaj`Fsor!Ytw3='
        b')j+F(Rs*dDS`D-sXf@Cppfx~ifYt!50a^pJ251e?8lW{mYk<}OtpQpCv<7Gm&>EmMKx=^30IdO91GEN9*1)WRS%W2OK-Pe)!ICu)'
        b'YarG@tih5s0BZo&0IUI61F!~Q4Zs?JHCVC+UJbk&cs1~9;MKsZfmZ{s7G5p9T6neaYSm>ez*>N{0BZpjGWbyjKR5$^J}da|FTiM+'
        b'wJ>X8*21iXS%}~V5&R&6A4Kqj2!0U34<h(M1V4!22NC=rf}hU_{_93St%X_(wH9hE)LN*uP-~&qLal{b3$+$%E!0}5wNPuJ)<Ug?'
        b'TE8RsMFhWy;1?16B7$E;@QVn35y3Aa_(cT2h~O6y{33#1MDU9Uem^7l&%ZE>3Vu<+FDm#&1;41^7Zv=Xf?rhdiwb^G!7nQKMFqd8'
        b';1?DAepc{bH@YymFt{+dFt~8Rjk&)+gI{Fu`<cOie*q@{?%?;cga3X4qer6$qgSIBcY85;akt^tfm<IP{NERJ0M`Lr2XGz0bpY1^'
        b'TnBI+z;yuE0bB=g9l&)2*8yAya2>#P0M`Lr2XGz0bpY1^TnBI+z;yuE0bB=g{jT5_75t)tUsUjm3Vu<+FDm#&1;41^7Zv=Xf?rhd'
        b'iwb^G!7nQKMFqd8;1?DAqJm#k@QVt5QNb@N_(cW3sNfeB{Gx(iRPc)meo?_MD)>bOzo_6B75t)tUsUjm3VuH;_`kqifOP@Z1y~ng'
        b'U4V6ee+s|I;1?PE;wbn<1;41^7Zv<|R`6dhwEoi*y4OJK0<Dh_{_7jHMuV*jwyw6V3%4%Zx^U~ltqZp<+`4e<!mSIpF5J3sE8teZ'
        b't$<qrw*qbj+zPlAa4X<ez^#B=0k;Bf1>6d_6>uxCX9d^_uoYk{z*c~*z@8OQE1*_Dt$<nqwE}7d_N>6371*-^WCh3y>{%hP0%8Tk'
        b'3M^UyumWHOzzTpB04o4i04$X7ixPfO!Y@krMG3zs;TI+R>OA=UOyR$tfLRY_J(%@i)`M9OW<8koVAg|K4`w}>^<dV6Sub>_Lw6c#'
        b'J*f4d)`MCPYCWj+pw@$04{ANA^`O>+S`TVHsP&-M!@I)=v_4w+zdtXa^?=p`S`TPFp!I;(16mJgeYEg@uM1{9nDt;*!mNZ@39~+0'
        b'_&={Rv1KL1N{E#ZD<M`wtb|wzu@Yh>#7c;j5Gx^8LafA=mDsWpUM0Lrc$M%f;Z?$`gjWf#5?&>|N_ds<3Mu>|h2PH<{_6`MRzj?7'
        b'O$J#BvQi(zr#^;948Mrs7cu-IhF_ctzi8nXr^4@N3jg<o7T7AVRsTJGqrw;bzT$#g1-A-r72GPgRdB1|R>7@;TLrfYZWY`rxK(hg'
        b';8ww{f?EZ*3T_qLD!5f}tKe3_t%6$xw+e0*_N)S1g*~gFRza=8o5KQH1+)r#R$<R7>{$h|3St$+D(qPWu?k`p_N)e24X_$}R>P}?'
        b'R}HTkUNyXGc-8Q#;Z?({hF1-*8eTQLYIxP~s^L|`tA<w%uNq$U!#mOjSPifmV11PEUjrakL#&2a4Y3+xHN<L&)ex)IJ2c2@kkufo'
        b'K~{sT23ZZV8e}!dYAjj3vAe$>Pdwgue1ASZ{~bIYcs%iV<8jPAn0qkyVD7=(gSi)TFXmp%y_kD3H!wFaH!wFaH!$~M?!(-Nxes$6'
        b'<|gJQ<|gJQ<|gJA<`(7_<`(7_<~HUw<~HUw<~HUI<__i#<__i#=I;OJ{snpf)&Q&lSOc&YU@gE}fVBW?|AF=Q!WLpJ#9D~8|HS$p'
        b'fGY-B`;V-zZ}iWs?<a80Kx={4h9*O;4PA5SZo{pGTMM`LX)@qiz_oyD0oMYqO<ghMTFAAKYa!P{u7z9+xfXIQ<XXtJkZU2=Lav2e'
        b'`%kXF_jxfFa4q24pK$#TFK_|{'
    ),
    '4.CSV': (
        b'c-n;BLGLt2awq2d0{b22?xBzw8Ih57v^`p^0WB=7wDOs&p=AQ2p}?VKuRpzCa<s#$di$|?7z{gE9hKGne>$o<e_j9Khadm>+n;_r'
        b'|7!eCzy0_B?mvC^&p-V5-+uEq-~I8Qe)>OtJI9xQzC8NB{PCxM`Q7(_`sv$0{^NJQ{rQ2v|MB~O`ls)H{LTOP```cg-JkyS+w03e'
        b'w}=1cyMO)e55FCM^=0e7{vUpMhQIsa|NQaGw*T{gj4%K458wUx)A#@Q?H_*gU;g<0Pk;LDKYaDjZ+`dvzy4C^_3&T+-#>l#$KU@l'
        b'I`^01zxnpReyQ`DFaPgvfBfOUe)pTd|8M{D=l%Qhzy9ZMfBOE1pJ(>VZp{1h{oR-U`R=!M+rz&M{p}Bb`u@vye|+r!fBfOc@Bg14'
        b'{`k{hcKYw05B~Z2;+HMH`K8ot@c-`_|I-h@|L(W{`G-IJ+yDFb|L)7N{+B;|`{iZ$_D?_k=665*^q1rO-(Qgb?c0C-?tlOC!hQW;'
        b'{Bp3r|KmUY_22*7(|-4*z~6uW&p$uy-+lYf|MG|L|M*Y8`P=V){QigE|7CacxjX;sAAbM;J)fWdgO79i58waz&;QrAKYsVyf0%!@'
        b'e%|PpUHIL1KYjcCAMShoyKjH`_RGJ2^UF)~<)80<-u&k$`tHxq^Ud{F*L8j#___9bU~3>X5E>X5I4XWr{HXX*@uTAXm*U_2`~rOU'
        b'=i~p&h^n`$cksDXerxV~>$$edZ%uxydZ~J8?xo_T;x#mpRP|K#RP|K#)YO-XFBM-ZzEpgvc&K=&c&K=&c&PYP@u}id#ixo-6(1@-'
        b'RD7uTQ1OA{N6H;3cck2ra!1OYUzL0Q85}8hq}-8mN6MYn%!S+$a;H^X#vK`Vwu+0mBjS#TJ0k9gxFh0@h&!d?67ER2BjJvOI}+|l'
        b'xFg|?ggX-MNVp^6j)XfB?nt;J;f{nm67ER2Gxd>vMBEW^N5mZwcSPI~aV_Fn#I=ZP5!WKF)yTDoYZ2E+$Gv4-%eZc}QZ3|K$hDAb'
        b'A=g5#g<NmWUCMQ-i)}I2Vy?wpi@6qaE#_LxwU}!$*J7^4T#LCDb1mjt%(a+nG1p?Q#axTI7IQ7;T8&&wxt4M*<yy+Mlxr!srQDWs'
        b'Tgq)Ix24>ca$Cx6DYw<gZ6UXX+*TvEW!#o=TgL4zWwwypLT;;*+cIveliMP0i?}V~HcwW!jN3A9%eXD$cCz~7k(Y8?%Iy_=E**L?'
        b'x5eE4mAQ8jIk$te7j%1WZlc+<q1%)7W!;u_Th{Ht^1`l!T?xAqb|vge*p;v=VOPSggk1@{5_To*O4yaKD`8h^=t|a=tSebpvaV!Z'
        b'$-0ttCF@G<T#338btURb)Ro$~Qae|It^{4Fohvz4a<1fDshuk^S8C@<%9Yx=5^|+>uGG$zh$|6SBCgcVm4r(Pml7^7eY~WIOA(hM'
        b'E=635xD;_I;!?z=h)WTdA}&Q-intVUDdJMZrHD%rmm)5UFS?9N8JDSsDurAMxfF6K<Wk6`kV_$#LN0||3b_<=DdbYfrI1S@mqIRu'
        b'Tnf1qaw+6e$fb}=A(ujK3ArWYmfE-_<CctDGH%JZCF7QiTQY9RxFzG3j9W5p$+#usmW*36ZppYM<CctDGH$7jTOw|WxTQ93Nw_89'
        b'mV{dpZb`T$;g*D35^hPjCE=EYTM}+bxFzA1gj*7BNw_89mV{dpZb`T$;g*D35-ucMNVt%2A>l&8#nA_Ph`11OA>u;Bg@_9g7a}f1'
        b'T!^?3aUtSD#D$0p5f>sZL|llt5OE>mLd1oL3lSG0E<{|2xDat6;zGoQhzk)HA}&N+h`11OA>u;Bg@_9g7a}f1T!^?3aUtSD#7z-5'
        b'Mcfo|Q^ZXXH`T;V2{$F&lyFnRO$j$8+>~%r!c7S`CEVmHxhdkNh?^pAinuA_rihy&Zi=`m;--k3B5sPfDdHx-c9=45%D5@xri`1x'
        b'>cN*?%1tRZrQDQqQ_4*#H>KQ^a#PApDL19ulyXzb4JkJ+9rq#ThL{^-Ziu;YSY6HyIXC3okaI)Mjn-wUA?Svn8-i}Al^b$y$hjfs'
        b'hMXI6ZpgVI=Z2gca&E}EA?Jpi8**-_l^bGih`Ax=hL{^_<%U|hA>@XT8$xafxgq4n<l9}!4JkLI+>ml(u)3HVpP4%fJsgD|jzSMd'
        b'p@*Z;!%^trDD-d?dN}vc!{eA7r5?^b_3$3pdahRaX8Bh6t@2xw->SZK<hP2KikFI)ikFIG9m!GX;oL(HUu~lD$s?bd{8IU)$uCu3'
        b'n*37nQ1MXlQ1MXlQ1PkaQ^lu>PZggkK2&_D_)zho;&SfDxg+O}oI7&v$hjluj+{Gk4zY)$*uzom;VAZS6ni*|Jsiazj$#i-v4^AB'
        b'!%^(vDE4p^dpL?c9K{}vVh=~LhojiTQS9L;_HYz?IEp<S#U74g4@a?wqu9ey?BOW(a1?tuiai{~9*$xUN3n;a*uzom;oM^n?*SE;'
        b'a7V%&33nvik#H^HTEaCJkQ~Jxj$#kz9(#B{myBx}*D|hUT+6tYaV_Io#<h%V8P_tdWn9a+mT@iPTE?}EYZ=!vu4P=yxR!A(<66eG'
        b'jB6R!GOlG@%ea<tE#q3owTx>S*D|hUT+6tYaV_Io#<h%V8P_tdW!#o=TgGj5aa+V~5x3RFZ3(v}+?H@#!fgq+CES*9Tf%J#w<X+`'
        b'a9hG{3AZKOmT+6bZ3(v}+?H@#!fgq6OFX<j2)BsaB5sSgE#kI_+am6kcX;owjN7rfiOLJPE#$V4+d^&&xh>?jklR9T3%M=iwvfAJ'
        b'9bQMVq+Ch4l5!>GO3Ia#D=Ak}uB2Q^xsq}v<x0wxlq)G$Qm&+2Nx718CFM%Wm6R(fS5mH|TuHf-a-}}5gj@-^5^^QvO30OvEA??D'
        b'<4S#8iMSGRCE`lNm53`5S0b)NT#2|6aV6qP#FhHElyE8GQo^N#O9__}E+t$_xRh`y;Znk-gi8sR5-ufNZk?p(-W*YR8J99HWn9WQ'
        b'r5%pa4o7K+!?eRu+TkeeaFlj9N;@2-9nL-N@ZK#kmtro(T#C6Ab1CLh%%zx1F_&U4Ctr3smvS!UT*|qWb1COi&Mi5&<lK^TOU^Ah'
        b'x8&TCb4$)GIk)88l5<PWEjhR3+>&!ko!n9<x1`*Xa!blBDYvBDQYW{B+!At2o!pXfOU5l3w`AOsaksp~^UrySxFzD2h+86ViMS=='
        b'mWW%SQ?@1JmW*3!<(7zBB5sMeCE}KdTOw|WxDat6;zGoQhzk)HA}&PSE${Gp34<$0Ldb=X3$7puDHl>Mq+Ce3ka8jALdu1d3n>>;'
        b'E~H#YxsY-p<wDAZlnW^rQZA%i=zKM#Tu8Z)av|kH%7v5*DHm$wLdb=X3n3RmE`(eNxe#(8<U+`Wkefno3b`rdrjVOLZVI_6<fi(#'
        b'sXlIsxGCbM`nW0Kri7akZc4Z*;iiO}5^hSkDdDDsn-XqHxGCYLgqvI;H$~hOaZ|)i5jREL6me6;O%XRm+~n5|Q^wr_4_|GfR&GkU'
        b'Ddi?lSErbpVs47LDdwh_n__PAn};FihMXI6ZpgVI=Z2gca&E}EA?Jpi8**;QxgqC<oEvg($hjfshMXI6ZpgVI=Z2gca&E}EA?Jpi'
        b'8|vhSm>Xhlh`Ax=hL{^-Ziu-d=7yLX>g0x$8|vhSI=La^hKw8P<c5eFB5sJdA>xLJ8zK&QhgRO9?|Fv@imkvy-vbX{ZKCo=<v$zO'
        b'N<6e~^U(Le!+W<{)mznD)mznDQ^%JN&D%V*5)ZA!L*Ek*@8>E_zBGA!`_Rff^gZ+N8c0=7RZq=5`L;_{U#h-TeX07=)R&5fiie7a'
        b'iie7aicb}vDn3<ws`ym#q2fcuhl&psmvl$cA@R^kJhT!It;9ns@z6><v=R@k#6v6b&`LbC5)ZA!L*Ek*&p)*z=8$-3B_3LdhgRaD'
        b'm3U|+9$JZqR^p+RcxWXaT8W2N;-QszXeAz6iHBC=p_O=OB_3LdhgRaDm3U|+9$JZqR^p+RcxWXa`kr`reNG$^cSPI~aYw`*5qCt~'
        b'5phSv-69XK8Mcg5=ArMIhp$FdU&{5-q3^@`a<1iE%ej_wE$3S8T+6wZb1mmu&b6FtIoEQo<y_0TmUAuVTF$kcYdP0)uH{_Ixt4P+'
        b'=UUFSoNGDPa<1iE%ej_wE$3RDT#LCDb1mjt%(a+nG1p?Q#oSgWx24=xC%4tfZ5g*^+*T*I)yZuMw<X+`a9hG{3AZKOmT+6bZ3(v}'
        b'+?H@#!fgq+CES*9Tf%J#w<X+`a9hG{3AZKOmT+6bZ3(v}oI($+&_gry(7eq<EA`O4%|qW~4_}RN2jraY^Uz8@w2}|4<U`++5APz9'
        b'ZcDn7bS3FZ(v_quNmr7tBwewYD^XXXu0&mlx)OCI>Ppm=s4G!dqOL?;iMkSXCF)Alm8dIGSE8;&U5UC9btURb)Rm|!QCFg_)XbHn'
        b'D>ZYaX0GI1shKM^bERgkgj}haD;ZZZu4G)PnJW=jA}&Q-intVUDdJMZrHD%rmulux!li`E)<;{4xD;_I;%?c8cM%zvGA?CY%D9wq'
        b'DdSScrHo4%moiSthgR~Tm3(L=A6m(WR`Q{BpNCfPp%r{+1s__$hgR^R6?|v~A6mhOR`8(}d}!V0p_O}R<sMqOhgR;Pm3wIA9$LAF'
        b'R_>vdduZhzTDgZ-?xB@?XyqPSxrbKnp_O}R<sMqOhgR;Pm3wIA9$LAFR_>wixre6=OT;Y^w?y0$aZAK45w}F#QX97<+>&rh!Yv86'
        b'B;1m4OTsM)w<O$>a7)513AZHNl5k7HEeW?I+>&rh!Yv86B;1m4OTsM)7ZNTcTu8X!7Y`xgLc|5XcnF@WhL8&(7wqIh%7v5*DHl>M'
        b'xR4~oT&R-^F&APk#9WBE5OX2sLd=Dj3o#dBF2r1jxe#+9=0ePcm<ur%V(u1vc)c0KT!^_4b0OwJ%!QZ>F&APk#M~|T@S3@p3o&;K'
        b'K0Kdlin%G~rkI;*<ffFHQf{h|n`-2yjGHoUs*#%_Zi=`m;--k3YUHMbn-XqHxGCYLgqsp>O1LTEri7akZc4bB`e2(PZi=`m;--k3'
        b'B5sPfDdMJxo9g7Ih?^pAinz(|9j1(%GH%MaDdVP$n`-5zjGHoUs+AiuZpgSH<A#j8g&tmqcL=#5<c5$NLT(7TA>@XT8$xafxgq3+'
        b'kQ+j72)QBThL9UVZV0&{<c5$NLT(7TA>@YIxFO?)j2kj;$haZnhKw6BZpgSH<A#hIGH%GYp*C)axFO<(h#Mkqh`1r*hKL&?4ylK&'
        b')Wg<Y9=1XcTX%Wb$~<gk9=0+MTbYNg%){1A9=0M6TakyY$ir6TVJq^mb&H3sTRiN0;NkH(v6Xn(y2ry-;9)E9uyv1zt-Qll-eD{6'
        b'u$6b%$~)|P-r+r<;!DMsiZ34brRq!7L)AmoLsJhG4;7y(K2?0G_*C(!;zPxUiVqbZDlX%Wj5{*!$hafpj*L4p?#Q@X-r@O%J2LLb'
        b'xFh3^j5{*!$hafpj*L4p4ta;Iyu()BVJq*jm3P?6J8b10w(<^Jd55jM!&cs5EAOzCci751Y~>xc@(x>hhpoKBR^DMN@3573*vdO>'
        b'<sG*24qJD4*!Q@@3-=@9j)*%V?ufV};*N+rBCbVTBfPrrd58BVp#fEwaY{RE-Qr>2!wz3<qViI%rCdw7mU1oSTFSMQYbn=KuBBW{'
        b'xt4M*<yy+Mlxr#1Qm&<3OSzVEE#+FuwUlcq*HW&fTuZr@axLXr%C(ehDc9=bTFAAKYa!P{u7z9+xfXI;$Zhp;TgGh}w`JUxaa(=d'
        b'7I9mB+?H@#!fgq+CES*9Tf%J#x3_fPB5sSgE#kI_+ahk4KGe62+qfa&z6Tz@8d3dZ{S1z92L#;~bX(AEL8s8eR_I~hLl5r(&OSJM'
        b'S+`}~mUUa!ZCSTv-IjG*)@@n0W!;u_CF@Gom8>gSSF)~TUCFwVbtUUc)|IR)Sy!^IWL?R+l69qSu0&mlx)OCI>Ppm=y19~crEad&'
        b'&6S)hb#tX|uGGzykSifqLau~dshcYqS2C{D&6T>jl5i#AO2U<dD+yN;?iP1=iCiMCL|lou5^*WwQpBZ*%S#_EDdX~J?kVI_$fb}='
        b'A*aN{z9$~O8c}&Mmts!0d)SIRY(*ZnA`e@UhpouNR^(yxZVy|ThkegHyf=|_De00OU5dICbt&pn)TO9PQJ11FMO})z6m==;Qq-lW'
        b'yM-QJM_1IPsJn$8UQeZFE+t(`x|DP&>6WBhl5R=5rDkpkx}|1r$+;!xmYiF1ZmF4DYUY-dTT*UGxus@q3ArWYmXKRQZV9<1<d&Mb'
        b'CF7Qwxh3M3h+AsrmV{dpZb`T$;g*D35^hPjCE=EYTRdT1JYiijZcTl#F7<PZ-$AU&`s(PGn7bt(-p@5yU(l_g&(@H1!S5kL)CHTm'
        b'kaZ#JLe_<>3t1P3lNWZuo-U-_E&TA+CYrps3vn0XF2r4kyAXFF?n2y!xC?O?;x5Eph`SJXA?`xlg}4iG7ve6&U5L97ccF$Z)X;^n'
        b'3pI2h>p~4(sG$o<7m_a2(1jYhkaMAiZeC)3s-v4?ZmOf3Qf^AQ$*)_dn44m5s->G!ZmOl5LT;+1n=)?7xGCeNjGHoU%D5@xri`01'
        b'ZpyeR<ED(8GH%MaDdVP$n=)?7xGCeNjGHoU%D5@xCR@5G<ff3DLT(DVDdeV*n?i01xhdqPkefno3b~<PZpgSH<A#hIGH%GYp-ygy'
        b'xFO<(h#Mkqh`1r*hKL&?Ziu)c;)aMDB5sJdA>xLJ8zOFqxFO<(h#Mkqh`1r*hKL&?Ziu)c;)aMDB5sJdA>xMGxWSXwA>xLJ8zOFq'
        b'xFO<(h#Mkqh`1r*hKNJ@q3-F2$H6MaA4>6uQv9J5e<;NtO7Vxf#~<GNdsO|XdUNir@~yeIs<*0d%^izMO1FF{1t5yIe5iW@;;RwX'
        b'FV!#AFU>w#K6UJqFMX<fs{GQyU#h+|_od=X#Y4qI#Y4qI#Y4rXicb}vDn3<ws`yaxq2fcuhl-0jgdj>Gh*Aim6oM#)AW9*KQV61S'
        b'(}z+Bq7;HCg&^u4f_VK&9YJ>l-4S$0&>;m;N<oxT5Tz7EDFsnVL6lMur4&Rd1yM>tlu{6-6htWnQA$CSQV^vSL@5PPN<oxT5Tz7E'
        b'DFsnVL6lMubx%RO-@G#J$hafp4!(gX#UM&Ch*Auq6oV+mAWAWa;%y&_If&wYA4)-px(6ZNn+UoVbj_BoC0$FpmUOL-t_58Sx)yXT'
        b'=vvUVpld<bg02N!3%V9`E$CX%wV-Q3*MhDET?@JvbS>yw(6yjzLDzz=1ziie7Idv%uH{@${;dnTRx{UfuH{_Ixt4P+=UUEfHFI0d'
        b'+*UKUh1^y%w`JUxaa+yY7I9m|Z4tLc+!k?L#BC9`Mcfu~drS8%<F<_3GH%PbE#tO~QxKvUgeV0eib06dy&p>VekjEtN->C145Ac+'
        b'D8?X4F^J;5AL^cicyFTef^G}Ct%h#Pxh?0moZE74%Q@Zlp%jBC#UM&Ch*Auq6oV+mAWAWaQVgOLgDAxyN->C145Ac+D8(R3F^EzO'
        b'q7;KD#UM&Ch*Auq6oV+mAWAWaQVgOLgDAxyN->C145Ac+D8(R3F^EzOq7;KD#UM&Ch*Auq6oV+mAWAWaQVgOLgDAxyN->C145Ac+'
        b'D8(R3F^JM#A4(~RQVOD!f+(dRN-2m^3Zj&PDBbm;6oM#*AW9*KQV62%A&9S@OXc|$L<+eSaw+6e$fb}=A(uifg<J}`6mlu#Qplx{'
        b'OCgs+E`?kQxlF$5QZA)jO1YGBDdkeirIbr4mr^dJTuQl=aw+9f%B7S`DVI_%rCdt6lyWKMmXupkZb`W%<(8CN>f@G>TS9KBk6SWs'
        b'sgGMCZi%=h;+BY8B5sMeCE}KdTOw|WxFzD2h+86ViMS==ZV8ANSWCt&HF8VDEfKdw+!Aq1#4QoGMBEZ_OT;Y^w?y0$aZAK45w}F#'
        b'5^+n!EfKdw+!Aq1#D$0p5f>sZL|pKThmdi)??dsv4|UHzd^Mukvy}@u7jiD-T*$eQbHUXlA?QNTg`f*T7lJMXT?o1mbRp<My<Et-'
        b'kaHpDLe7Pp3pp2ZF63OuxsY=q=R(efoC`S@axUat$hnYnA?HHQg`5jH7jkaOxhdzSoSSlPs+F5!ZmN}=YUL)8n?i1?mzy$f%D5@x'
        b'ri`01Zqm3Z<ED(8TroGfVs1*gDdleIhp(PXvlnxd{oIsuQ_jug>=$P*>87Ncl5R@6De0!9o04uyx+&?Vq??j%O1dfOrlgyaZc4f-'
        b'>87Ncl5R-4A?b#s8<K8Fx*_R?q#KfM=%jTBx*_O>pc{g22)ZHYhM*fdX&rKI$ho0jZm5?VQf^4Op<ZqXxgq3+dbuIvhKw8P<%W8>'
        b'A>oFE8xn3vxFO+&gc}lWNVp;4hJ+guZb-Nx;f9185^hMiA>oFEL;fLk_lH#cAr*hfd;H<?37N`2<URlJ-sGtKQTfC2N99|SZ&hzq'
        b'|BPHJ|B%W*q;3F_ia(^{56SpL>IM+08$hJO52^4&D*TYT0YobMkh%dxD*BL$KBS@#spvy0`jGeN!~5_qRS#7URS#7UO+8e6s`ym#'
        b'sp3<`r-~01A1Xdne5kmXJ7VsLxm)((`O+f$kjg%!vJa{3Ln`}_%08sB52@@!>IM+0=tC;{kcvK}q7SL)Ln``^iaw;G52@%wD*BL$'
        b'KBS@#spvy0`jCn~q@oY0=tC;{kcvK}q7SL)Ln``^iaw;G52@%wD*BL$KBR5{kxD+Kk`JlmL+btyso+B@_>c-d<URQC`T#j1?ufV+'
        b'aV_Fnom`8!7IDo+u4P=yxR!A(<66eGjB6R!GOlG@%ea<tE#q3owTx>S*D|hUT+6tYaV_Io#<h%V8P_tdWn9a+mT@iPTE?}EYZ=!v'
        b'u4P=yxR!A(<66eGjB6R!GOlG@tBY$9*CK9<xGmzgh}$A=i@2>WZcDf=;kJa^5^hVlE#bC=+Y)X|xGmwfgxeBsOSmoJwuIXfZcDf='
        b';kJa^5^hVlE#bC=+Y)XseWq>^w^wo#m6vf_#%&q5W!#o=TgGh}w`JUxaa+c18MkHJmT_CgZ5g*^T*<gXel-<%NCh5JfrnJ!Ar*K?'
        b'1s+m?hg9Gp6?jMm9#VmaRNx^Mct`~vQh|q5;2{-wNCh5JfrnJ!Ar*K?1s+m?hg9Gp6?jMm9#VmaRNx^Mct`~vQh|q5;2{-wNCh5J'
        b'frnJ!Ar*K?1s+m?hg9Gp6?jMm9#VmaRNx^Mct`~vQh|q5;2{-wNCh5JfrnJ!A@6~QmzF8wQpBZ*OA(hME=635xD;_I;!?z=h)WTd'
        b'A}&Q-@*9VgaVg_c#$^SIXzoHTg<J}`6mlu#Qplx{OCgs+E`{7J@9;iYQZA)jO1YGBDdkeirIbr4mr^dJTuQl=aw+9f$}K6kq}-Bn'
        b'OUf<vaZAW8A-9Cw5^_t(Eg`pr+!At2$SonagxnHxOUNxDx75ci8MkEIl5tCY+!Aq1#4QoGMBEZ_OT;Y^w?y0$aZAK45w}R(5^+n!'
        b'EfKdw+!Aq1#Oa0)sl-Dn@sLV9q!JIQ#6v3akV-tH5)Y}wLn`r*N<5@)^^giYq;B<)y46GKRu9QrJtX4}sklQb?vRQ*q~Z>#xI^kr'
        b'52>_6D(#R;JEYPMskB2X?T|`4q|y$lv_mTGkV-qG(hjM#Ln`f%N;{;|4ym+5D(#R;JEYPMskB2X?T|`4q|y$lv_mTGkV-qG(hjM#'
        b'Ln`f%N;{;|4ym+5D(#R;JEYPMskB2X?T|`4q|y$lv_tAX52>(2D(sL7JEX!6sjx#T?2rmOr0(;O$~vU74ypS*r0(;ON;;&H4#}iL'
        b'D(R3)I;3v&kP14af)1%0J)~~*kcv5^Vh(waIlQMX<ED(8GH%MaDdVP$n=)?7xXDIt3b`rdrjVOLZVI_6<ff1tLT(7TA>@XT8$xaf'
        b'xgq3+kQ+j72)QBThL9UVZV0&{<c5$NLT(7TA>@XT8$xafxgq3++PER(hKw6BZpgSH<A#hIGH%GYA>)SHxFO<(h#Mkqh`1r*M(DI<'
        b'$haZnhWfZ6;)aMDL+4CG#tj*Fi#mMuTp!CXWgV8X4og{wrCU8LMIDxI^{|w5SV}r9CLNZN4ogXgrKH1B(qSp-u#|LI_oTynf48dR'
        b'S2LHg4(pzEcny^1UaDTKUaF3t&0GpQEQKAG!VXJeho!K?QrKZB?64GeSh~-{x@R5UQx8=SRS#7URS!*ls`ym#sp3<`r-~01A1Xdn'
        b'e5kmbJ96&Gxg+O}oI7&v$hjluj+{f-VJYme6n0n&J1m7AmckB8VTYx#!&2B`DeSNmc328KEQKAG!VXJeho!K?QrKZB?64GeSPDBV'
        b'g&mf{4ohK&rLe<N*kLK`uoQM!3Og)?9o9YU@Y3>#xFh0@h&v+gh`1x-ZfS@2{>r!`<Bp6wGVaK@mT@iPTE?}EYZ=!vu4P=yxR!A('
        b'<66eGjB6R!GOlG@%ea<tE#q3owTx>S*D|hUT+6tYaV_Io#<h%V8P_tdWn9a+mT@iPTE?}EYZ=$-;#$PDh-(qoBCbVTi@00Z;VE28'
        b'xR!7&;abAAgxeBsOSmoJwuIXfZcDf=;kJa^5^hVlE#bC=+Y)X|xGmwfgxeBsOSmoJwuIXfZcDf=;dbdF!4`44!NXGAVJYsg6n9wn'
        b'xWl{1VtFaIrQBYey_nm<@^Ws=xh?0moZE74%eg((NbbqW>wI-fx-IFpq}!5iOS+PDCFx4im82_4SCXzIT}ir<bS3FZ(v_quNmr7t'
        b'Bwb0ml5{2MO4602D@j+9t|VQlmn%V6>g7t#m7FU%SL)?Ty<ADTl5(Y9u7q5vmn#`pGOlD?$+(hnCF6?5m72K{aV6qP#FdCE5mzEE'
        b'MO=!w6mcoyQpBZ*OA(hME=635xD;_I;!?z=h)WTdA};ZZnM;v}rO3l#<Y6iDuoQV%iaadc;$bQAu#|XMN<1tj9+nahONoc2#KThJ'
        b'VJY#jlz3Q5JS^SeVJYyi6nIz)JS+tsmI4n;frq8Q!&2a3De$lqcvuQNECn8x0uM`pho!*7Qs7}J@URqkSPDEW1s;|H4@-fErNF~d'
        b';9)87uoQS$3Op<Y9+m<ROM!=_z{67DVJYyi6nIz)JS+tsmI4n;frq8Q!&2a3De$lqcvuQNECn8x0uM`pho!*7Qs7}J@UR$oSPDEW'
        b'1s>Kt@bJ}h>4WVSdH8BX^`+dBa!blBDYvBDl5$JREh!h5rXJVkNb1Rgo2b5^3qcoxE(Bc&x)5|B=t9tipbJ44f-VGI2)Yn-A?QNT'
        b'g`f*T7lJMXT?o1mbRp<M(1oB2K^KB91YHQa5OkqdF63Ouxlk(?YUM($TnM=kav|hG$c2y#As0d}gj@)@sa9^Pm75}Ns+F4(Zc4Z*'
        b';iiO}5^hSkDdDDsn-XqHxGCYLgqu8Fog!|ExGCbMh?^pAinuA_rihy&Zi=`m;%4eYb;`IY<ED(8GH%Ma$z^g=$j#u(F6E|_n^JB{'
        b'xhdtQl$%m+O1UZJrj(mfZc4c+<%X0SQf^4OA?1dY8&Ymaxgq6-lp9iRNVy^9hLjspZb-Qy<%asWA>@XT8$xafxgq3+kQ+j72)QBT'
        b'hL9UVZm5qNGH%GYA>)RO8!~RFj~gOxsE-?5CpSdg5OG7q4G}l^#lw(sL&gmmH)Pz9aSA;wg&snohq#9x9?uo`)WcUJp(j)Mqw<I4'
        b'KRXx9J%n-(q1;0#_YlfGgmMp|+(YON524sYF!m6NJ%nNpq1Zzx_7J+mLn!qSy2C>#^biU?ghCIY&_gKn5DGnnLJy(PLn!nRy1_#z'
        b'^AO5Bgfb7I%tI*i5XwA+G7q85Ln!kQ$~=TJ524IMDDx1?JcKe2q0B=l^AO5Bgfb7I%tI*i5XwA+G7q85Ln!kQ$~?q9^YFCfh`1x-'
        b'j)*%V?ua;K9zvOiQ05_&c?e}5LYaq9<{^}M2xT5ZnTJs3A(VLtWgbGAhfwArlz9kc9zvOixMv<-A5teaM^s+M9T|6I+%5F*t}f)x'
        b'3~r+GQtn8(Bjt{iJ5sKtTuZr@axLW=znmG0J%nNpagRNG^;{}1=UUFSTDg{UE$3R!wVZ1?*KFll(6yjzLDzz=1ziie7IZD>TF|wi'
        b'YeCn7t_58Sx)yXT=vvUVTDev$*J7^4T#LCDbFEgcrCh6(Ya!P{?iPG_zpJHOOSzVETgq)Ix24>ca$CLJ7IIt2ZS`_n#%&q5W!#o='
        b'TgGh}w`JUxaeM1U-)iQzjN3A9m%55A<hGDg_92vg2;J!+6nzLqAL1T;cvly5Tg+`Sx5b>25255k=uQuD4?et~E0QC6<t5#gbej#`'
        b'7Ij<H?WwM2%et+GZi~7t>b9sWQCFg_L|uux5_Ki&O4OC8D^XXXu0&mlx)OCI>Ppm=s4KN|CFx4im82_mb0z3X(3QHml5-{JO3sy>'
        b'D|K@v=1R<!m@6??>gGzym6R)WbER&sWL(L(l5wSOu0&jkxKcM)60Rg%Nw|`5CE-%SrG!ffml7@|TuQi<a4F$Z!li^u36~P?mV9`9'
        b'NTrCoMIYXq$hc%XmqIRuTnf2lJC{-}rCdt6lyWKMQp%;2ODUI9E~Q*bxs-A#<#O?57jr4*Qp}~8OEH&XF2!7mxm)<*H9;|#VlKs8'
        b'in$bXDdv`#TVigBxh3Y7m|J3QiMb`_mY7>&Zi%@i=9U_{CFPbHxh3S58o4FomW*36ZmE%5B5sMeCE}KdTOw|WxFzD2h+86ViMU(%'
        b';pNhjaZAQ6b#hC@EfKdwoWc*G+dYJC_YjIc1aJ2c%07g$4{^^vyq`<TEh)F8+>&w%KZI`h5XwG;Zuk(2K7^tVq3A;>`VfjfgrX0j'
        b'=tC&_5Q;v8q7R|yLn!(Xiavy*525HoDEbhJK7^tVq3A;>`VfjfgrX0j=tC&_5Q;v8q7R|yLn!(Xiavy*525HoDEbhJKEyrx@cQFa'
        b'BNsw0gj@)@P$L&IE@WJ&kqZ$QA}&N+h`6anZc4Z*;iiO}5^hSkDdDDsn-XqHxY@#einuA_rihy&Zi=`m;wHagy=5Q18c}&6H-+3R'
        b'mX~r<%1tRZrQDQqlixl}F*g@qcsVy0C(rL6rlgx(P%`-i#8gK&W!;o@Q`SvcH`&llVK;@{6n0bC-4YP*>e6mXyCLm{v>VcHNV_5J'
        b'hO`^fZb-W!?S`})>gR^A8^Ue~yITa}`9ck0H-z0#KR49R4N*5l-4Jy{{oIgrL(&aNHzeJVbVL2z5OhQR+>moa{oGJLH>BK9KR49R'
        b'4H-9N+>miY#tj)a)XxnOH$>bJaYMun5;sH~0uWOHh^f0iOy(b^@()w_hpGI-)NLQ;J^t{nepLOadaHV?`e)?kJ^t|CtycA|>RZ(}'
        b'=e{*}{QA;V0AeZtF?rvIsRYDS0%9rwF_nOrN<hqe0^+?}sp@FyrZNyy8HlM2#8d`iDg!Z<ftbobOl2UZG7wW4h^Y+3R0d)y12L6>'
        b'n94v*Wgw<95K|e5sSLza24X4$F_nRs%0NtIAf_@9QyGY<48&9hVk!eMm4TSL@559EVk!eMm4TSbKul#IrZNyy8HlM2#8d`iDg!Z<'
        b'ftbobOl2UZG7wW4h^Y+3R0d)y12L6>n94v*Wgw<95K|e5srx=m-S=TC0Wp<;m`Xs*djjHx`w?+R#2pc*`#wx1Am%**@zo|OFXWDp'
        b'Ya!P{u7z9+xfXIQ<XXtJkZU2=M+d&8TuZr@axLXr%C(ehDc4f2rCdw7mU1oSTFSMQYbn=KuBBW{xt4M*<yy+Mlxr#1Qm&<3OSzVE'
        b'E#+FuwUlf1aV_Lp$hDAb^>HoZT76uLxE66O;<kv}B5sSgE#kI_+ahj@xGmzgh}-JpwuIXfZcDf=;kJa^5^hVlE#bC=+dNy{B5sSg'
        b'E#ee^n2JA4-uhuG|1j_Qhp#r#?4{i1>FO49Tg)i{F%^KA3P4N-Af|5pFz@+?uQpM6Nw+24mULUvZArH!-IjD)(rrn%C0$9nl5{2M'
        b'>e6v9QCFg_L|uux5_Ki&O4OC8D^XXXu0&mlx)OCI>Ppm=nz@p6CFx4im72K{bfspl<XowlD=}AMuGGwxlq)G$YUWDFm5?hnb0y<S'
        b'&0LAN5^*KsO2n0jD-l;Bu0&jkxDs(C;!?z=h)WTdYUc7%SCJwvMO=!w6mcoyQpBZ*QvhP#0}x-0sQgw_ky0+DTuQl=aw+9f%B7S`'
        b'DVI_%rCdt6lyWKMQp%;2ODUI9E~Q*bxs-A#<x<L}luId>QZA)jO1YGBDdkeirIbr4cZ)x~{&1w+l5$JREh)F8+)^L6gxnHxOUNxD'
        b'w}jjha_jTYFHOZCrs5A%@rSAS!&LlXD*iAPf0&9tOvN9j;tx~thpG6(RQzEo{xB7Pn2JA4#UG~P4^#1nse3<6r5~o!4^!!fsr19V'
        b'rypLQcT2=A5w}F#5^+n!EfKdw+!Aq1#4QoGMBGv%w?tfsxDat6;zGm)8@Z5iA>%^Eg^UXs7cwr?$c2mx_HiNPLdeC|8$P65NV$-5'
        b'A>~5Kg_H{^7g8>yTu8Z)av|kH%7v5*DHl>Mq+Ce3kaD3uE`(eNxe#(8<U+`WkP9IfLN0_{2)Ph)A>=~HO(8dh+!S(C$W8TeQ^rjh'
        b'H)Y(EaZ|=k88>CzlyOsi+!S$B#7*^aQ^G0xFqM6n%05hGAEs{nFz?ZacXb&zW!#i;Q^rjhH)Y(EaZ|=k88>CzlyOtWO&K?3+>~)s'
        b'#!VSFW!#i;Q^rjhH)Y(EaZ|=k88>Czka0uC4H-9N+>miY#tj)aWZaN(L&gmmH)Pz9af5x_5OPDv4Iwv#+)x`gWZaN(L&gmmH`K-r'
        b'5jRBK5OG7q4G}j)+z@d?#0?QQMBET@L&OacH$>bJaYMun5jRBK5OG7q4G}j)+~6uXr5>hI4^yd!>(7yZ_%aahfvthmKxkn6W#F;D'
        b'*Pj>aejZWzqw**CTmzf5Cf{4n)hgdC-<<wd{muGY^|y}xX8BV2(!noPFI6uc`&98%@l^3t@l^4p;)@4<B}Y_#@yG|?_rV<q?m+OZ'
        b'7u<u%J(xQFQ^$Ym08AbKq2oVPe5m+P@$tF%5qU@C9g%nL<lPVAxs&%CxQX`~*c$k?cs>V00|Ns$@6K~zYm3wtp@CnE&&RR4VfXWe'
        b'h1<Z^K;8!OJ`ftX7vHymdL8y(2kwXc{!};WUIV`tU(cr+w}JaSZq%*EQ{AXrkKMgdx4yibw}Ja>{-|+(&F?jC)a9eby1(Z48Y?yM'
        b'Yw>!_{07}~;O5+GU~3>X5SU;L9K?Bl%^w4L!%h9EO$dAL#Jxs_{?doa5546^V2b1Of$zI<`?f{jjqBTs+jry6bKuwF*KK$EZrpX-'
        b'-M$-lJ@)Em+<6Y<ZLb~!ahu0uKr`?2@qN-y;~oQ_-|u}luAT$qwpaIon{oX(>{~P6_rreQs~vbfZpJ+a>fotLuX)`M`~9V@z}u$='
        b'4oTks((lD{<Auu9K<KIN#qWEy4&Eg<<Mv~MH{)Iddc!Tv{9)W<U}JYfGk+NO*kXKSzwgGaNAbBoe&+PuxYs~@{P^lSanFHqK7JDR'
        b'UATD61H$!PxOhI*_&Bid!redW_g%O<ulp|Cd_L9ZUS_L{+jrr{<EakX_kAbsn`f2#iR)`*Xym5cH_yr3PhFq$zd85Ky=~f$bJyp='
        b'e4e}BpKU)*AnwohJbAr8+YP&K)?=@4+I_R0b;f<<UgzfBH}^OHzJvEoKA!C+-Z%Mtwz!Yn_xfhuH~FY@L+_h<j@<TOKW2Jk@0)tm'
        b'N$x=o&2$|d(dU`$eWuSd*?XPES7C7v?x(WPqjNu%ea`fLD!ZQ}w>{{`eD0^R{it(4l|7HWz9IO{d2W*&x$VLA*n@kkcRlLdAbjoh'
        b'O~TjfGg5!2<MBSY|J4sPo27Hw(m8G2HGDl)Xp4a@a86sgh*x(EUt7E_{B@J@`54#3!uwxgJuJNcMb*Q?^&G+9Ks_vcj%@rD*2}`Y'
        b'kGwDE{5ldG85*ghI<>Xe#bfZa@UhMK>}S0!yzg7Wfue==u&|zW@Oe-V3+H36*Z%xDTD+=P4+~!-1Mg~G)vG&&uT4?|p+j?r@Ohvo'
        b'D6R*D&lhy^O)x`m>gPG{eVf6z;n<%WiF!czd{Ym!CG~*t-pb@1!pEB+?-IU8k|V)7<MY1d*WYRUI<h&EpN~$yzSX$z!NIGL4-nU*'
        b'P6LB{f%q5+KHJd9<{s?Lk>m)E!5%tp#aE%WUb#xW+7)_-)cD+p=L5vo2sp|Ii1{1=NBIEpIkFB%Hb;^p+=InEcs7TR5gvm`j)abA'
        b'@Ku=fVA6xh_ratGlOm17k<AhA!QeabY2vRV>t*7&k8F<MRag%bUnAr5tG)O=?^27n^gZvoi})J&+@G$yhp&OHfz&`~K*jML@48F)'
        b'8b}?i9Xi09hrSL+wqD5Mi(h>41>quu>jB|wBsmfs;U27`y<R*9D|Or!Uj;$9*cvW=;k(ouuDgS;0lf*Q-h`8{z)W!>1lJ3}`!<8G'
        b'z~KCc{s_nC>vii+;C%e(=k@u!$I!*XcwQ`g41{W^`1$;;YuxS2=Rj)UGe4mvE+P9u_J!;VedQa^uYBWu|Jtv=r;WRP@wz9f20{a$'
        b'zw(Vce9z*g*Sr!FEX}+I22!tiZoErUZ@3K1BQ^7NG@!%2Qd@)uR6KOpLx(*y^U$AK=ud4P4Q#v~Q?L0{@u}iN#fN4-H1iP}_*{J6'
        b'g*&ew#MQuOWTqCiOkDt+`Z{E)yPN9nrn<W+;ieYx&AV_Penjo#fQq+@Z+x!L$BAyfe)fnX!4Z5DG#~xlYXqO|ll|R%o#=k`!xbO5'
        b'N$!D8bmybLdyND~hW2e8j_??)<Ooi&=A*xRj>O>z_aKtna1VleFu4bldoXznCih@+4|JkCUnjcpUXA{Fo1gb!eSMkc-_|mJ82R%;'
        b'->5Gm>pikLl8=$RjRZ#qM$T&w;<YsG^T=}##%pQQ=aKs!-2Y1Yu?P7Wx$i+dMv^19Js4koFaKd+-|IiW0N)x&4TJ_zyza%vSKpj`'
        b'9MIJ3aqyob10%_UpO1rozPf(*c_cV8@WxoLH{j#40!NY~!4d62Joez(Qf-bTM*>GQ_$tiD9>iB4qn}6e^BZL9FJS6T_{-mB)X%2n'
        b'*KgbZJ}@2^DSsJQUk${4AYNxAJLb<%b?#gIe1D0nv)>!6`{TF&zVLYC@$3Hh{Q&+P*8MW_^^yJg*H|!6w}HG5v<9xv1Mw*S7`PWd'
        b'_kq$tYvB4kupY%91MBl%EzNvAioajJPXnpLo;vJ#FMfMv^IrS$`1SGex>tTYek**dTlcz;#~Y7dA0OZMx{t>f9{)Vwjk+82Z5`>>'
        b'k?zOK{1|8r;AK84epLMEJ$6+5=&$S2U)QDL*Pn~$hk@3>^?Bgu+}zJ2U!7q693|91{bk@m<7hm7eSUoYTKOs6!sCU<8;@U~9}m=x'
        b'^?5%7^9|Gu)LrxQ^DNX|sJn2S*M%1>^%AFE;w%lc22{LMyi~kYyi~mK4%@1|amY6g`NkpNDBO6d8+99X8+99X59%J&J*az7_n_{D'
        b'x)<tRsC(%ZzCORbLVShziaTGgf!2VEPZggkK2?0G_*C&w@lf&4ki{Lc=i@5<S+am+0m%ZA#a*&T-JEzFbrW?Hb@TfCcwxSUx`n!h'
        b'x`nzMbvNp6y}P#FU0W}4>)q9QceRSQiZ`ZxFzthB9~|z3!+lWqLfs2>FVwv>;k#mwGnu<$x0C(#Q0zHSda7IT$GOk=_(!$wh&@lc'
        b'el7i|d?W03nQh$#dp>@Bex5+x8()w8|E$-#>-9XRyYuyUrW;?cfrY19sC@%0-)E+u9#70UajdiOc;k7l&yU{#d!8t-&+m!Cd_M!W'
        b'0N4Ux>kinX?k+rz<GeB7#&N#k^?bbWIF57U^*!_R{vXWu;5Z)~=Y!*Xq3(sc7mo9V<9vObZ{FQ{KECjH;qi~BQVFvXr;d3y>v4@`'
        b'79MXretmvCCLUj(_a_REW4?j;F3fkK?n2##`7X>CXO4-ED|O<SI&n;$I3}8|)Olk9tpr*Lv=V6L9j#~Ktx0c9dTY}C`J*Y-cIBh('
        b'dTycOt>Q<o_EGVpnI9EDDt>9^H^W{7n)%fl_}mfHqyKsiEDe+fS_7Y(zj~ZSJ_nWtN&~F{%{){*R6I2E(9A={*XK7}sS&F?Y_EaV'
        b'z=eVQ_-9_At#C3~IGHS*Ocq=#oJ<y2E3j5~)}9mII@%kRw+{B!!QQIf`U7jd(yijHw^{3Lc2xZ6uluO@(aeu#e(9)Rn)!tn{K5-<'
        b'eSD+sJ8I9zKQ;{;*Y!7=vyJ9#-#vRg&qUpcx}Te~jpl5lIor6dzi~pjbwauCjy*O2V_S@EF}B6n7Gqm2+SbMWtsZTwN89Srwl3~('
        b'V77H_e?zm4CT*ih+puio%KpZc{f#U88&~!>uIz7I+26Rbzi~}$<C@r3YqoJ}xh2?^U|WK13AP`d*>ga}aecpaeZO^mzjb}Tb$!2e'
        b'eZO^mzjb}TVOhhnhGh-Q8kRLIYn*5{C~I8VZ(P}LTo`MdXg03wH=40VGuCLv8qHXv8EZ6St&_|~H`eg0;a9`2hF=Z88h$nWYWUUg'
        b'tKnC}uZCX@zZ%!{8+<jcj5V%|HLmG5uIV?fj5V6CM)P&he4RU6&&LaoHy%gbiMkVYC+gz+$OEyCzK%TlI`TYDB%cGAGro^JYQPTI'
        b'I$-Ol{W@yD4&XX~>j18!-s`CMI_kZS>^f?`juuuN=ymRTJ^uU8*Yyvs>z}(`&&RLNk2j8X<7hXIcH<q^I@kw=4+<X?J}7)J;R^@)'
        b'!hybUpf4Qg>to#ud|lw{0$&&Sy1>^3zAo@}fv*c|!Y-@{yRatg!iD_{7xpjoTNf_uU%0S;0j>*hUFf$iT-d+B)^%s=u|J9VCgz)%'
        b'Z(+WL`4;9|m~Uae8}r?$yHR(e?nd24-Ntcl)NRyl)IF$sFyDi^2XznXUZ{Ja?uEJ+>RwoY_W8aHk2^WM9^ZKUe13d>ydR+MNAb9U'
        b'#d`x(o(CIzysO6JHXQE_Fy}z!z?>HfFBM)WytX#boD-!}rBjDHRXbI?P`gySFzr(5Qt7RC*~YtUqx8m0z421F3O5Qj3O5Qj3Lh0d'
        b'D11=(pzuNAONB2KzEJqmt9*T~{P}iRx__2#xTQOE>6Tu)2bgXern`(iZcFw`C`}XwV4rW5_PDp&t3aqS1e*|SO0WsRKHr2+cddKe'
        b'=I&J>G3P|#MBzkX;57?%Vb|vy4(bktx~1XcCWx=cQFrSlhFqWT<oLMt<Lika*X{0-?+s8{blq-w`5w?4`>6K8C&d(B6MWt74*K3e'
        b'wJ+2T02Tl&M(T;S20kD10AT^b0)z#6uz+9z!2*H>I<P<ocDu*!>p%do5MTko0)Pbo3p8K>zXE;*{0jIL@GIa~z^{N`0lxx%1^f#5'
        b'6==Kyd<FOl^j!hH0)1Biug`ZF*Dc9)4|Co0Tz5%-+*keeIO?M7`h2T*-5Xvvlh@tnb-Q}q-~Q*v|5Epr`9ocZwIJ4lSPNn;h_xWr'
        b'f>;Y;Ep%K9V6BC^@M^)U1+SJGuB9`^1+f;yS~_1`AZzJ-alxzwvlh%+I$K;@m7&&xS_^6|oh~l0wZPT_TMKM0^;`>XEx5Jd)`D9L'
        b'Zmmm)8gMPZwE$NFt^`~OxDs$BTCD_J3Ahq)CE!ZHm4GV&R|2jCTnV@ma3$bMz?Fb20av2YN;Fytwi0Y58m)v{iAF1dRsyXAT8Tz0'
        b'(P$;eN|2QxE752r#7c;j5Gx^8qR~o#l>jRNRsyU9Scyg};Z?$`gjWTx3SJewDtJ``AHxN(3Sbq$DqY?xU7#<JRk}c5Fsooz!K{K='
        b'1+xlfm9Ed1uFn_LDs@{0wn~@hOYK$xt^!<DsVBnGhFk@?3Ubv}VbE2et3X$Qt^!>Jx(aj^=qk`vpsPSvfvy5w1-c4!73eC^wL#Yg'
        b'T^n?5(6!NOZOFAD*M?jht=2}Xwc*xAtF^(_23s3!ZLqb$)&^S}t=2}XwSm?~tF>X)Mys{aYHf(MA=XB#wE@-!SQ}t%fVI(TZFsfe'
        b')z;bJ23Q+lZGg1_)&^J`U~Pc40oK;l_zkf(#M%&R>stH<Sv_#58)mhx#W&Du^;@kK5e>E)Y&F<wu+?Cz!B&H<*0+fbw;FCW+-kVh'
        b'aI4`~!>xu}4YwL@HQZ{r)o`ofR>Q4^TdnI_4Y(R`HQ;K%)o8RDZZ+I$xYcm0;a0<~hFcA{8jV(itp-~Swi=CAL#=~G>j15TM(e<='
        b'gGTG1(K-`nEeB*Bkaa-T0a-_9hzDjJv|0yb9guZE)&W@uWF4I#9+-7t*3tT{1GEm%It4rq)H+b>K&_+2TL<mdfm%oBhX-t(t;%rg'
        b'Xbr>xTt{ml4&*wJ>p-ppxenwykn2FM1Gx_5I*{u?t_yNqkn4h67v#Di*9EyQ^ja6-x&YS&xGunTq1U?L)&;jNxOKs;3vOL->w;Ss'
        b'TCEFgU0~}%t93!G3u;|xwJxA_0j&#YU1+r~n03Ld3!T;lvMzL57sR^IX<Y#8LZ@}Xs|#LT@alqB7reUQ)djCEcy(#D)&;Pxk6``V'
        b'?TkJ?9(X+Q_{QU?J5hI{?nK>*x)XH+bw77nfyMH1Uo8LL0F@UiFU)wM@WPB2>V97N5cjq6uMJWgpmL&eVaA2Rg&7y>e(tp5zD)kP'
        b'!A9YY!W)G*CfumosN1O9sN1M}Q1_tjLEVG82X!yhy-@c;-3xVr)&yD;XicCsfz||C6KGAKHG$RyS`%nZp!IoyeB2kvzaB?jkTpTp'
        b'1X&YgO^`J~)=boeSf3Zi2NuT%7RLt`#|IY22NuT%7RLt`#|IY22NuT%7RSeZas2CX)P+}A8z1+z@$UgthFB9~O^7ui78b_GePR4_'
        b'1CTXA7M8`weOdf#gU^KnW(CX&m=!Q9U{=7afLQ^v0%irwx-E%+{m4Af3ZNB0D}Yu2tpHj9v;t@a&<daxKr4V&0IdL80ki^W1<(qh'
        b'6+kP1RsgL4S^=~IXa&#;G+BWrD?nC&tN>X7vI1lUnyi3W0kHyN1)8h?SOKsWz*+!n0jveE7Qk8nYXPhUuol2t6K4$zUDg6v3t%mP'
        b'wE)%vSPNh+fVBYD0$58Yh6`dXh_xWrf>;Y;Er_)s)+&7XFOY==@o`@e|JneRVb+3K3uY~twP4nQSqo-iIecI_d|)|zU^#qXIecI_'
        b'eB77AzaB?jptXQj0<8pE3A7SuCD2Nsl|U<jRsyX=mz6LpVOGMdgjor*5@sdLN|==}D`8fm$x4uwAS*#uf~*8t39=GoC7P^+ScxVp'
        b'0agO61Xu~M5@033N`RFBD*;xb$x3*Y@G9X|!mGRhWdf`OSP8HaU=_eBfK>zMqXn@FVim+Hh*eXC=hgr!>&&o#R)tDKt%6zwwF+w8'
        b'*2BLFgDtFw53Gle`+E4-<B7+0umM*At^!;IxC(F;;3~jX8<&;}auwt%$W@T5AXh=If?Nf;3UU?XD#%rks~}gQ(<;DKfU5vk0j>?W'
        b'HsIRmv^Lz@aBIV@4YxKrt&L7=L#>TY>$VvFdE-2ES{r6<n6+WnhFKe()&^M{oz{j}8)9vUwb5y9fVBbE23Q+lZGg1_)&^J`oz{j|'
        b'8(wXAwc*u<S6i3hH^ABeYXhteur|Qjx&*%=)`nOcVr_`EA=ZXiSPLIm3m^Bj@UO?Wo(Pp;R>Q1@Sq-xqW;M)eG+GU_8fG=jYM9k9'
        b't6^5dtcF<)vl?bK%xaj`Fsor!!>oo`4YL|%HOy+5)iA4JR>Q1@Sq-xqJywIP23ZZV8e}!dYV=qQu^M7E#A=Au5UU~9L63C+)&W=t'
        b'J=TF&2VNa`b>P*3R|j4lcy-{_fma7!9e8!%)qz(BULAOK;MIXw2VNa`b#zvE0M-Fm2Vfn5b#w*(K&%6?4#YYT>p-jnu@1yK5bHp!'
        b'1F>%F-(Lp;WF3%oK-K|S2V@<PbwJhuSqEetkadBq3uIj&>jGI9$htt*1+p%Xb%Cr4WL+TZ0$CTxx<J+ivM!KyfvgK;T_EcMSr^E<'
        b'K-LAaF0@z|#JV8X1+gxOb)m(&0M-StE`W6btP5aW0P6x+7r?pz)&;OGv{)Csy5Q9XuP%6X!K({iUGVCHR~Nkcy#9S*{rkfD_jO<Y'
        b'{=5VH=LPWVz5xEU!RLJR!V>tUUyNV(HSq5N)ehDEybOL}8T`U;#xJabU-wn;uL29RUMRdY>4nOv%8AN}%84l_3K!mDg}Q~hg}Q~h'
        b'8+AA8Zq(hVyHU4Mw^6rIw^6rI_n_`U-GjOZbr0%ZsC%LAg}N8&g02a=Cg}RS3VvY~{K6{ug;nqitKb(_!7r?WUswgdunK-*75u^~'
        b'_=Q#Q3#;H4R>3c<f?rq#zpx5^VHNzsD)@y}@C&Qp7goV9tb$)y1;4Ngeqj~-!YcTMRqzX|;1^cGFRX%JSOvfCtKgp>025$MfHeWu'
        b'1XvSbO@M{phhJC)zqAT|VHNzsD)@y}@aw(`{`EL!eBn?B%(|_De{Yc3U~2=F23rBP0&Io4t$<shW-9<!L~4VL4Nw_!1>_3I6_6_+'
        b'S3s_STmiWPas}iH$Q6()AXh-HfLsB&0&)f93dj}cv;uGi;0nMMfGYr30Ioo%6>uxiX$9B{uoYk{z*c~*K&Q2!)<UPXfYw5%wa{rT'
        b'khMV80$B^4)`D0Ioz?<a3!T=2R|{S(og6NJwE)%vSPNh+fVBYD0$3~Y*{~qif>;Y;Er^As@aw)5{xtxz7R*{Z^+Y(<Kx+Z51+*5>'
        b'T0m<7tp&6e&{{_mhFS}1EvU7i*1A*}Y%Q>rU@O5^f~^Ewsa7lDR>G}>TM4%kZYA7GxRr1#;a0+}M4y#lE5TNRtpr;Mwi0Y5*h=(S'
        b'3AGYxCDcl&mFTk)eOAJ(gjtC`D?wI*tOQvJvJzw^$V!lvAS*#uqR&c*l}CTTAS*#uqR~o-l@O~SRza+SSOu{PjaC7y0$8O>@ddF;'
        b'SK<p~704=(Rk2k7Q&z84D>Q)8P^+L;L9K#X1+_}eR)MW5O&M+#+$y+LYPJe+72qnsRe-AiR{^d9Tm`raa24Pxz*T^&09OI7LaSA9'
        b'tKe3_t%6$xw>I3`aBIV@4YxMj+Hh;bt&K)&gRPB5YeTIKwKmk+XtXxa+Gw;k8m$eoHpto_YlEx}vNp)tXtXxO+Gw;kz}f(71FVfk'
        b'Ys0GzuQt5e@M^=W4X-x5+VE<_tK9&41FQ|OHo)4t62H}I?Q3WNrD4{FSzFiQ8)&tz#W&PyU5jt9)nKc^R;%4=xYcm0bt%39R|BqA'
        b'yVa1ZAy-4LhFlG~8gez{YRJ`)t07lIu7+F<xf*gc<Z8&(kgFkAL#~Eg4Y?X}HRNi@)sU;vX*J+#z}0}O(P=e0twyKSP^+O<L#;-q'
        b'b%54Er*&Y~L8o;<)&W@uWF3)pK-K|S2d&nDSO;PqiFE>>4+mr&kaa-T(MqiYvkuHUFbls8zwq1e3(Mjcmc_68viMhlQf07pz}5j<'
        b'2W%Z(jX!YfXz{}VTnBI+z;yuEQOk88*MVFIavil?2Xr0KbwJkvT?cd>&~<^X3v^we>jGUD=(<4H1-dSrBwmo~f?OBmx**pDxh|X}'
        b'UTC#0xOKs;3vOL#wJxxAfvpQ{U1+r~sC7ZD3u;|xwJx+;7tFd~)&;XJn02Amx<J;2R_lUT7sR?C)&;RHv|1Oyx&YP%ur7gh0jx`>'
        b'h!@1VAl3!3J}-{X`{MX};l%3r#IM6A7RM)k9X_=-KCw1F@$>MhrSXa1hfn-IeBKwvzaB^7z>EWR7wUe#8lPAfpIR56SQnr7b@8vq'
        b'G2=wx#EcVl3v~;13v~;13w1Z@Zq(hVyHR(eZli9aZli9aZlmr&-GjOZbr0$u)V)ymLfs2>FVqEEpO?iamc=KQ#V3}<Czi!0mc=KQ'
        b'#V3}<Czi!0mc=KQ#V3}<Czi!0mc=KQ#V3}<Czi!0mc=KQ#V3}<Czi!0mc=KQ#V3}<Czi!0mc=KQ#V3}<Czi!0mc=KQ#V3}<Czi$M'
        b'eOdhTgJA-!39u%>ngDCI3PY@U@aaE6)&yA-WX;bP;}h%R6YJtrzYd>R7@zoc_{6&S)Vlb@y7<J;!zX?oKJTmIUj@Kc%+v;`47UPq'
        b'1>6d_6>uxyR=}-*TLHHMZUx*5xD{|K;8wt`fLj5#0&WG|3b++;E8teZt$<sBJ}bc1ZBhL5{Ry@LYz6wPfLeh*D}Yu2tpHkqJ}Y2W'
        b'z^p)@6(B1>)&f}zeb$0l3t}yZwa{lRfVI$PEqJxy)q+<GUM+aF;MIax3tlaFwcyo)R|{S(c(vfwf>#S(EqJxy)hc`xE`YTF7M8>('
        b'mc%EP#HW_T=Y2{1dqAbpYpvF#TeZQ~YL$jt3vMmAwcyr*TMKS2xV7Ncf?Er2CEQB5m2fNJR>G}>TM4%kZYA7GxRr1#;a0+}gj)%>'
        b'5^g2jO1PD9E8$kct%O?%w-Rn8+)B8W=(7@RCD=;zSqZfgY9;!tM4y%Dvl3(_`m98sl>jTzXC=H!c$M%f;Z^G7FacHqtOQsIuo7Su'
        b'z$%>^;y2<G%i<Hu;uFi_^S&(pRRCrc%)+wx#IpFrviQWZ_|&rayf2G?4XmL7l!jXcx9}VBiFNUbb@7RH@p)et|9Tv+GT<t}Re-Ai'
        b'R{^d9Tm`raa24Pxz*T^&09OI70$c^S3UC$RD!^5MtI%i_+$y+LaI4@}!L1FqHX5ysMr%W@+oJf#-_eFz8;#ZmS{rC>G+G;GZJ4!T'
        b')<&bX(P(WnS{q<(fVBbEMx(Xi)rMCaUTt`_;njv$8(wXAwc*u<R~ue!c(vixhF2S2ZFsfe)rMCaUTt`_;njv$8(wXAwc!<3#3xq7'
        b'CsxEKR>UWMAU?4mKJokTiS_WQ_3(-H@OfVk|Jnd#HOOj^)#|evW;M)enAI?=VOGPehFJ}>8fG=jYM9k9t6^5dtcF<)vl?bK%xaj`'
        b'Fsor!!>oo`4YL|%HO#s#hkw2gVOGPehFJ}>8fG=jYM9k9tI=aM$ZC+)Age*v0a*uR9gua<V;zWf&|@8)86JprAl5;XbpX~06^2*`'
        b'VjYNeAl89c2VxzFbs*M(SO;Pqh;<;=fmjD(9bJY$APc_{pI8f@SPP$63!nIn_`ENLe-+kM_}g0e_W(-kD*S<42W}m>b&ifU;5vZo'
        b'0ImbL4&XX~>j16;xGunT0j>*hU4ZKXTo>TF&}m(8>w;Ss+`8b_g+}WFTNl{6z}5w}F0gfhtqW{jVCw=~7udSM)&;gMuyuj03yszV'
        b'wJtPT7tp$Z)&;aKpmhnY3!T=5PU`|$7dovAVqFmHf>;+itqWjX0P6x+7r?pz)&;O`YvEr9`lDEdUy3iRg)gjyulri~d*Q-j_|jtd'
        b'!eaQsV))Wx_`+iN!eaQsV)(*h__{BKe-#cCUMT!|HGE+;eBD>WzaBq$qD19X<<H^6!g~0+uZMqaP$*qE+J!k63KtIcM%}F`Zxr4r'
        b'yivGOxG~{I-A3I;-GjOZbr0$u)IF$sq3(sc7wTT93%MrbnviQkuFost3oGIaE8+_);tMO{3oGIaE8+_);tMO{3oGIaE8+_);tMO{'
        b'3oGIaE8+_);tMO{3oGIaE8+_);tMO{3oGIaE8+_);tMO{3oGIaE8+_);tMO{3oGIaKNMg3q4>J5h<^{DGQ^q?YeK9Eu_nZt5NkrL'
        b'39;s-34^Q&vgWn*M4v0iKxM3multJl_rOL8lm=S?wy+|;up++hE8<@p0ImRB0k{Hi1>g$66@V)MS1i2B0l5Nl1>_3I6_6_+S3s_S'
        b'TmiWPas}iH$Q6()AXh-HK&KVxv;uAg+zPlAa4X<ez^#B=0k;Bf1v;$&TLHELomN1tg-&as(^}}X7RXv4Yk{l<vKBh61+f-7tp%_a'
        b'z*+!n0jveE7Qk8nYXPi<PHVxd1+NyoTJQ?L6JPg5@vi}hg;nu&Ulsozz?@;$ZB_j1iGUWC#TR}lzVJ)&g+=j&Uy3jMQheQ)#J_e3'
        b'Y^|dy!>t9k)=_1^wE))wTnlh5z_kEZ0<HvH3Ahq)CE!ZHm4GV&R|2jCTnV@ma3$bMz?Fb20apU91Y8NY5^yEpO2CzXE752r8m$Cd'
        b'iAF2YXeH1}pp|H}5@sbDtpr&KvJzw^$V!lvXtWYyB^s>+SP8HajaI^|gjWf#5?&>|$_t-|1+WTWmCg<eVim+Hh=n!rg*EYoHSu*{'
        b'6aRcX@S#*Nt6)~atb$nuvkGPv%qo~wx)xtRtAJJktpZvFv<hey&?=x+K&yaOsnaT`RZy#-Rza<TS_QQVY8BKfs8#B-3TzeFDzH^x'
        b'tH4%)tpZyGwhC?5hFTkHZM0b%Xl<aif!0QwwPDtVSsP|;v{|<$@sHPigRBj*Hpto_YlEx}vNp)tXtOrN+7N3)tPQa?#M%&RL#z$4'
        b'HpJQxYopED0BZxR4X`%A+5l@eK7=>K+7N3)tc^Zv1FU_}c5H}+--)mLs`%#yAZvrH4YD@KYLL|+t3g(StOi*PvKnNyuEICWYM9k9'
        b'tI=jP%xaj`Fsor!!>rqq_}7GiRs*fun)uffL9K>b4Ye9-HPmXT)ljRURzt0ZS`D=tYBkhqsMS!b(PlN!YM|9XtASPntwxvCFsspJ'
        b'HOOj^)gY@uR)eeqvJS{PAnSmv1F{atIw0$y%Q_J2K&%6?4#YYT>p-jnvCaY<2V@<zSqEYrh;`6r9e{PTM(gN0`~g`9WF3%oK-K|S'
        b'2V@<c8XhguI=Tvfpw@v}2WlOtb)eRPS_f(!sCA&$fm#P@9jJAn)`40FY8|L`pw<PoE~s@utqW>hQ0szP7u33-)&;dLsC7ZD3u;|Z'
        b'>w;Pr)Vk1TT|nysS{Kl|fYt@HE}(Ux%ev5IT_EcMSr^E<K-LAaE|7Jh%ev5IUFfndcy+<63tnCD>Vj7nyt?4k1+V_!y6!EzjVp<w'
        b';FMVR=5ziN)1O_yLi~b@ItDETlk)EBg;xu&7G5p9T6neaYT?zwtKU6*p@%Q@@P!_}@ND=(4qxva{_FLh-w6vje7$q{uh%jB`!C?@'
        b'ox^{B0Lur<2g_H>7hmg(;fvvm;m~kk`1hq%sNoAwhcCqNg&4jN!xv)sLJVK;82;-G6ow1Kh2g?*;|(`<8@r9&#_nKuushft><)Gp'
        b'yNlh$?qU~gy<+%}f4>jddcf8zhX49?unV?+$MA(1z7WF~V)#M~Ux?ugF?=D0FU0VL7`_n07h?EA3}1-h3o(2lhA+hMg&4jN!xv)s'
        b'ddKjepga)kfmjd3`dz~pYWPA8U#Q^=kB2Wj9=_1R7h3p23twpA3oU%TTllXpG{kx!7M>1YcshI`hA+hMr5L^t!xx?oU+)(F>-E6v'
        b'h1anQvkuHUFzdjq1G5gyIxy?NtOK(S%sMdZz^nta4$L|*>%go7vkuHUFzdjq1G5f}tOK$R$T}eFfUE<u4#+wn>wv5SvJS{PAnSmv'
        b'1F{atI=Hb8#5xe`;Kn)t>j10+unuml1FtT;y7214s|&9#yt?q}!mA6fF1)(%>cXoFuP(g0@an>=3$HG`y7214s|&BLek{Ck^Z_hK'
        b'=@T#wvM$KFAnT@vVb+CN7iL|Ubz#<pSr=wqm~~;+g;^J7U6^%Y)`eLYW?h(dVb+CN7iL|UbzxS(tbkbovjS!X%nFzlFe_kIz^s5-'
        b'0kZ;T1<VSV6)-DcR=}))Spl;GW(97nz>O6UD{x~4zzTpB04o4i;KmAg74Yg+z<;<G@G9U{z^i~)0j~mH1-uG)74RzHRluu&R{^gA'
        b'UIn}gcopy};8nt_gjWf#@(<q#6JRC4N`RFQ{z^C@Rzj?VSP8KbVkN{%h?Nj4Ay!_!VUUFYz7W6{0{B7zUkKpq9l(DL1FZyF3A7Su'
        b'CD2Nsl{m5zXeH1}pp`%?fmQ;o1X>BS5@;pRN}!cMD}h!5tpr*Lv=V3~&?=x+K&yaO;m9hORWPeyR^iAhkX0b7KvsdQ0$ByJ3S<??'
        b'Dv(t;vI=4qj;zuTh6S+-Vim+HiB%A*AXY)Ff>;Hy3St$+Du`7Ot4ePeWEIFNkX0b7KvsdQ0$ByJ3S^c3_Pt<M!K{K=1+xlfHOy+5'
        b')iA4dXf@Dkpw&RDfmQ>p23ifY8fd)&_^&qxwHj(Q)M}{JP^+O<L#@V{)j+F(Rs*fZkA@Ai8fG=jYM9k9t6^5dtcF<)vl>@cgRBNw'
        b'4YC?!HOOj^)gY@uR)eg@mDLdI)xUrJfdN?!vKnL!$QqC}AZu`D4a6FVH4tkc)<CR*SOc*JVhzr$0aydD24D@q8vSHA5NjaT=*}9D'
        b'HR`ekW(~}G1@K?LkUEEfS_8EPY7NvHs5MXvkA*J;@Pz=r-U0mAF9f#+ZcXFl8NfA49R_j@<Qm8|kZW*j4d5EUHGpdY*8;8uTno4s'
        b'a4q0kz_oyD0oMYq1zZcb7H}=#TEMk{YXR5d(ptE+aBFdCE!bLIS_`!nYAr6U#ig|{Yhl*Htc6(%vleD8F0BPw3$hlM)<Ud>Sc^+*'
        b'0oLNuT6neaYT?zwtA$q!uU2K&?+CsT!8aoKMg-r8;2RNq>zVM4XTmok_(lZZh~OKKgl{|&zLCH;68J^}-*_T?qkwM|@Qnh#QNTBz'
        b'2j2+b`yIf4y<Yei78p(pCx#Qli8q|sE$kL{3%iBg!fs=?vD?^f>^61>yMx`q?qGMYyVzaqE_N5Y5bJ?h55#&P)&sE~i1oXFZ}jht'
        b'{=Lz^H~RNR|K8}|8~uBue{b~fjsCsSzc>2#M*rUE-y8jVqknJo?~VSw(Z4tP_r}BE8~J-9e{baPjr_flzc=#tM*iN&-y8XRBY$t?'
        b'@BPl-Kff_N0P6u*55Rf=)&sB}fb{^Z2Vgw_>j7BL???T$9tGdX-y8XRzw`I+U+DK+J}~RRtOK(S%sMdZz^wE8e`weU;2QyaBY<xN'
        b'@Qnbz5x_SB_(lNV2;dt5d?SEw1n`Xjz7fDT0{BJ%-w5Cv0emBXZv^m-0KO5xHv;%Z0N)7U8v%SHfNuowjR3w8z&8T;MgZRk;2Qya'
        b'BY<xN@Qnbz5y1C5fdBY&dLh<@SQkgu1y~ngU4V4~)&*D>U|oQ9ab#V1b>Y>8R~KGgcy-~`g;y6|U3i85z0tq-yMO<B9p4OZ{^<jF'
        b'%OLB5tP8R($hsiwf~*U&Zs{GvtlRop|NcK^Z6xrG1itkc_(lWYXy6+Se4~MHH1LfEzR|!p8u)%U@Skri@WWxit$<qrw*qbj+zPlA'
        b'a4Q~McmcQqa0TEBz!iWi09OF609*mM0&oT33cwXOv;uAg+zK38fkP{xRzR(QS^>2JY6a8^s1;BvaA*b43LIL2Ln}a5fULlw6%Z>R'
        b'RzR$PSOKvDVg<wsh!r@r5@033N`RFBD*;vltOQsIuo8z>J`g$qRsyU9SP8HaU?spxfRz9%FN98rl@Kc-RtCQNC&)^Wl^`oYR)VYq'
        b'SqZWdWF^Q-kd+`SK~{pS1X&5P5@aRFN|2QxD?wI*tOQvJvJzw^$V!lvAge%Dfvf^q1+ofc704=(RUoTCR^h}dh*c1)AXY)Ff>;Hy'
        b'3St$+Dx6paunJ%mz$%<r1+NNT6}&2VRq(3dRl%!*R|T&MUKPA5cvbMK;8nq^f>#Bv3SJewDtJ}!s^C?@tAbYruL@ojylQyW@T%ce'
        b'!>fi@4X+wrHN0wg)$pp}Rl}=>R}HTkUNyXGc-8Q#;Z?({hF1-*8eTQLYIxP~s^L|`tA<w%uNq!8ylQyW@T%ce!>fi@4X+wrHN0wg'
        b')$pp}Rl}=>R}HUT_4~(t-tem7Rl}=+R|BsGUJbk&cs1~9;MKsZfmZ{s23`%k8hADEYT(tttASSouLfQXyc&2l@M_@IsK^?CH2`Y>'
        b')&Q&lSOc&IU=6?;fHeSX0M-Dk(O<j|#2Sb-5NjaTK&*jS1F;5T4a6E;SOc*JVhzMvh_!$CLNCZ#khLIdLDqt-1z8KS7Gy2RT9CCM'
        b'YeCk6tOZ#MvKC}5$XbxKAZtO^f~>`fwGe9|)<Ud>SPQWhVlBj4h_w)FA=W~yg;<LdYXQ~*tOZyLuohq~z*>N{0BZr(0;~mC$le>-'
        b'`yhKCWbcFQeUQBmviHHW-v`zEpn4xv?}O@n^yv4|qu&SJ`=EOtbnk=iebBwnyL<oo2f7%(7!C{vh68Unup8J->?U>-yNTVzZeh2u'
        b'Ti7k^7IquEjorp>W4Ez8`Yk;e9t;nL2g8duyx3jrE_Pwo1G65O^}wtLW<4<LfmsjCdSKQAvmTiBJA5C6?}PAt5WWw>_d)nR2;T?c'
        b'`yhNDgztm!eGtA6!uLV=J_z3j;rk$bAB69N@O==z55o6B_&x~V2jTl5e4lst{t3qeu^x!^K&%I1{SMy;;rk$bAB69N@O==z55o6('
        b'hwoosAc*xqtOKzQ#5xe`K&%6?4#YYT>pXbF2V|Y!b;BThA3XSd(7g}3_d)kQ=-vn2`=EOtbnk=iebBuRy7xi%KIq;D-TR<>A9U}7'
        b'?tRd`54!h3_de*}2i^OidmnV~gYJFMy$`zgLH9oB-Ur?LpnD&5?}P4r(7g}3_d)kQ=-vn2`=EOtJotUky$`zgLH9oB-Ur?LpnD&5'
        b'?}P4r(7g}3_d)kQ=-vn2`=EOtbnk=iecs*sC%6}2U4V4~7Q**I_&x~V2hV*UJokN&y$`bYLH0h#-Ur$HytDVOHw>~a$hsiwf~*U&'
        b'P`wYT_jy<EU!MT9F3`F_>jJF{v@X!PK<fgn3$!lK3ZNB0D}Yu2tpHj9v;t@a&<daxKr4V&0IdL80ki^W1<(qh6*#g2W(CX&m=!p('
        b'0%Qfq3Xl~bD?nC&tN>X7vI1lUj;w%K0kHyN1;h%76*#g2U<Hn>fL8&p0$v5Y3V0RpD&SSXtAJMluM%D*ym~e7pU@}3N`RFBD*;vl'
        b'tOQsIuo7S;z)FCX04o7j0;~jB39u4iCBRC6l>jRNR{ky=2Fd#%c^@S2gXDdXybqH1LGnIG-UrG1AbB4o?}Oxhkh~9)_d)VL@8tdS'
        b'>kqLKVkN{%h?Nj4Ayz`Hgjfl&5@Hp^Du`9MunJ%mz$$=M0IL920jvU81+WTWy^{Bj-zNpI3Sbq$Du7i0s{mF3tO8gCunJ%mF06uA'
        b'1+NNT6}&2VRq(3dRl%!*R|T&MUKPA5cvbMK;8o$mDtJ|cAA$>D6~Mwn-v`nAyrcK;PY8Vg(=e<5@SiyvXf@Dkpw&RDfmQ>p23ifY'
        b'8fZ1pYM|A)vKnZ;y7#XyDAa1G)ljRURzt0ZS`D=tYBkhqsMS!bp;kk!hFT4^8frDvYN*vvtD#myt%h0+wHj(Q)M}{JxUw2(HPC8Y'
        b'Sq-xqS61W7YKYYkYj9-^z#4!x0BZo&0IUI61F!~Q4Zs?JH2`Y>)&Q)*l{N5c;MKsZfmZ{s23`%k8hADOv2Xwup7}l~-{)Pve}4kr'
        b'GRPW`H6Uw1)_|-jd`}vfH85*n*1)WRSp%~MW(~|5m^CnKVAjB_fms8y24*eHT9~ykYhl*Htc6(%vleD8%vzYWFl%Af!mNc^3$qqx'
        b'EzDY&wJ>X8*21iXSqrlkH`aoz#f`NPYjI;Oz*>N{0BZr(0;~mCiyLd<)xxWVR|~HeUM;*@c(w3q;nl*cg;xu&7G5p9`W?P6!uLh^'
        b'z6jqJ;rk+dUp?=AQNAzA_kEY|-@nk{0|#&Ucly3a-xuloB7I+^?~C+(@yPdmm+xQypjU4>upC$pEC=3k;2kG+6T6As)H_ZLmwtN{'
        b'mJ7>;cU%~5yyM1hW4E!}*d6Q+b_ctI-NEi+cd@(JUF^cG2W~xZ>w#Mj+<M^F@A!QYzc1qVMf|>q-xu-wB7R@Q?~C|-5x+0u_kG9j'
        b'AAi0cnDsk;U&QZ=_<a$-FXHz_{Jx0a7xDWdeqY4zi}-yJzc1qVMf|>q-xu-wB7R>y^?lL4FWUD-`@U%37w!9^eP6Wi`)=Pqp??P7'
        b'3t#p7_XjWyvL2B2fUE<u4#+wn3r~Gt<nN38eUZPf^7lplzIg2WqJCf0@B6ObzlNdKfm#P@9b8%mY8|L`pw@v}2WlOtb)eRPS_f(!'
        b'sCA&$fm#P@9jJAn)`40FY8|L`pw@v}2WlOtb)eRPS_f(!sC95<9b8!lSJnYp2V@<PbwJhySr=qokacloU5Ird)`eIXSJnks7hqk0'
        b'bph7Jm386Og;y6|U3hij)rD6VUR`)~;njs#7hYX>b>Y>8S2yvE;R37+u&#bCyb$YEzkh!MmJ4^sg;^J7U6^%Y)`eLYW?h(dVb+CN'
        b'7iL|Ubz#<pSr=wqm=!Q9U{=7afLQ^v0%irw3YZlzD_~Z@tXKX1@%u1fR=}))Spl;GW(CX&m=!Q9U{=7afLQ^v0%irw3YZnRu>xcT'
        b'$O_z80kHx%RsgKPjTP`J;8nn@fL8&p0$v5Y3V0RpD&SSXtAJMluL52Lyh?bL@G9X|!mC&O{`n0l0agO61Xu~M5@033N*q}Uuo7S;'
        b'z)FCX04o7j0;~jB39u4iCBRC6l>jRNRsyU9SP8HaU?spxfRz9%0agO61Xu~M5@033N`RFBD*;vltOQsIuo7S;z)FCX04o7j0jvU8'
        b'1+WTW6%MR|R|T&MUKPA5cvbMK;8nq^f>#Bv3SJewDtJ}!s^C?@tAbYruL@ojyefEA@T%Zd!K;E-1+NNT6}&2VRq(3dRl%!j{3u!g'
        b's{mF3tO8gCunJ%mz$*RCdqJ#%SOu|aaq2Y4>c68;2z)}}bu9y}23oDZdvB=KP^+O<t8!?t)nKc^R)ehuTMf1vY&F<wu+?Cz!B&H<'
        b'23rlb8f-P#YOvK{tHD--tp-~Swi;|T*lMuVII|jRHPmXT)ljRURzt0ZT8%TSfmZL`-CwUKUT?g<KVP4}2d`heo_M|SI(8rIKG=P*'
        b'`(XFM?!oTC?!oTC?!oTG?)`7~6VSOBz8Jn34h#q0a3ubLqV)kRCzcb-iFcfM$BEs-Zeh2uTi7k^Hg+4kjorp>V|TDS*d6SS{sj+)'
        b'7sD`XVAjB_fmsW)7G^EXT9~ykYhl*Htc6(%vleD8%vzYWFl%Af!mNc^d$9|$7Gy2RT9CCMYXiH1U4XR!YXQ~*tOZyLuohq~z}o-8'
        b'`uKCY0BZr(0;~mC3$PYoEx=lUwE$}Y)(&>z)xxWVR|~K9r&s>~9MBJM'
    ),
    'DS0001(2).CSV': (
        b'c-n;BU+*+Wb|3b81AY&C@1;;x=TB8%#*qvMkP*RBlxtecC<InZfy*dyUVTWeyia#O^HF%QSWa=~bob2fGpA?Hcltm5@Z&#y`_pgd'
        b'UyT3w+kgM>{>yj&^uv$;{WpL8-5>t(r~mV}bNu|z&$s?BfB5O2|ML4k{`Bo1{^7gd{&K_L{rLSq{^NH){^o!B-S2+<?vH=`?e+6N'
        b'x10a^yMOua_rD!~@$<d^>OcJY2!H#-|Mla~_x+#$^ZNN;{{FikfBOC(zWx1g{<lAT|I;6T`}ben^qar@{$GBr^Sb%3{-2+|`@`>k'
        b'U7h{&@?U@Z-+r$1o1g#R-~9N)fBWt?fA`=2`A^&Tr+@vg-~ROd55Mf$ubVONkN3Af|Ic^7t=ks<dFgL{_~Z9K-|r8P?f=g|{P_L<'
        b'`@<i8`twHr?eoSzJ>L5D9^d?0>bCg*_lW=Hhu?ko+yDB*@Bj7x{ri9S^RfQt-+%k_)9~#dfBMZ|{_xYEkMsX{LjLz}|K+>?<L4*t'
        b'>;K}{gZ=#<{^76w?q47FyPpgE-S_|W%ftTdxBv9dzyJOZ|M;7~`R>Q>fB4;>H#eV~^S}Q7cmMYO{PG`s+^7Ha{g40je}DVqcfbAn'
        b'`4{oaoqpYfzx?i}Z@>TjeXIZS+n>Ju`QN|!^{M&!pYMLT`!5gl-Jc%ko9i#G>-@6tOYQeUYaz9;v@o!6RQ#y;QSqbVN5%J_i~n-`'
        b'zx&hS|8+sdw~B8S-zwhP^H%Xz@mBFx@lx?p@lx?p@lx?r@l^3t@l^3t@lf$l@lf$l@lf%l;!DMsiZ2ykDn3<ws`ym#sp3<`hl&ps'
        b'A1XdneEhliH_yN4^M$ytR#5(YG489oEZqe=e}vo-a!1G=A$Nq_*(xsN&gQO5xwCob#oQ5dr&(Uk9XWU8+>vuf&K)^-<lK>SN6sBN'
        b'cjVlWb0_$c7j#F^9YJ>l-4S$0&>cZ{1l<vIN6;NXcLd!LbVtw~L3ae*5p+k;oq^(8t=txKTdmxda$Cx6DYvEEmU3IlZ7H{<+?H}%'
        b't=#^C-184-tCia_Zp*l>R&I;9E#kI_+ahj@xGmzgh}$A=i?}V~wusvzZmX5s5^hVlE#bC=+e^M~5w}I$7IAy&ZFS4I{l&Pi?xONv'
        b'ko)Q`+WW!1AFMCuww&8?uH{_Ixt4Q%Y3EzewV-Q3*MhDOt4q3;bS>#x(zV*TmUJ!YTGF+oYf0CVt|eVdx|Vb;=~~jYq-#mnlCC9P'
        b'OS+bHE$Ld)wWMoF*Xre3(6yjzLD%Z#TF$k4xfXM+UaqBFtCwr_axLRp#<h$q8CNo{WL&A2D-l;Bu0&jkxKb}y60Rg%Nx0$}Yl*nx'
        b'8EeV7l5r*DO2(CptI`{5Rda=VFXl?jm6)q!`DFQAT;b8@d27kKl658PO4gOED_K{vu4G-wx{`Gz>q^#@tSebpvaV!Z$-0ttCF@ew'
        b'rL0R?m$EKpUCO$Ybt&so)}^dVS(mad)y<`-OLcQ8=~B|Aq)SPck}lQFrMkIPH<w~A#axQHR5zFE=2G2Us+&u7b1C6c!li^u36~Ns'
        b'C0t6llyE8GQo^N#O9_`eX-yHAA}&Q-intVUDdIxJ#icjakZ~d7;?mn{2)Ph)A>=~Hg^&v&7n}RejxNMph`A7R(JU|LLe7Pp3pp2Z'
        b'F63O4Uh^U7LePbv3qcoxF4)Y4qzg$Gk}f1&NV<@8A?ZTWg`^8f7m_X{T}Zl6FBgI?1YHQa5Og8vLeMQix75omIk)88QZKi}+!Aw3'
        b'%q=mu#M}~dOTFAuFSmr;5^_t(E%kCs#w{7QWZaT*OU5l3x75om5w}F#5^+n!EfKd!+!Aq1#H~m^mdXdq3%MoamXKRQZV9<1<d%?I'
        b'LT(AUCFB;rf>=^+4IcTixPtmq&P_Qt<=m8WQ_jt!ou7hk3c4xirl6af)g|4ObaO90mdcB|De5LiN~WxvvTn+{DeI=Jo3d`ox+&|X'
        b'tediK%DSoZ)+y?ysGI8Mrlgzd=BA*Vf^G`BDd?u4n}Ti%x+&<Upqqkjs+*f~ZmOG`>gJ}D8&Ymaxgq6-lp9iRsGA!?ZV0&{<c5$N'
        b'>gI-w8$4$nYUc*Og&0C^2)RMzMlY^t@A*B%5OYJ!4W70RIXC3o;MWjC&<#O11l{0w5F=QgUEPp%L)HyhH)P$Abwkz-SvO?ekaa`W'
        b'4Ous2-H>%d)(u%VWZn2|-BA$YC<t*Bgg6R990ehcf)GbRh@&9HQ4r!N2yyN~i1&huZx!DvzE!+cyj8qayj8qayi~kYyi~kYyi`0@'
        b'JXJhZ9AC8_B_WQI5JyRfqa?&p65=Qcag>BON<thZA&!y|M@fjIB*ak?;wTAml!Q1+LL4O_j*<{ZNr<B)#JMLSo(o6B9T9g#+!1jH'
        b'UqT$^AdYenM>&Y29K=x$;wT4k?m39}_6oVP7w^KI7jvgsU(OvlcjVlWb4Sh{Id|mTk#k4R9XWU89D)!>L5On?LcAALT+AIYcf{Ng'
        b'b4Sb_F?Yn=5pze(9Wi&r+!1p}%pEaz#M}{cN6Z~Dx5eBRb6d=9F}KypZ7H{<+?H}%%55pPrQB90w}sqRC%4tfZ4tNC$!!U@CES*9'
        b'Tf%J#w<X+`a9hG{3AZKOmT+6bZ3(v}+?H@#!fgq+CES*9Tf*I95U=;UE#kI_+ahj@xGmzgh}$A=i?}V~wusvzZi~1UaV_Hd(s6GY'
        b'*D|hUT+6tYaV_Io#<h%V8P_tdWn9a+mT@iPTE?}EYZ=!vu4P=yxR!A(<66eGjB6R!GOlG@%ea<tE#q3owTx>S*D|iv#kGiQ5!WKF'
        b'MO=%x7I7`&TEw-8YZ2EXu0>plxDs)tF0Le8Nw|`5CE-fKm4qt^R}!uyTuHd%scMP15^*KsO2n0jy9FTLn<e8)#+8gK8CNo{WL(L('
        b'l5xdP<VwhukgL!~1SRE4%9WHWDOXaiq+Ch4l5!>GO3Ia#D=Ak}uB2Q^xsq}v<x0w>luId>QZA)jO1YGBDdkeirIbr4mr^dJTuQlA'
        b'AD2Qdg<J}`6mlu#Qhi*?xRh}z<5I?@j7u4pGA?CYs*g(%mm)6J$EAcz36~NsC0t6llyE8GQo^N#O9__}E+t$_xRh`y;Znk-gi8q*'
        b'5-ucMNVt%2A>l&8-GUFV_tX$^A>u;Bg@_9g7a}g$$c2mx85jJ@A%t8s%S*Yy49QXS;VAlW?$L+$f~t$T5OX2sLd=Dj3o#dBF2r1j'
        b'xe#+9=0ePcm<ur%VlKp7h`A7RA?8BNg_sL57h*2NT!^_P=9ZXSVs44KCFYixTWaK%lv`46Nx3EEmXupkZmE%5LT(AUrABVaxTQvJ'
        b'iMXXkZb`UR`V@AFxFzD2h+86ViMYkH)g|MWj9W5p$+%na;ay$GEg`pr+!At2$Somv3qHKJSIR9Zx1`)H`S30>SYFO8Ik)88l5<ne'
        b'O*uE^+>~=u&P_Qt<=m8WQ_f8}H|5-vb5qVuIXC6plyg(gO*uE^+>~=u&P_Qt)yYjUH^tm7^zeMOPcb*e+*BtwrQDQqQ_4*#H`U2a'
        b'AvcBG6mnC@O(8dh+*BtwW!#i;Q^rkoa#Nk$lyFnRO$j$8+>mfX!VL*GB;1g2L&6OSHzeGUa6`fk2{$C%kZ?o74GDLPJiOl4hKL&?'
        b'Ziu)c;)aMDB5sJdA>wX%hxZ<;l^Zf{$haZnhKw6BZpgSH<A#hIGH%GYA>#%INQRIbLT(7TA>@XT8$xafxgq3+kQ+j72sxx3w$ct;'
        b'X@{+}!&cg1EA6nAcGyZgY^5Ex(hgf`hpo##Y^5Ex(hgf`hpn{3R@z}J?XZ=0*h)KWr5(1?4qIu5t+c~d+F>j0u$6Y$N;_<&9kwp}'
        b'uoZUL3Oj6t9k#*_TVaQ-u)|i^VJqyg6?WJPJ8Xp=w!#iuVTY}-!&cZ~E9|fpcG&l@!#A%#!lCLz)kWM9aYw}Ist;RnhpnqVY^5Ex'
        b'(hmEcc6e9E?*wn<9k%igTi1QqiaTsw_F>=C4zC3%cck2ra!1M?DR-pYk#a}M9VvID+>vre${i_pl25vrJ7VsLxg+L|m^)(bh`A%?'
        b'j+i@Q?ufY~=8l*<V(y5!Bj%2nL)c*}?64Jf*a|yrg&nrS4qIV|t+2yZ*kLQ|uoZUL3Oj6t9k#*_TVaQ-u)|i^VJqyg6?WJPJ8Xp='
        b'w!#iuVTY}-!&cZ~E9|fpcGwC#Y=s@R!VX(uhpn)~R@h-H?64Jf*a|yrg&p=i?C|n+i#T2NVJq&i8F$$CxWiW~Ln~rV*L~PZJZvQ%'
        b'wh|9piHEJk!@egT-kT-pwxH`v2frm<OS+bHE$Ld)wWMoF*OIO!T}!%_bS>#x(zT@P=5ZHwE$Uj-wWw=R*P^aPU5mOFbuH>z)U~K<'
        b'QP-lbMO}-!7Im#=t|eWonQKAUYUWzaT&tODDc5S|TFAAKYa!P{u7z9^xfXIQ<VxLK$+(hnCF4rQm5i&ScbDQhYe~6V?BRVEF;|=Q'
        b'<y^@*UHM@v_^=gx*a|*u1s^tp51W^M*u3<^R`g*j`mhy!*or=EUio1w`>^lXhxdZ2%es<vCF@Gom8>gSSF)~TUCFv)OIO0Kgk1@{'
        b'5_To*O4yaKE1kKPtSecUvMyy^%DR+wDeF?!rTV#4KbMj&)z77%OZ9Up=Tgq4`ng;5;rU8TF_&U4#aybLOSN;Ub}nUH%D9wqDdSS@'
        b'T#C39aVg?b#HEN!5tkw^MO=!w6mcoyQpBa&xtw}uOA&X=K74f-m6vfT<5I?@j7u4pL%fHO3n3RmE`(eNxe#(8<bwTN@H>bQb0OwJ'
        b'%!Qa!{9!BpuoZvU_xQtmfjckgLeK@<xsY@r=|a+lqzg$Gk}f1&NV<@8A?ZTWg`^8f7m_X{T}Zl+bRp?N(uI1t5OkqlF63OuxsY=q'
        b'=R(efoD21GOUx}Xx5V5Mb4$!EF}K9rQZKjE%Pk?dgxnHxOUNzta!bZ78MkEIQZKhe+!Aq1z1)&;OTsM)w<O$>a7)513AZHNl5k7H'
        b'EeW?I+>&rh!Yv86B;1m4OTsOlt}ZrnOU5l3w`AOsaZAQ68MpZ1+!At2$W0+Ph1}#o$&_+a%1tRZrQDQqQ_4*#H>KQ^a#PApDL19u'
        b'lyXzbO({2}+>~-t%1tRZrQDQqQ_4*#H>KQ^a#PApDL19ulyXzbO({2}+>~-t%1tRZrQB2>H`T{Y88>CzlyOtWO&K?3+*BVoMcfo|'
        b'L&OacH`K=s2{$C%kZ^<FKMWB!MBET@L&OacH$>bJaf4qz3>i0M+~C=&F8HwT@rU<{m>Xhlh`Ax=lz-UDKWyb6w(<{K`G<YaKfJ5Q'
        b';fl%!55A}yqHc(~A?k*x8=`KAx*_U@s2iegh`J%_hNv5&Ziu=e>X3kFB_LV}h*kokm4Ij^AX*8CRsy1xfM_KkS_z2O^&VOYh*kok'
        b'm4Ij^AX*8CRsy1xfM_KkS_z0&0-}|GXeA(835Zq#qLqMXB_LV}h*kokm4Ij^AX*8CRsy1xfM_KkS_z0&0-}|GXeA(835Zq#qLqMX'
        b'B_LV}h*kokm4Ikn?xF7ih{roiD*@3;K(rDNtpr5tdJnAtL@NN%3P7|15UtBSv@ZA1ia)gC53Tq^EB?@mKeXZxeUCr9p28#Jj*L4p'
        b'?#Q@PtS;n^kUK)|pp|PSAX*8CRsy1RwTD&!q7{H>1t3}hh*kii6@X|3AX)*4Rsf<EfM^9ES^<bw0HPIuXayiz0f<%rq7{H>1t3}h'
        b'h*kii6@X|3AX)*4Rsf<EfM^9ES^<bw0HPIuXayiz0f<%rq7{H>1t3}hh*kii6@X|3AX)*4Rsf<EfM^9ES^<bw0HPIuXaykp9)NhE'
        b'xJBF+aa+V~5w}I$7I9m|Z4svcL@NN%_W;CuLDiSuTegr>{Gk<pXvH7;9)Ea0mYCaOZnK%&xa4#z0MWYULo5H#%0IO753T$|EC0~S'
        b'KeX}>t^7kP|Io@mwDJ$F{6j1M(8@owF8R=kKeXZxt@uML{?Lj)wBiq~_(LoH(275_;t#F(Lo5E!ia)gC53Tq^EB?@mKeXZxt@uML'
        b'{?Lj)wBiq~_(LoH(275_;t#F(Lo5E!ia)gC53Tq^>yi(x^g}EC&`LkF(hsflLo5BzN<XyH53Nf+v@ZG3%09F%`Ou0!G@}oFk3M|0'
        b'qVh7XWL(L(l5y2~V=Ez7LQcttz9%2PT48;jvzDMML3hhOeDzqYpR6zIO4gOED_K{vu4G-wx{`Gz>q^#@tSebpvaV!Z$-0ttCF@Go'
        b'm8>gSm$EKpUCO$Ybt&so)}^dVS(madWnId;R5zFE=2Ft7q)SPck}f4(s+&temx3<U&851zR5zFE=2FO|kV|!QsctUS&837(36~Ns'
        b'C0t6llyE8GQo^N#O9__}E+t$_xRh|oQ`Qu5DdJMZrHD%r7b5PKeRyqe$heSkA>%^Eg^UXs7cwqn+%5X>-a{c5LN0_{2)Ph)A>=~H'
        b'g^&v&7eX$CTnM=kav|hG$c2y#As1}qLdu1d3n>>;E~H#YxsY-p<wDAZlnW^rQZA%iNV$-5p*}8zTnM=kav|hG$Sona)W<Cuw`AOs'
        b'aZAQ68MkEIl5tDME%k9r#4QoGMBEZ_OT;Y^w?y0$aZAK45w}F#5^+n!EfIH1KD=C7GH%JZrABUvxFzD2h+86ViMS==7QcR2GH%JZ'
        b'CF7QiTQY9RxFzFm$%prj2)QNXmXKRQZV9<1<ff3DLT(DVDdeV*n?i01xhdqPkefno3b`rdrjVOLZVI_6<ff3DLT(DVDdeV*n?i01'
        b'xhdqPkelk`ri`01ZpyeR<ED(8GH%MaDdVP$n`+~xh?^pAinuA_rihy&Zi=|6Hf~C|DdDDsn-XqHxGCX=gc}lWNVp;4hJ+guZb-Nx'
        b';f9185^hMiA>oFE8(VK{L&OacH$>bJaYMun5jRBK5OG7q4G}j)++ZI!I7c#s+z@g@$PFPkgxnBvBlxsSxgq6-lp9iRNVy^9hLjsp'
        b'Zb-Qy<%X0SQf^4OA?1dYL*}8Bc_?KbN|}dJ=Ao2%C}kc>nTJy5p_F+jWgbeIhf?ODlzAv+9!i;qQs$wQc_?KbN|}dJ=Ao2%C}kc>'
        b'nTJy5p_F+jWgbeIhf?ODlzAv+9!i;qQs$wQc_?KbN|}dJ=Ao2%C}kc>nTJy5p_F;3d*<Ombt&{v3O$rU52esUDfCdf>O(2>Q1{Hk'
        b'Ye$ZZJ2Fm}eJG_KN~woZ>Y<c+C|>rV6niMe9_k)@ct4hyJ7VsLxwCob<=pA!T~uDs9YJ>l-4S$0&>cZ{1l<vIN6;NXcLd!LbVtw~'
        b'L3ae*5p+k;9YJ>l-4S$0&>cZ{1RY`zrPxC$_E3sFlwuF1*h4AyP>Ma2Vh^R*Ln-!9ianHK52e^cDfUo`J(OY(rPxC$_E3sFlwuF1'
        b'*h4AyP>Ma2Vh^R*Ln-!9ianHK52e^c-D3|gU$=<cB5sSgE#kI_+o`v<E#tO~+cIv;xGm$hjN3A9%eY(Y;k}1KZVS0B<hGF8LT(GW'
        b'jV~TbSA8g@9!jZ)x~CrA3)*!l*HW&fTuZr@axLXr%C(ehDc4f2rCdw7mU1oSTFSMQYbkfjJUm}ME#+FuwUlcq*XrY1$hDAbA=g5#'
        b'g<PwTYZ=!vu4P=yxR!A(<66eGjBE9AE#g|lwTNpG*CMV(T&s_33D**?C0t9mmT)cMO2U<dD+yN;t|VMZxRP)s;Yz~kdJm=0L)}9U'
        b'?*&yCamAC>l5wR@t~&Wxiz_<xQm&+2Nx718N<5Sj52eII>1q$9t34F+4yC+9Deq9qJCyPcrMyEa?@-D+l=2RxyhADPP|7=$@(!iE'
        b'Ln-f2$~%<u4yC+9Deq9qJCyPcrMyEa?@-D+l=2RxyhADPQ1`sU^Cf!AJG>WET%BBsxfF9L=5Be1*H)>MODUI9F4f7UI=Pf_DdSSc'
        b'rHo4%m+ItF#HEN!5tkw^)ybuVO9__}E+t$_xRh`y;Znk-gi8sR5-ug2F7{COz{6K7DnIlIcnY~3tS{wK%7v5*DHl@imUwt=R*1O}'
        b'bD?%F_~k>$xsY=q=R(efoC~#c!FDbLT?o1mbRp<M(1oB2K^KB91YHQa5Og8vLePbv3qcoxE(Bc&x)5}?$iwS77jz-$LeSkJ5APb<'
        b'b3qq^F4W3}TDcH&p;j)WT&R@`As0e!3ArWYmXKR&<(7<FGH%JZrBl`=;+BY8B5sMeCE}KdTWaN&gj*7BNw_89mV{dpZb`T$;g*D3'
        b'5^hPjCE=EYTM};ZY;}paCE`{jA4}!c%q<zW_|?N&EU$J>7kntC9!jZ)QtF{}y@yihp%i*3g&s<whq{Lz-g_wLrktB{ZpyhS=cb&S'
        b'a&F4GDd(n~n{sZ-xhdzSoSSlP%DE}$rktB{ZpyhS=cb&Sa&F4GDd(n~n{sZ-xhdzSoSW+8rkI;zZmN@;Qf^AQDdna*xv5TW%DAac'
        b'Zi=`m;--k3B5sPfDdMJxn<8$ilbaH5NVp;4hJ+guZb-Nx;f9185^hMiA>oFE8xn48y{irpH$>bJaf5?$L&gmmH)Pz9aYM!p88=Gr'
        b'szb;PAvc8F5OPDv4Iwv#+z@g@$PFPkgxnBvL&yywH-y{}azn@sAvc8F5OPDv4Iwv#+z@g@$RYHQ3O%Gk52?^YD)f*FJ)}Ypc@I6@'
        b'|8P>Fhg9ex6?#a89#Wx)ROlfUdPs#HQlW=b=phw)NQE9!p@&rHAr*Q^g&tC&hg9ex6?#a89#Wx)ROlfUdPs#HQlW=b=phw)NQE9!'
        b'p@&rHAr*Q^g&tC&hg9exb)kn;<{_1NNM#;UnTJ&7A(eSZWgb%3dB}U@;hQHWN5mZwcSPI~aYw`<@{qdFLn`r*N<5@4^pFZXqyi7A'
        b'z(eXn52?IE>N*dpxI-%LkcvB`;tr{}Ln`i&iaVs@4ym|9D(;YqJEY<csklQb?vRQ*q~Z>#xI-%LkcvB`;tr{}Ln`i&iaVs@4ym|9'
        b'D(;YqJEY<csklQb?vRQ*q~Z>#xI-%LkcvB`;tr{}Ln`i&iaVs@4ym|9D(;YqJEY<csklQb?vRQ*q~Z>#xI-%LkcvB`;tr{}Ln`i&'
        b'y2?W;?T|`4<UQ^1dh6UGZi~1r;<kv}B5sSgE#kI_+ahj@I9=r-b(M!y+98#8NTnT8X@^wWA(eJWUE?7Yc1VRCQelTw*dY~mNQE6z'
        b'VTV-MA@5;_*E7{Zu7z9+xfXIQ<XXtJkZU2=Lav2e3%M3@E#z9rwUBEe*Fvs^Tno7taxLUqZCuN^mT@iPTE?}EYZ=!vu4P=yxR!CP'
        b'Hm*fni?|kXE#g|lwTNpG*CMV(T#L9CaV_Fn#I=Yk5mzFv)W(&BD+yN;t|VMZxRP)s;Yz}lgewVG60Rg%Nw|`5CE-fKm4qt^R}xNF'
        b'cSuDY@*Z_~Er_@haV6qP#FdCE5mzFvL|lou5^*KsO2n0jD-l;Bu0&jkxDs(C;!4Doh$|6SBCbSSiMSMTDdJMZrHD%rmm)4jT#C39'
        b'aVg?b#HEN!5tkw^MO=!w6mhqp!xNblaVg?b#HEN!HE}87Qo^N#O9__}E+t$_xRh`y;Znk-gi8sR5-ufNO1PA8DdAGWrG!ffml7@|'
        b'TuQi<a4F$Z!li@@2^SJBBwR?ikZ>X4Lc)cF3kerT9}0zt3lSG0E<{|2xDauHsgYF1A(e4RWgJo&hg8NPm2pUA98wvFRK_8daY$tx'
        b'QW=L-#vzq)NM#&S8HZHHA$3`YRK_8daY$txQkQi|WgJo&hg8NPm2pUA98wvFRK_8daY$txQW=L-#vzq)NM#&S8HZHHA(e4RWgJo&'
        b'hg8NPm2pUA98wvFRK_8daY$txQW=L-#vzq)NM#&S8Hc=Q99~*35w}F#;+g7_aZAQ68MkEIl5tDMEg83D+>&ui#x49Da4O`G3OS^%'
        b'>yWyxLn`8sx~@Yi;gCu=q!JFv>pG+&4ylMkD&mleIHa!YkV-hD5)P?^Ln`5rN;sqv4ylAgD&deyIHVE|sf0r+;gCu=q!JFPghMLf'
        b'kV-hD5)P?^Ln`5rN;sqv4ylAgD&deyIHVE|sf0r+;gCu=q!JFPghMLfkV-hD5)P?^Ln`5rN;sqv4ylAgD&deyIHVE|sf0r+;gCu='
        b'q!JFPghMLfkV-hD5)P?^Ln`5rN;sqv4ylAgD&deyIHVE|sf0r+;gCu=q!JFPghT4O4yk}cD&UX`IHWG?koSDU>#cK$xFO<h0f+Zv'
        b'RdYo<FXV=h8$xafxgq3+kQ+j72)QBThL9UVZV0&{<c5$NLT(7TA>@XT8$xafxgq3+kQ+j72)QBThLA(TA(U_kB^*Kthfu;HlyC?o'
        b'96||)P{JXUa0n$FLJ5aZ!XcD!2qhds35QU^A(U_kB^*Kthfu;HlyC?o96||)P{JXUa0n$FLJ5aZ!XcD!2qhds35QU^A(U_kB^*Kt'
        b'hfu;HlyC?o96||)P{JXUa0n$FLJ5aZ!XcD!h<n1}o7dm-Q1zkeL)Asx5phSv9T9g#+!1j{#N84OuVZp#+%4kp)m?^Gq}-8m2VvDv'
        b'$|00;2&EiCDTh$XA(V0mr5r*jhfvBP?kR`&uFJV2=Z>5^a_-2vBj=8sJ96&Gxg+O}oI7&v$hjluj+{Gk?#Q_#=Z>5^a_-2vBj*rv'
        b'2*n&iF^5phArx~6#T-I0hfvHR6mtm096~XNP|P6|a|p#8LNSL>%pnwW2*n&iF^5phArx~6#T-I0hfvHR6mtm096~XNP|P6|a|m73'
        b'A(V0mr5u76bqK{ALNSL>%pnwW2*n&iF^9Ov9Nyb2<F<_3GH%PbE#tO~+fzNo7IIt2Z6UXX+!k_M$Za9Fg<K1{7IH1*TFAAKYa!P{'
        b'u7z9+xfXIQ<XXtJkZU2=Lav2e3%M3@E#z9rwUBEe*Fvs^Tno7taxLU;F^A{hV9U6caV_Io#<h%V8P{s#TEw-8YZ2EXu0>pnxK<n2'
        b'60Rj&OSqPBE#X?iwS;R4R}!uyTuHc+a3$eN!WGX}OT?9kD-l;Bu0&jkxDs(C;!4Doh|@J4LP3X6&><9b2n8L2L5EP#Ary271s&oZ'
        b'ba)rh$!bZtl5!>GO3Ia#D=Ak}uB2Qo?z)&OF;`-)#9WEF5_2WyO3am*D=}AMuEbo4xfF9L=2Fbvq7Kg|D#cuixl|*EYdVCY4xy++'
        b'DC!W3I)tJQp{PSB>JW-LgrW|iYdVCY4xy++DC!W3I)tJQp{PSB>JW-LgrW|is6!~~5Q;j4q7I>`Ln!JHiaLa%4xy++DC!W3I)tJQ'
        b'p{PSB>JW-LgrW|is6!~~5Q;j4q7I>`Ln!JHiaLa%4xy++DC!W3I)tJQp{PSB>JW-LgrW|?s6!~~5Q;j4q7I>`Ln!JHiaG?N4xy++'
        b'DC!W3I)tJQp{PSB>JW-LgrW|is6!~~5Q;j4q7I>`Ln!JHiaLa%4xy++DC!W3I)tJQp{PSB>JW-LgrW|is6!~~5Q;j4q7I>`Ln!JH'
        b'iaLa%4xy++DC!W3I)tJQp{PSB>JW-LgrW|is6!~~5Q;j4q7I>`Ln!JHiaLa%4xy++DC!W3I)tJQq02dhk`AGyL+El2p`b(DgAOmM'
        b'mxx;;Zt+WpCF68KhfvlblywMY9YWW02t^%2QHM~}A$U25P}U)obqHl0LRp7U)**B?hfvfZ6m<wi9YRruP}Ct5bqGZrLQ#iM)FBjg'
        b'2t^%2QHM~}Ary58MIAy>hfvfZ6m^Ju)ZzJpn^JB{xhdtQl$%m+O1UZJrj(mfZc4c+<)-?$DdeV*o9g4HjGOA?rihy&Zi=`m;--k3'
        b'B5sPfsXlH>xGCYLgqsp>O1LTEri7akZc4Z*;iiO}5^hMiA>oFE8xn3vxFO+&gc}lWNVp;4hJ+guZm5qN5^ij@6GOxeo~sUi>o9t8'
        b'MSCyhhLjup+F^*f!DepAxgqC<oEvg($hjfshMXI6ZpgVI=Z2gc{Mum%x*_O>pc{g22)ZHYhM*gQZV0*|=!T#hf(~(qrMSaV++iv1'
        b'uoQP#iaRXD9o9YW@Lo{yqvA)!w~B8S-zvUUe5-h?c&m7;c&m7;c&T`)c&T`)c&T`*c&d1+c&d1+c&K=&c&K=&c&PYN@ulKR#g~dN'
        b'6`v|TReY-WRPm|eL&b-R4;3FOF5!-ZI}+|lxFg|?ggX-MNVp^6j)XfB?nt;J;f{nm67ER2BjJvOI}+|lxFg|?ggX-MNVp^6j)XfB'
        b'?nt;J;f{nm67ER2BjJvOI}+}EzJ~Nt&|xX)uoQGy3OXzW9hQO)OF@UFpu<woVJYaa6m(b$I;?xp;dKxt+>vlw!fgq+CES*9Tf%J#'
        b'w<X+`a9hG{3AZKOmT+6bZ3(v}+?H@#!fgq+CES*9Tf%J#w<X+`a9hG{3AZKOmT+6bZ3(v}+?H@#!fpJz($e)DmU0eDIftd3!&1&+'
        b'Dd({6IfwW5%D64#wv5{{Zp*kW<F>lEE#tO~YZ=!vu4P=yxR!A(<66eGjB6R!GOlG@%ea<tE#q3owTx>S*D~&wb9laXTgJ7FYZ=!v'
        b'u4P=yxR!A(<66eGjB6R!GOlG@%eYn-*CMV(T#L9CaV_Fn#I?G(mT)cMTEex2YYEp9t|eScxR!7w;Yz}lgewVG60Rg%Nw|`5CE-fK'
        b'm4qt^R}!uyT;VH+b<a6`wW9JeuGGhsjH_aGAy-1Kgj@-^5^^QvO30OvD<M}xu7q3(xe{_E<VwhukSifqLau~d3Aqw-CFDxTm5?hT'
        b'S3<6YTnV`naw+6e$fb}=A(uifg<J}`6mlu#Qplx{OCgs+E`?mGjk~2Bp2(z#OA(i9<5I+>+PIW(DdAGWrG!ffml7@|TuQi<a4F$Z'
        b'!li^u36~NsC0t6llyE8GQo^N#O9__}E+t$_xRh`y;Znk-gi8q*5-ucM;KGATF^8p?!&1y)Ddw;gb6AQwtb5GiwIJd`#D$0p5f>sZ'
        b'bfy|2E<{|YiwhAKA}&N+h`11OA>u;Bg@_9g7a}f1T!^?3aUtSD#D$0p5f>sZL|llt5OKGV!|ShB#D$0p5f>sZL|llt5OE>mLc}c*'
        b'w?y1h6SpMXl5k7HEeW?I+>&rh!Yv86B;1m4OTsM)w<O$>a7)513AZHNl5k7HEeW?IoI(yuA%~@q!_ws(mNE`YmvdP6h{JocWZaT*'
        b'OU5l3w`APHVBFI69M(PK@YP*ZUX9$6a!blBDYvBDl5$JREh)F8+>&xr%1tRZrQDQqQ_4*#H>KQ^a#PApDL19ulyXzbO({2}+>~-t'
        b'%1tRZrQDQqQ_4*#H>KQ^a#MZW6mnC@O(8dh+!S(CecY6BQ^rjhH)Y(EaZ`QV6me61+>~%r!c7S`CES#7Q^HLNHznMZa8trf2{$C%'
        b'kZ?o74GA|S+>mfX!VL*GB;1g2L&6OSHzeGUa6`fk2{$C1A`VN}b683^EMCuHDdMnnIftc$!_ws(mI4k-0f%)DIJ~!4$PFPkgxnBv'
        b'L&yywH-y{}azn@sAvc8F5OPDvjj1kjNVy^9hLjspZb-Qy<%X0)!eJ`mFqLqaN;pg<9HtTuQwfKugu_(AVJhJ;m2j9!I7}rRrV<WQ'
        b'35ThK!&JgyD&a7daF|LsOeGwq5)M-dhpB|aRKj5@;V_kOm`XTIB^;&_4pRw-sf5E+!eJ`mFqLqaN;pg<9HtTuQwfKugu_(AVcruC'
        b'56Pw?4pR|_sffc=#9=DpFcoo__lU!LdxxsaxFh3^j5{*!$hafpj*L4p?#Q?!<8(cTsgT3GhaA4TOL9f^#oQ5dN6Z~Dcf{Ngb4Sb_'
        b'F?Yn=5pze(9Wi&r+!1p}%pEaz#M}{cN6Z~Dcf{Ngb4Sb_F?WkOyk3c74l#$Rn8Q@eVJhY@6?2%1IZVYIrmp8O6?2%1IZVYIreY3L'
        b'F^8#`!&J;+D&{a1bC`-bOvN0gVh&R=hpCvuRLo&2<}ej=n2I?}#T=$$4pT9QshGo5%wa0#Fcou{iaAWh9HwFpQ!$6Bn8Uos9A0lZ'
        b'Tf}V<w?*6*amqPN<s7DR4pTXYdCxh#i!7Fxa$Cx6DYvEEmU3IlZ7H{<+?H}%%55pvQm&<3OSzVEE#+=Ohu1UJVy?wpi@6qaE#_Lx'
        b'wU}!$*J7^4T#LCDb1mjt%(a+nG1p?Q#axTI7IQ7;TFkW?xt4M*<yy+MlxsC|E#z9rwUBEya;@{#7I7`&TEw-8YZ2EXu0>pnxE66O'
        b';#$O&h$|6SBCbSSiMUcDR}$`)ba<gyBCbSSiMSGRCE`lNDe5p8b(o4eOhp~0q7GA0hp9_COeGyAFX=E9b(r_4!}~5`uEbo4xneh0'
        b'a<1fD$+=tD;ax<~m7ps@SAwntT?x7pbS3CY(3PMoL05vV1YHTb5_Bc#O3<aCOF@@{E(Ki*x)gLN=u)j*%DI$tDd$qorJPGSmulrw'
        b'tz1gETiW6Ir<PJKrCdt6lya$7E`?mGl}j0yGA?CYs+CI-mm)4jT&k5z36~NsC0t6llyE8GQo^N#O9__}E+t$_xEy*@PZ5_QE=635'
        b'xLe%eRsEKC_-ZBfV2gKAeJK}*<;7eamX~uO=R(efoC`S@axUat$hlBA7jiD-T*$eQb0OzK&V`%{ITvy+<Xp(PkaHpDLe7Pp3pp2Z'
        b'F63OuxsY=q=R(efoC`S@axUat$hlA_7h*2NT!^_4b4#7vQYW{B+)^jEWZaT*OP$;jaZ8=tl5k7HEeW?I+>&rh!Yv86B;1m4OTsM)'
        b'w<O$>a7)513AZHNl5k7HEeW?I+>&rh!rcN7uXo5L;ugPqSTb(OI7J?&A`erSb(l&#OeG$s5)V^}hpEKFRN`SO@i3Kmm`XfMB_5^{'
        b'4^xSUsl>xn;$bTBFqL?iN<2&@9;OlxQ;CPE#KTnLVJh)3m3Wv+JWM4XrV<ZRiHCVlJUsslrjVOLZVI_6<ff3DLT(DVDdeV*n?i01'
        b'xhdqPkeh1bri`0v<EDt4B5taUn-XqHxGCYLgqsp>O1LTEri7akZc4Z*;iiO}5^hMiA>oFE8xn3vxFO*NPgRGA8zOFqxFO<(h#MS}'
        b'8!~RlxWR88hL9WT<c5$NLT(7TA>@XT8$xafxm)Dnz3WnLNVy^9hLjspZb-Qy<pw*sA?Ajd8)9yVxgq9;m>Xhlh`Ax=hL{^-Ziu-d'
        b'=7yLXVh)jq>raV*_4C4dFSHg?3rh<F3rEF|iXRm}Dt=UatN2#&t>Rn7w~Dulw~Dulw~Dulmx`B)mx`B)mx`x~=a=FSGp;{<jqvd<'
        b's-CJIsvg?)Q1MXlQ1PYWOU0LpFBM-yM{%n9<bj{b71f{G`JwWI<%jlusQSR(pSyGKmCpzN2)c8J?)_K;cY$<A(j7^6?$W)gA64Jn'
        b'`Q2PmePDO)*u5UBS--XWoxBV8zqtX$4JaOf;uaLQptuFeEl3@L<OZakg48P@^!f``4?X{(;-TV8#g~fT+<Wf%&Ar#cz`~8a*FtOI'
        b'OYwXzEG-Ny+|WDEh1No9VQJw@@p(Il`%d0#p>GT8zL2+tye}**+>7^Z;k*v}mxcRbzZXCE=lovd+@JGDjW5OH@lZGKUJHF&xIffQ'
        b'yVt_HEu8zp{W-taxM7!%{kU0|j~ab{!QE^0(!!VG^*FjW=w1trhbk4Xh0U#jLz4HyelMPlS4n<;wQmWx@4`J$+;`&MEB*1$bKi}7'
        b'ZdWY4fMVdF+>GnTJ#NPB=RNNGet(_ZjJqEDelzZR)VLY<yvMvV^SDReHuwG<-i)iq=01$O-(wsclA)K}&A4Zcn{oYI7|++qzb^FM'
        b'xcfouyK&cZA>S&_e_gm6_w)bxvM_Lub3Q&FeK+p8xi{nXb76ga`Sjhme%xc<|I*KWp>>bX&w1aC+mGTm<M!hoeXxVAJ?}(~Uf8OK'
        b'arcFXardY8W?VgrS88GDp~lCTXy1){ZdJDKk*JZMf$h6-@u(4>-`Dys+<H7z<i~g0z6&>>_ZT1F>iaI-YoWD}T3A{bA3Kx23pXFn'
        b';QVZR--R2`d)&JB*IjP!Ezj~d<Gy)r@%`-ec_7ETegCrZIDdH58J}0`b^e<7mB$%O-&by1aDPLt#~Exq>x}!#z0OU$Z{kts#@#pf'
        b'7i`_V`z9ZC&f{5lJleRg+#l_J8e5P3xu3@NbLF-L`?<2XGVYId-|BvB!F{W5@V)l5Q^)#wCi{4_p_Mv%w0dm8{h7O;$=2hD*5epF'
        b'9__XT=W(BW{4IQWKdi?mc=xx^dYsDs@(sjoW$`|I9Xva|`&Qo^{N~<J)*ZsfUwz#reDsZ_6WY=VZRv!z?iBv|75kUnT+ah=Um5qO'
        b'?f!&L9)QUMFnIv*SzhV0yn1lhkEiC&;q#?7Qh)GciH`w{^1<PqwtR88?ko8j@qBPNpO1#OgnV%Le6+(YIQbdzd~o>OpEExH!RLd+'
        b'JIMLqFdubxe*UA%2Zyhf@%b$-9~|CaZTaHxv4T&vQeR}GzQ{;@k&*mVFZE4E@<YAUR~f0VGEyJurM}5Xo$n^kck`iOJ)V|mJ(ZE4'
        b'|J3rK;PYh`LoeEO^uk!JKaMAa_d1JzgllMJ9<H?NOfoRn=Z-BOo#FEs3_>(oc#YH_?pU9@h<s3ZKi2V}@cu^_4+`_Sf<MCGcN_7b'
        b'@VSC7H{vDX<1VSYus*)s2z`+o`XV><-A3rUjnEgl@gOiB&rA2`zo>W-nD-UD0^`vU&c}V~aHT*0b;P41oR4j*#rq7c<l#!IPI3#L'
        b'9pU|mer4gGSK{c12L1};VPQR2*5|W{cv#qvtzJiuws2+;4-40Gg?|O>^XYj!zvs9=*86*oFAJYP{fxVV&!=Xp_&B)7pa3KFG!OJb'
        b'@#+Wr_LH8^e(>B^@MpMQ2tHQUXHwT4!F*nd`+Pi@OW$`aec!R3-*>F{RlPse*q^^<TCYa%v4X$9^=Jgw*H_&Ac)<E_1)acpDEQp!'
        b'{`@81x+6HhzFy>~BU7JW{Oc~@>mHv!n_74O=GWJ8`})0MJ@k9-Ve;rd^~?8Vq(0s!4p;d6<LljZ$M1Cy?f?4x$G@H*46gU}Gd@0)'
        b'ifgQ8JzBu$L7sT~Tdx*yy?^@s^9m3D;1)=~rO}pk$M1EI&+^PWeXoVjUvN(iw#<inug4mn_i?_?gTMMM$`40uKF)(*D~pfDEx4Zt'
        b'Ket-^O?7}%9pF?4I62%hANIXgQajU&D?^8-_!JbMf@cT#)sJ(0Ug1-ad<v4sAd~mu%P_eG!DA536&{1&7R1p0EWQr#so>Oj-_&^D'
        b'ygT?Iuk`WAO#Tr}?*HVgaPk=#+=9U^7<>)<>R^02n!iFApXLRB-GKV}-{ZZ~Usk>z6#aacrMnE=<=pGPhS_+R<jSr7H~H1T&!?6B'
        b'H~`O;=E{8wo--JWD+4S2*n;Oub0xX5xS}m6Zb5Miid#^87K+b8@>xhe3(40(KDOYwq8DQD4<Yo25cdsuE-WpK&kIX`21~`4iZ2yk'
        b'Dn7aEQ`IMr{p7Kq{{{lTF8o4TK3AG6`KOimdS(3^E46yfUmoo|UQ54RVB*@j|8$qzLTI5L3%4R$3zrst$?X2T^7V1~<yTp-P+Hhp'
        b'xV|jhp2&5t{kT1F`>pcr@z=fZ<Mz*W<6d{&k9Xj|FZ}l#|NZsxzfbIMqHf;nK5pN*{rda_l%C|$lU#a|OHXpCc&m7;ct48gePQb!'
        b'*XMg|72hhpwdY&KxAy$t`97+B@Pr>c;Ri493xzMd(5^q#&986Yxc&Ng`!9*PUl!iY?k^ca++Kga{VB%SxWo1N_BruDGqj-6b8F$k'
        b'!cy_2;!DMsiZ2xp6%Q2;6%Q2;6;C|hiRU{Dx8vzfJl%!5g}U{n?(?k-bqjSHbsKdXbsKdXbvNp6)ZIAF8+AA89@IU0Z5<RodTAY0'
        b'zVx!bu*(-JU)bgAW8nqD76@B+!k)Kp+>W{vbtmdh)Sak1QFo#4LfwVB3w0Oj2I>at2I>at2I}VL7uLF)^;#$`Y%N?^C>1a4b*Xl#'
        b'cByu&c5BBQhrCt0am+WK>#fQ+zwYhax&!uFxUg`v+egI@cKc@7^S}R6`ce5t*z@+l?e+QbZ-V9f?B?t33<wqwEYyU>ov_!!g@uXI'
        b'6Q$?JU5-0qj}zi~eZGB7+#a~SaQpteJvYRjCydzN!2W(lEPz-5vA84l*q)nV&)c!TiQ}9&&V}P#IL?LVw{V;b$GNe;jpN)n&W+>T'
        b'sJl^j<2Y~Z@5cUa)IF$sQ1@Ve5BB%qb%m}g&~*j6u0YonFe_nJ-pzV^lJdFhN_1U`t}6jn0<1*Wl{#rmoiwIS8dE2YiOwr^&X{Pu'
        b'Qm2dwwo+$|sWZmZ8DpybO0{39_AAwXrP{Al`;~XUUJEK-DqbqyD&8vID&8vID!x^GtN2#&t>Rn7kBT1^KX|Dh{COX|(l7jZUwENk'
        b'sC#Ljuh08jchFu7p@omN3#1iDE09(otw37gM6z%qSvZj_oJbZpE1XCcG%NIA1<VRPSfK|iQC6a?L|KWl5@jXIN|co-D^XUWtVCIf'
        b'vJz#5hOE$#70x6JXOe{v!U`XR6+Q?nd=OUnAgu0!J#R<dgSrQGFVwwI_d?xE&++GmtmRkZEV6YL**c4Cokg}j%x`^|-}*4W)t9yU'
        b'vQ}T#>dRUm=C>NN24xM(8k99CYf#oWoosxJ-}o56@iBhmWBkU)_>GV88=r>tov`Qas9UJpsN1O9sN1O9sJl^jqwdx#YoqW+;e)~l'
        b'g%5W4pzcB43w1Bly-@c;-OqY$oK9}wwb6TR^j;gi*GBKP(R*$5UK_pFM(?%Jdu{Yy8@<;?@3jHe23T9)J#L7#bt1Vz)&^Nyz1LRn'
        b'wbgrV^<G=O*M?ggZf&@=)p%_+UK?_4okDKVwL#YgT^n?5G+rBWZOFAD*M?mCF4uF14+<X?zI3QBRKBpsmtJ3=@jAfk0IvhQ4)8j_'
        b'>j19<ybka>!0Q071H2CKI>74yuLHae@H*(W4(vMUwhrhzpzEO9I_S0zx~+q5>!8~@Q0qXg1GNsit%Gjsz^sFA>wv5SvJS{PAnSmv'
        b'1F{atIw0$Stb=CjK&%6?4w|h4unxdFXtoZ#y5Q9XuP%6X!K({iUGVCHR~Njx;ME1ME_ik61n~k`m(C9_h;`|M`wL`UAnO8I7s$Fm'
        b')&;UIkadBq3uIj&>(Yl@7tFd~)}>DC0$LZ)x`5UNv@W1^0j&#YT|nysS{Kl|fYt@HE}(S*tqW*fK<ffp7tp$Z)&;b5-348!L05Lr'
        b'r66?e2wiMKSEtbBEOh+~T`)se*w7_8bPW$(^g~w(d0bcIbvx<?>IUj2>L%(Y>L%(Y>K5u2>K5u2>K5uY>Ne^&>Ne^&>TcBCsJl^j'
        b'qwYrCgSrQG59%J&J*a!3?uEM7$LlHS;#0a>l`e0k>tX2vTDszvF43iHdOa=z_8NzWShwqpz3+g^AZvoG39=^0njmX}tO>Fv$eJK)'
        b'f~*O$Cdir~Yl5r^vL?uyAZvoG39=^0njmX}tO>Fv$eJK)f~*O$Cdir~Yl5r^vL?uyAZvoG39=^0njq`*^*wb#P+dV(mmJkKN_A0F'
        b'T?JK_P1SW)bzxRr$yJwr)wN=EaamoBR+q1RT=4dFJL<x#1+NyoTJUPYs|BwXyjt*T!K($Y7Q9;UYQd`ouNJ&o@M^)U1+NyoTJUPY'
        b's|BwXyjt*T!K($Y7Q9;UYQd`ouNJ&o@M^)U1+Nyo3V0RpD&SSXtAJMluL52Lyb5>~@G9U{z^i~)0j~mH1-uG)74RzHRluu&R{^gA'
        b'UIn}gcopy};8nn@fL8&p0$v5YijNnV{^fSmg;(*#t5=K^b>UUOtAJMluL52Lyb5>~@G9U{z^i~)0j~mHCA>;_mGCOzRl=+M;?+C#'
        b'i^2da0agO61Xu~Ma^c)NAyz`Hgjfl&5@KcG6gWXvf~*8t39=Go<>$*3{}RYX;aa)_DwlQ{YGv!y1-24xR)VbrTZuL+!B&E;1X~HV'
        b'5^N>dO0bn+E5TNRtpr;Mwi0Y5*ebA9V5`7Zfvo~tg*L08Rza<TS_QQVY8BKfv{?nT3TPG3Dxg(BtI%c@%qo~wFsooz!K^}?RUoU-'
        b'W);LLv{?nP3T;-wt7^O}6~HQhRRF6tj4p^(5UU_oL9BvU1+fZZ)xo<`fvf^q1+ofc)rG<hvl?dgNIg*F4xbNo1FhDl?+vvYYW37E'
        b'gRKTz4YnF=HP~vf)nKc^R)ehuTMf1vY&F<wu+?Cz!B&H<23rlb8f-P#YOvL4vl?nO)M}{JXtNq<HPC9H)j+F(Rs*dDT8%cVVOGPe'
        b'hFSf^tmoe;+N=$-Hptp&vo^%q5NkuMjW%lotPQX>+N=$)HoV&KYQw7yuQt5e@M^=W4X-x5+VE<_s|~L<yxQ<;!>g?`!ws;u&I~ui'
        b'+7N3)tgS|CgRBj*Hpto_YlEx}vNp)tAZvrH4YD@K+8}F#tPQd@$l7SLHpn_4>wv5SvJS{PAnSmv1F{atIw0$StOK$R$T}eFfUE<u'
        b'4#+wn>wv5SvJS{PAnSmv1F{atIw0$Stb-QoK&*on>j10+unxdFXt55wI`Hbis{^kNygKmez^enV4!k<>>cFc5uMWIA@an*;1FsIe'
        b'y5Q9XuP%6X!K({iUGVCHR~NjxG?8@ytP5aW0P6x+7r?p}PDU??bwR8PVqFmHf>>AL!{`fST_EcMS(iS5zhKq{vo4r*!K@2rT`=o{'
        b'Sr^Q@VAch*E|_(}tP5scTd!%LbpfplXk9?-0$LZ)x`5UNv@W1^0j=vJTK{_Wp^vvO+@83-aXacx)Sak1QFo&57qcF}lK6S>ePQr@'
        b'Veoxn@O@$MePQr@Veoxn@O@$MePQr@Veoxn@O@$MePQr@Veoxn@O@$MePQr@Veoxn@O@$MePQr@Veoxn@O@$MePQr@Veoxn@O@$M'
        b'ePQr@-3Q;FKTrC3@_k|QePQx_-6!8)1upFIg~AsKUnmT*Cd9f;zCSl~f~?!<`}+>446`u%zA*c~?z8W&+i|QX)Vj^Szwdy`OZyDB'
        b'Cfu5EYr?Gww{Fw#uMNtf1yqJyxB2(?17Vjz*92V?bWPATLDvLb6Ld|`H9^+|T@!Ro&^1BV1YHw!eI9^c7=T|GfL|DZUl@R27=T|G'
        b'fL|DZUl@R27=T|GfL|DZUl@R27=T|GfL|DZUl@R27=T|GfL|DZUl@R27=T|GfL|DZUl@R27=T~*0r)2b3uY~twa{xV^jfzG_*Y?&'
        b'wa{#>1eObAEs(W9)&g0#8TeOWm~|V1f8PO>fz|?A3uvva!cc4N)*Vn;9oK?e3vMmAwcyr*TMKS2xV7Ncf?EN%0&WG|3b++;E8teZ'
        b't$<qrw*qbj+zPlAa4X<ez^#B=fmSQPR)DPlTLHELjaERdfLeh*D}Yu2tpHj9v;uurz^p)@6(B1>R)DMkS%E$)AXcEy3V;>pvjScP'
        b'yb5>~@G9U{z^i~)0k7i1n^FR-1Xu~M5@033N`RF*KTL>~5Gx^8Lac;X39(XVhY7M0WF^Q-kd+`SK~{pS1X&5P5@aRFN|2QxD?wI*'
        b'tOQvJvJzw^$V!lvAS*#uf~*8t39=GoCCEyUl^`oYR)VYqSqZWdWF^Q-v{(hP3St$+Du`7Os~}dP#VUYR0IL920jvU81+WS&R>7-+'
        b'R|T&MUKPA5cvbMK;8nq^f>#Bv3SJewDtJ}!s^C?@tAbYruL@ojyefEA@T%Zd!K;E-1+NNT6}&2VRq(3d)olX)ec3k@23Q5K8elcR'
        b'YJk-Ms{vL6tOi&OuzI4?X?+CWAge)Ee@+`NOu#Qpz%NX|FHFEMOu#Qpz%NX|FHFEMOu#Qpz%NX|FHFEMOu#Qpz%NX|FHFEMOu#Qp'
        b'z%NX|FHFEMOu#Qpz%NX|FHFEMOu#Qpz%NX|FHFEMOu#Qpz%NX|FHFEMOu#Qpz%NX|ulof2>-NueH^|x`YlEx}vNp)tAZvrH4YD?R'
        b'tc@OP1FQ|OHhQcLuQt5e@M^=W4X-x5+VE<_s|~Mi6Y#IAiUO<+ur|Qj0BZxR4Y0O8eBV8^fXX0igRBj*Hpto_YlEx}vNp)tAZvrH'
        b'+XVdU0Ku#cvo_4yFl)oC4YM}PIxy?NtOK(S%sMdZz^nta4$L|*>%go7vkuHUFzYq~|GWk-%sMdZz^nta4$L|*>%go7vkrQ!1F{at'
        b'x{biUu4#<AAnSmvgC6Uk$2tJ(0IUPB4!}A9>j10+unxdF0P6s(1F#NytOKtOygKmez^enV4!k<>>Vj7nyt?4k1+Ol6b-}9(US06&'
        b'f>#&3y5Q9XuP%6Xp~t%5)olR&`NnnutP5aW0P6x+7r?pz)&;OGfOP?^3q95aur7dg0jvvPT>$F>SQo&$0M-StE`W6btP5aW0P6x+'
        b'7r?pz)&;OGfOP?^3t(LU>jGF8!1_G?9vFWQjK2rQ-vi_Cf${gi_<LadJuv<r7=I7^?t5VTJuv<r7=I6pzX!(O1LN<3@%O;^d)&w0'
        b'9~U0~JpLXSe-Dhm2gct6<L`m-_rUmjVEjEW{vH^A4~)MD#@_?u?}736!1#M${5>%K9vFWQjK2rQ-vi_Cf!}=(Ouq-F-viU{f!}=('
        b'48I44-vh(%f#LVS@O#{c-@keM0Zf230oDXq6JSk%H38NHSQB7j^gS^89{Ab!xKF;n9|)B})&yA-WKEFudGbB*yYGSDeGkmN2j<=b'
        b'zxy5-dk>7g2gcq5WAA~n_rTbDVC+3G_8u5}4~)GB#@+*C?}4%Rz}S0W>^(5{9vFKMjJ*fO-UDOrfwA|%*n42?Juvnj7<&(ly$8nL'
        b'17q)jvG>5(dtmH6F!ml8dk>7g2gcq5WAA~n_rTbDVC+3G_8u5}4~)GBe)l~v^&Xgd4@|uWrrra;`yLp24-CBre)l~v^B$Oa56rv='
        b'X5It8`yLp14~@JBM&1J>?}3r`z{q>xhu`Bq@&4K%h;^HIf87COEs(W9)&f}zWZfp-UmFCo7R*{OYr(95Spl;GW(CX&m=!Q9U{=7a'
        b'fLQ^v0%irw3YZlzD_~Z@tbkbovjS!X%nFzlFe_kIz^s5-0kZ;T1<VSV6)-DcR-nfUkQE>+KvsaP09k<^D<D>&#|nTI04o4i0IUF5'
        b'fgUU1Rluu&R{^gQUM0Lrc$M%f;Z?$`gjWf#5?&>|x(&QP-xv~LrOpZyVkN{%h?Nj4Ayz`Hgjfl&5@Ka)he1|?tOQvJvJzw^$V!lv'
        b'AS*#uf~*8t39=GoCCEyUl^`oYR)VYqSqZWdWF^Q-kd+`SK~{pS1X%^L3S<??Dv(tmt3XzPtO8jDvI=AsTC9Rtg%+y-RspO6SOu^O'
        b'U=_eBfK>pi09K*JDtJ}!s^C?@tAbYruL@ojyefEA@T%Zd!K;E-1+NNT6}&2VRq(3dRl%!*R|T&MUKPA5cvbMK;8p9B_Z~|NiAQSO'
        b'j??c3Sq-upWHrcYkkufoK~{sT{v0a|jJyX%-UB1=fsyyX$a`SqJuvbf7<mtjyaz_!10(N&k@vvJdtl@}F!CPvk@x5Ehgl7?8fG=j'
        b'YM9k9t6^5dtcF<)vl?bK%({)dKaU&CYM9k9t6^5dtcF>)k@v@TZuD3iJ=TU;8)9wrSQ}t%fVBbE23Q+lZGg1_)&^J`J=TU-8(wXA'
        b'wc*u<R~ue!c(vixhF2S2ZFsfe)rMCaUTt`_(PM3Rwc*u<SGRfh=Nr-nSQ}t%fVBbE23Q+lZGd$fcz^8xx~vVc_Ce3FLDmLY8)R*e'
        b'wb5oBwOI#d9hh}s)`3|EW*wMyVAg?I2WB0Zbzs(kSqEkvm~~**fmsJ;9hh}s)`3|EW*wMyVAg?I2WB0Zbzs(kSqEkvm~~**fmsJV'
        b')&W@uWF3%oK-K|S2V@<Pb<krSh;<;=L63C+)&W=tU>$&U0M>;b>w;Goyt?4k1+Ol6b-}9(US06&f>#&3y5QBNPu?$pbpfmkU|j&~'
        b'0$3Nox&YP%ur7dg0jvvPT>$F>SQo&$0M-StE`W6btV{C^7sR?C)&;RHh;>1%3u0Xm>w;Jp#JV8X1+gxObwR8PVqFmHf>;;C`aJWV'
        b'n0ZgkyeDSf6Ep9LnfJuZdt&B2G4r07c~8u|CuZIgGw+F+_r%P5V&*+D^PZS_Pt3e0e)2sr^PZS_Pt3e0X5JGs?}?fB#LRnQ<~=d<'
        b'o|t)0%)BRN-V-zLiJAAr%zI+yJu&m1n0ZgkyeDSf6Ep9LnfJuZdt&B2G4r07c~8u|CuZIgGw+F+_r%P5;s@UoBkzfk_r%D1V&pyV'
        b'Bk%V&yu{3VV&*+D^PZY{Pt3e0X5JIO_MRAdPmH`LM&1)6?}?H3#K?PM<UKL+o)~#gjJzjC-V-D5iIMlj$a`YsJu&j07<o^OyeCH9'
        b'6C>|=A9;ViejwI_SQBDRh&3VBgjf?|O^7ui)`VCSVoiuOA=ZRg6Jkw>wIJ4lSPNn;h_xWrLWi{g)&f`yU@df53tlaFwcyo)R|{S('
        b'c(vfwf>#S(EqJxy)q+<GUailUrbx`ZCuZIgGw+F+_r%P5V&*+D^Pcya_t)*Hix%rP^8Pvy5NkoK1+f;yS`ceNtOc<a#99z*K`czX'
        b'CnnzWKJos(11iI;fLQ^v0%irw3YZlzD_~Z@tbkbovjS!X%nFzlFe_kIz^s5-0kZ;T1<VSV6)-DcR-nfUkQE>+KvsaP09gUD0%Qfq'
        b'3Xl~bE6`&F#0rQN5Gx>7K&*gR0kHyN1;h%773i@7U<G=tfL8&p5?&>|N_ds<D&bYatAtkxuM%D*yh?bLpDzKH_>uR-#Cu}mJu&f~'
        b'_lfscVThFwD<M`wtb|wzu@Yh>#JWwqzjh2{CCEyUm4$96VOF*dHPA|+l|U<jRsyXAS_!lgXeH1}pp`%?fmQ;o1X>BS5@;pRN}yFh'
        b'tAJJktpZvFv<hey&?=x+K&yaO0j)xlRWPeyR>7=-S%oI6KvsdQ+tB;t@3BBufvf^q1+ofc6`HJqSOu{PVilUK0$7D6tKe0^tAbYr'
        b'uL@ojyefEA@T%Zd9ROVb>o)cNz5^;lEd0`YV(vXL_nw%0PyEt*;+NhNQ}2nX_teyTV(L9H^`4k|PfWchrrr}%?}@4R#MFD@cit02'
        b'?}?%J#L#<U=shv?o)~&h4813Y-V;ObiJ|wz(0gL&Ju&p27<x|(y(fO>Ju&p27<x|(y(fm=6GQKbq4&hldt&H4G4!7Iq4($C5zK1z'
        b'SPiloJyt`khFFars{z(+=>5500BZxR4X`%A+5l^#$J+2}!>bLiHoV&KYQw7yuQt5e@M^=W4X-x5+VE<_s|~L<yxQ<;!>bLiHoV&K'
        b'YQwAB(EIa^Z3C<gu<$eQc^`X!FJPBJ)&^M{WNnbOLDmLYd*i)-!>kRnHq6>;vo_G$Kx+f74YW48tPQjd&^kct0IdVG4$wM4>j13-'
        b'v<}cZK<fal1GEm%Iza0Htpl_U&^kct0IdVG4$wM4>j13-v<}cZK<falgC^_1tOK(SnyiB+>p-jnu@1yK5bHp!1F;UoIyf;r0P6s('
        b'^D$YNn0il4y(gyL6I1VrsrSUxdt&N6G4-C9dQVKfC#K#LQ}217dVhj`0j%5D`}+>446!bVbwR8PVqFmHHunD7P>^+jtP5maAnP{w'
        b'{wfT!E|_(t4mHraQk9|B1+^}ybwRBQYF$w4f?5~Ux}eqtwJxZ2L9GjFT~O<SS{Ky1pw@*p>w;Pr)ViS71+^}ybwRDqlkbJe_rm0R'
        b'Ve-8&`CgcOFHF7{Cf^H_?}f?t!sL5l^1bep@3+6Z!sL5l^1U$mUYL9@OuiQ;-wTuPg~|8A<a=TAy)gM+n0zlxz85Cn3zP4K$@jwK'
        b'dtvguF!^4Xd@oGC7bf2elkbJe_rm0RVe-8&`CgcOFHF7{Cf^H_?}f?t!sL71C*Qw${=r-*e4+4#!T<|D_g<KNFU-CdX5S05?}eXx'
        b'ulwlxYlA@61X&YgO_23@^t~|pUKo8ZjJ_8}-wUJfh0*uI=zC%Gy)gP-7=15{z86N{3#0Fa(f7jWdtvmwF#28?eJ_l@7e?O;qwj@Z'
        b'doPT>7e?O;qwj^$_rmCVVf4K)`d%1)FO0qyM&Apg?}gF#!svTp^t~|pUKo8ZjJ_8}-wUJfh0*uI=zC%Gy)gP-7=15{z86N{3#0Fa'
        b'(f7jWdtvmwF#28?eJ_l@7e?O;qwj^$_rmCVVf4K)`d;|8_rm0RVe-8&`CgcOulwZt^9^i$9?B}rz87ZS>puJbejw~J$hytGza9u?'
        b'Ets`n)`D3JW-XYtVAg_J3uY~twP4nQSqo+@m=!Q9U{=7afLQ^v0%irw3YZlzD_~Z@tbkbovjS!X%nFzlFe_kIz^s5-0kZ;T1<VSV'
        b'6)-DcR=}))S%Dra&|?L}3Wycxu>xQPz`BjTKfDTf74RzHRluu&R{^gAUIn}gcopy};8nn@fL8&p5?&>|N_ds<D&bYatAtkxuM%D*'
        b'yh?b5(f87?y%%QR3$yQq+4sV)y%$E`3#0Fa(f7KKzQ1+=WF^Q-kd+_{v+sr3_rmOZVfMZ7bMJ-G_rlM;7bf2elkbJe_rm0RVe-8&'
        b'`Cj+Q_t${~S_!lgXeH1}pp`%?fmQ;o1X>BS5@;3BDxg(BtAJJktpZvFv<hey&?=x+K&#MX6`HI9Sp~8RWZg#JAFt2b==<w-)CF0E'
        b'CaWM;L9BvU1+fZERspO6SOu^OU=_eBfK>pi09FC40$8`v_a{IFu?k&Q0j%5X`>O!Ny3M}7?|{l6t3XzPtO8lL(f8K|!K{W^_{sOe'
        b'@OxqSz3#*B&)b1k1FfDo)D5*7YBkhqsMS!bp;j+^%F<w~Lx(!F%Ydr^R|BpFTn)Gya5dm+z}0}O0apXA23!re8gMn>YQWWis{vP|'
        b'(`vZYaI4`~qtUv}zdv6;XtWw?-R9q)j{~(DYBkhqG+GU`HqhE=v^LDzFl)oC4YM}P+Gw;k$l4%lgRBj*Hpto_YopQH5NqSyaO2!?'
        b'!>bLiHoV&KYQw7yuQt5e@M^=W4X-x5+VE<_D@?x^rr%4y`d;_(_tyf*+Um77%-S$(!>kRnHd?I>vo_4a&%PJt-wX5ah57fw{Ci>k'
        b'y)ge?n13(KzZd4;3-j-V`S-&7dtv^)F#lede=p3x7v|p!^Y4ZE_rm;pVg9`^|6Z7XFU-H!eg6INx;jAX0IdVG4$wM4>j13-v<}cZ'
        b'K<fal1GEmBtOK(SnyiB+>p-l7ChGvK1F#OjIsoectOKwPz&bcJJn-tks{^kNygKmef>#&3y5Q9XuP%6X!K({iUGVCHSJ%WF!v(M|'
        b'fOY93_zPlP5bJ_i7sR?C)&;RHv{{!vg1<o41+p%Xb%Cr4WL+TZ0$CTxx<J+ivM%*m7tFd~)&;XJn03Ld3uav~>w;Mq%(`IK_5am%'
        b'Z#jx&F%(5dR8qx#_}KHGSpRhei1kZ|GLYoLq;}tiSp%~MW(~|5m^CnKVAk*Yy->dw>i0tZUZ~#-^?SYR_t*QUP`?-I_d@+%sNW0q'
        b'd*R&oLj7K--wXA7p?)va?}hrkP`?-I_d@+%sNW0qd!c?W)bEA*y->dw>i0tZUZ~#-^?RXyFVyda`n^!U7wY#y{a&cw3-x=UelOJT'
        b'^{(IleExzp1~&%({#IBx`Mr?87xMQ){$9x63n#x9>i0tZUZ~#-^?RXyFC6?{h~EqGdm(-=#P5apy%4__;`c)QUWnfd@p~bDFU0SK'
        b'_`MLn7vlFq{9cIP3-Nm)elNuDh4{S?zZc^7Li}Ec-wW}3A$~8!?}hli5Wg4V_d@($h~EqGdm(-=#P5apy%4__;`c)QUWni89lw8g'
        b'_`s_LuNJ&o@M^)U1+NyoTJUPYs|BwXyjt*T!K($Y7Q9;UYQd`ouNJ&o@M^)U1+NyoTJUPYs|BwXyjt*T!K($Y7Q9;UYQd`ouNJ&o'
        b'@M^)U1+NyoTJUPYs|BwXyjs5theG*YDBla^d!c+Ul<)N}-@hIZ#M%&RL#z$4HpJQxYeTFJu{Olo5NkuM4Y4-F+7N3)tPQa?#M%&R'
        b'L#z$4HpJQxYeTFJu{Olo5NkuM4Y6M7`^WosL#z$4HpJQxYeTFJu{Olo5NkuM4Y4-F+7N3)tPQa?HmnV>Ho)2dYXhteur|ICZg_Rz'
        b')qz(BULAOK;MIXw=fQXH1F#OjIsoectOKwPz&Zfy0IUPB4!}A9>j10+unxd_h3{W4G{ia(>m2-%<AAIKvJRH41F{atIw0$StOK$R'
        b'$T}eFfUE<u4#+wn>wv5SvJS{PAnSmv1F{atIw0$StOK$R$hsiwf~*U&F37qd>w>I{73)H*3$ZT5x)AF^tP8O&#JUjcLaYn1F2uSJ'
        b'>q4vxu`X7u3$QN0x>&I;yt?q}!mA6fF1)(%>cXoFuP(g0@an>=3$I??`zQ1lU|oQ90oDar7hqk0bph4|SQlVjfOP@Zt9t)@feK<F'
        b'doN_~h3vg_=6k)n_wNh8pK}4N0$K&M3TPG3Dxmf1-oKsz)GDZj?!9pAdm(!-WbcLSy^y^ZviCyvUdY}H*?S>-FJ$k9?7fh^7qa(4'
        b'_Fl-|3)y=idoN_~h3vhMy%)0gLiS$B-V4XR7qa(4_Fg#ly^y^ZviCyvUdY}H*?S>-FJ$k9?7fh^*E@Uv_<ao4tU#<ltU#<ltU#<l'
        b'tU#<_%?iK@zzWu^z^lNkz^lNk=!;<hRsdE2RsdE27S4Sygzts$y>#w-z03EnJ7_Y@3d{=33d{=33d{=33d{=33d{=33d{=33d{=3'
        b'3d{=3YM9k9t6^5dtj1TvhFJ}>8fG=jYM9k9t6^5dtcF<)vl?bK%xaj`Fsor!!>q=R)gY@uR)eetSq-upJ61!ihFFaqs{vMH$7*=h'
        b'@T%ce!>fi@4X+wrHN0wg)$pp}Rl}=>R}HTkUNyXGcs1~9;MKsZfmZ{s23`%k8hADEYT(ttt6A7N24D@q!lCc=uHL^dXf(`vb?@JI'
        b'(CnkxU~9nEfUN;r1GWZi4cHp6HDGJN)_|=6TT^<e;nqa2G~gP*HGpdX*8r{oTm!fUa1G!Zz%_tt0M`Jn-|2ghz6a@h-s$`6=aP5&'
        b'{&hR%KA8Jp?oYUa^gT%5gY-Q}--Gl$NZ*6>JxJe!^gT%5gY-Q}--Gl$NZ*6>JxJe!^gT%5gY-Q}--Gl$NZ*6>JxJe!^gT%5gY-Q}'
        b'--Gl$NZ*6>JxJe!^gTHEJ@4}U&*v{{V{l{eVDMn@;0X`r0;~sMJpk)>`5u(-LHQn(??L$<l<z_L9+dAv`5u(-d6(~>Ki~&qJrL`G'
        b'SP#T{Al3u19*FfotOsH}5bJ?h55#&P)&sE~i1k3M2Vy-C>w#Dg#Cjms1F;^6^+2o#Vm%P+fmjd3dLY*C^gT%5gY-Q}--Gl$NZ*5V'
        b'--Gl$NZ*6>JxJe!^gT%5gY-Q}--Gl$NZ*6>JxJe!^gZwN{S&MOuol2t0BZrP1+W&tS^#VP{#F>o??L<?#P31;9>niK{2s*bLHr)X'
        b'?|H}XUr+e=e-Ity??L_^<nPhJ??L|_^zT9c9`x^d_wQeWq1J+0>-Yb6H3;DI4&cA;0Jj$0T5xN@tqr&KU;L0a;M#y||85<E1U^XM'
        b'g9JWE;DZD{NZ^A6K1kq$1U^XMg9JWE;DZD{NZ^A6K1kq$1U^XMg9JWE;DZD{NZ^Ch--84`NZ^A6K1kq$1U~Ns{_}kcwKg`b4YW4U'
        b'+CXaqt&L4<!>kRnHa4vdvNkrYR|5a_3lFk3$l4(5fUJX6>(#)2KA;1#4tA{ru@1yK5bHp!1F>EW{MQozSqEetkaa-T0a*uR9guZE'
        b')&W@uWF3%oK-K|S2V~*&_n?6f8u*}r4;uKOfe#w^pn(q>_@IFg8u*}r4;uKOfe#w^pn(q>_@IFg8u*}r4;uKOfe#w^pn(q>_@IFg'
        b'8u*}r4;uKOfe#w^pn(q>_@IFg8u*}r4;uKOfe#w^yc_tB-_r}QF2K3~>jJC`ur9#50P6y*3$QN0y4bNUyt?q}!mA6fF1)(%>cXoF'
        b'uP(g0@an>=3$HG`y7214s|&9#yt?q}HohcXfOWNGU5Ird79#i{g3mjG|GNF?A1d78_iJ4+t6)~atXBg6^?-m@Y0WCARZt5Nd=SA0'
        b'5q#be{MYRm47Lhv71%1URbZ>YR)MVoTLrcXY!%oluvK8Iz*d2+0$T;P3TzeFDzH^xtH4%)tpZyGwhC;O*ebA9*s}_171S#1SpixB'
        b'TEU(bm=%~6m=%~6m=)|<0a?MG6^Ipx73^67SOHi8SOHi8SOHi8Scu?*2tMx!{`(GiO9y5JW(8&iW(8&iW(8&iW(8&iW<`I556}wG'
        b'O7v1kFE!W-*b3MR*b2TK25LpiR=`%kR=`$+tp-~Swi;|T*lMuVV5`AagRKTz4YnF=HP~vf)nKc^R)ehuTMf1vY&F<w>{$)98hch_'
        b'&uW;}Fsor!!>oo`4YL|%HOy+5)iA5EXEn%bkk#0;8hcg)tOi&Ouo_@Bz-oZiShE^lHN0wgHSlWS)xfKPR|BsGUJbk&cs1~9;MKsZ'
        b'fmfs0VF1<utN~a9um)faz#4!xiX8@G4a6FVHCnSqe}W&FH85*n*1)WRSp%~MW(~|5m^CnKVAjB_fms8y24)S+8kjXOYhc#Etbthr'
        b'vj%4UPT(5}d?SHxoc`W8{k@UEHxl?p0^dmB8wq?Pfo~-6jRd}t!1p_WfBil-68J^}-$>va349}gZzS-I1iq2LHxl?p0^dmB8wq?P'
        b'fo~-6jRd}tz&8^3MgreR;2Q~iBZ2RC0{`dp4G|0u1_y(K!Hp-}nA@1!nA?~;m^+v|m^+vYupWT*0IUaKy%PA(AJ_x29*FfotmpUt'
        b'e6<n5HzN2(1mB3@TM>LCf^S6djR?LG!8aoKMg-r8;2RNqBZ6;4@Qnz*5y3Yi_(lZZh~OI$d?SKyMDUFWz7fGUBKSrG--zHF5qu+p'
        b'Z$$8o2)+@)HzN2(1mB3@8xedXf^S6djR?LG!8aoKMg-r8;2RNqBZ6;4@Qnz*5y3Yi_(lZZh~OI$d?SKyod4cv;2RBmqk(TU@Qnt('
        b'(ZDwv_{RC~jRd}tz&8^3MgreR;2Q~iBY|%u@cmBUKR-woz(NAwNZ=a@d?SHxoB-eN0{&|-$XXz4fh;8O{Z8P&FC1OK=+>(Zwiei0'
        b'U~7S`1-3TW+F)ygt*uRK!>tXsHr(29Ys0M#w>I3`aBIV@4YxMj+Hh;btqr#}+}dz!!>tXsHr(29Ys0M#w>I3`aBIV@jXi6Ft&Kfv'
        b'L#++9Hq_csYh%yaKx+f74YW4)tc^WuW6#<UYh%ya0BZxR4Y2kI4v2Li)`3_DVjYQfAl89c2VxzFbs*Mx@Y!%c)>%t;z+{+pVAk3A'
        b'Ks;NMq1J&~2WlOtb)eRPS_f(!sCA&$fm#P@9jJA*Z5^<6z}5j<r}VCdTL*3(xOL#xfm;V|9k_Mi)`43GZXLLF;MRd#2W}lKS_f=h'
        b'uyw)K1zQ*24=>cZQ0qdi3$-rPy4bTW(7M>OF3h?x>vsm<$lx0pd?SNzWblm)zLCK<GWbRY-^kz_8GIvyZ)EU|48D=UH!}D}2H(iw'
        b'8yS2fgKuQ;jSRk#!8bDaMh4%=;2RlyBZKdE2LB21?a~E&Hr)BE3z!YFu6C^pv<hey&?=x+KnsV#H$wP+hw$HbIJ$t*aI4@}!L5Q@'
        b'2;mzcd?SQ!gz$|Jz7fLrJB0swK#;2-S3$0VTm`ubauwt%$W@T5AXh=If?Nf;3UU?XD#%rks~}fFu7X?zxe9U>Hm$;@RoJu&n^r-s'
        b'f?9=5tAJL3R)AK3R)AKpX$58ln^v%C1!4ta1!4ta1!4u8RsdE2Rshy3g#QF65Gz=<0<Z$GULpL~;8VJQ$rqo*fmwlBfmwlBfmwlB'
        b'fmwlBfmwlBfmwlBfmwlBfmwlB(Y6(!6`&QM6`&QM6`&QM)j+HNV&Q41)ljRoYBktuu+?Cz!B&H<23rlb8f-P#YOvK{tHD-d&uXaE'
        b'P^+O<L#@V|)mXC{W;NEV23ZZV8e}!dYLL|+t3g(StOi*PvKnMH$ZC+)Age)EgRI7y)ex&8Rzs|YSPiinVl~7Xh&5QV24D@q8mw6Z'
        b'uLfQXyqX6e{sXWEU=6?;C0YZqa3Xvog>R(rjTF9-!Z%X*Rtn!p;TtJ@BZY6A2H!Xhz7fLrJB0swpg?PY)&Q*mS_8BOXuV4KuLlaX'
        b'25Jq|8mKi;YoOLZt$|ttwFYVp)EcNYP-~#p?-YKJ!VgmTK?*-e;Rh-FAcY^K@Pib7kirjA_(2LkNZ|)5{NObBK?*-e;Rh-FAcY^K'
        b'@Pib7kirjA_(2LkNZ|)5{2+xNr0|0jevrZsQusj%KS<#RDf}RXAEfYu)8Gds{Gfy%l<<QReo(>>O87wuKPce`CH%Zg_&3~x6n>Dx'
        b'4^sF+3O`8U2Pym@g&&*-KPce`CH$a-ADjk12;m1I{2+uMgz)nY;lFOjT!8fetOsB{0P6u*55Rf=)&sB}fb{^Z2Vgw_>j78~z<L1I'
        b'1F#-|^#H5~U_Aiq0ay>fdH~h~upWT*0IUaKJpk(gSP#H@0M_pge$c@WI`}~cKj`2G9sHn!A9V184t~(V4?6fk2S4cG2Oa#NgCBJ8'
        b'gARVs!4Eq4K?gtR;0GQ2po1TD@PiJ1bQb&|gdc?PgAjfY!Vg0DK?pzZ5dQ0d-qHn323ZSaEs(WJgJIT!Sqo+@n6;w8Kx+Z51+*5>'
        b'!ddWxv*1T(!Oy#c|Gt3HaBIP>4YxMj+Hh;btqr#}+}dz!!>tXsHr(29Ys0M#w>I3`aBIV@4YxMj+Hh;bt&KfvgRKp=HrU!=YlE!~'
        b'wl?;xjXi4vtyc&C@%yrY)&^P|d)9_o8)j{owPDuAp0z>N23Z?qZIHFGXKjeJv1e`USsPw$c(vixfma7!9e8!%)qz*93jPz~1F#Oj'
        b'IsoectOKwPz&Zfy0IUPB4!}A9>j12?@mY8v)`3_DVx5D9=YXv9`yiS@1wW|Z2NnFFf*%|OKZxMx9l?J+P@r{y)&W`vXdR$+fYt$8'
        b'2WTCjb%53ZS_fzypml)O0a^!W9iVl9)&*J@XkDOnfz}0D7faTKSr=wqm~~;+g;^J7U6^%Y)`eLYW?h(dVb;ZxbwSnzSr=qoELj&~'
        b'T`XA_U|oQ90oDar7hqk0b+Ke!cy-~`g;y6|U3hij)rD6VUR`)~;njs#7hYX>b>Y>8R~KGgcy-}b!K;E-I0$|azz+_BAN22ogWyN`'
        b'`yhWG<nM$0ect)|*8q@JAgdOZj)GYQvkGPv%qo~wFsooz!K{K=1+xlf70fD_RWPeyR>7=-Sp~BSW);jTm{l;VU{=Abf>{N#3T73|'
        b'DwtI;t6)~atb$nuvkGPfW(8&iJ61qeKvqCjKvuA01!4ta1!4s|RsdE2R<L6QUIktSUIktSUIktSUIktSUIktSUIktSUIktSUIktS'
        b'UcJ)y&kvXYtN^S4tN^S4tN<*W0YC5f{rkew1wCh=6`&RU{XI}CvuEbl?H9KPx6jYp`|pF>H*UYUJ-8inAIyC)_rcr;b05rIn7c4{'
        b'VeZ1*g}EDZH|B24-T%-1c>naq+=ICXa}VYoJl})47jrM>Ud+9idoi~#w=lObw=lObH<%mD4dw=OgSm~ljk%4vjk%4vgSqgk;Z?({'
        b'hF1fx23`%k8hADEYT(tttNG{EC-n0VtnUk${7<Z}JNzT-`vRUb%$k2@ef>i~Yk<}Ot=Sk1wC3n3L#;V_wZYaLJ!QBxaBJY!z^#E>'
        b'1Gffl4cr>IHE?U-*1)ZSTLZTSZVlWTxHWKV;MTycfm;K&25!wyxBdgSqv@&'
    ),
    'DS0001.CSV': (
        b'c-n;BQO_*NksjuI0s0TTH@V1+$jHdLw7n8!K!OGBTE3>V<28X@a>1o#%fH?;Lvc=ied>{UL68}z`|G!>y87v;>W+W?hd=(~Z~ye`'
        b'`B&qA`t`s37ys^ufBeHA|Ld>*?GL~I`#=4kzn<gA|NOZ6zx@7B|Mb_t`{zIX_V@q!hhP7+@V9^b-QWNH4}bjC|M+)*_s1Xp`JaCs'
        b'&wQ@sfBVCK`QaaaJ^t#)TmP&7;pcn!n?L+tfBf;b|MS0zAOFvP{Nay(`rUv2?LYkLfBXIK{`Akk{*T`+{pzoO_g{YgJ0Hvc>i_xE'
        b'55NDrpR05KSpK)a{ZBvsonQU<zyAFn|L~uF_|@P3*MIu+<NNdf{SUwW)9?Q9Q?j2Q#=P(EZ+`r*AATLzBmAZG@Bi@6zx(lazyI?1'
        b'|IHu%_`Co255ND@FAw@}zApax{{H-@e)IF6x{CjQ_xSJr@OMA_`ak^PAO73_`j`LW$LIQ={^7SjZie6f^Phh8*MIoaFVFK|Hpu_}'
        b'+yC;z|MBC7`~F}2{AB;`_y74{{q29d+aG@X1AqIwfBflg|K_*<_)q`vyWju)um1fHfBfAa{_dBDyS_f0|MegK?!TYsr+@I8>0kfu'
        b'kN^08|MrhR{Q5u6zsjF(^z%db>mUB~+u!}e{iy%?w}1NWkN^KS_0w&C{Lc?R{fVFM=!ZYw&u`XWJ<p@UQK41Xe<}Rt{MFC@EB;a`'
        b'-DIhdDufDC1w}uKeiZ#E`q?Td`BCzt<VVSmk{=~+C2u8fC2u8fC2u8fCGXIV)r#JVz7>5d`d0L<=v&eEQjo3eTU&iA`d0K(^g4Q2'
        b'rR=5brR=56UW#6dUXzbk*_W~}ZT6+;OVO92FGXL9z7&0FtFNKwBo#dsJrz9_Jrz9_Jrz9_JrzB*(L>2Y$wSFQ$wSFQ$wSFQ$wSFQ'
        b'$)}P}C7((@m3%7sRPw3hQ^}{2PbD8pK9qbY`B3tq<U`4ak`E;xN-oJAN$yB;N0K|wR-sf_Dx?ab!c;-Y1-T>09YO90az~Img4_}0'
        b'jv#jgxg*FOLGB20N02*$+!5rCAa?}0(@HMK9XamEaYv3ja@>*QjvRNkl8bRij5}i75#x>+cS^~nxFf|KDeg#dM~XXA+$kj&;*Jn^'
        b'gt#Nb9U<-raYu+dLfjGJju3a2lFM*MhC4Fck>QREcT%5Pju3Z*xRaCjrR-ANk>bub#eMz}G46<QM~pjS+!5oB7<a_DBgP#u?uc<m'
        b'j62^L_xWVWaYv3ja@?8R?1J18<c=VB1i2%~9YO90az~Img4_}0jv&{9TnlpjXoqh}t|hsa<XVzzNv<WimgHKJYe}vpxt8Qwl50t>'
        b'CAmIoOtmQ2qFjq|Ey{IstIKjN%e5@mvRun@Ez7km*Rov8axKfXEZ4GJ%W^HtwJg`NT+4DT%e5@mvRun@Ez7km*Rov8axKfXEZ4GJ'
        b'%W^HtwJg`NT+4DT%e5@mvRun@Ez7km*Rov8axKfX`f)AFwJ6u>$F(Hal3Yu2t$th!axKWUAlHIi3vw;UwIJ7mTnln7$h9EX>c_Pl'
        b'*K%CTajkw_i*YT+wHVi8T#Ion#<dvNVqA-HEylGNx5c<E#%=ZEwiLIexGlwPDQ-(~TZ-FK+?L|D6t|_gEyZmqZcA}nirebPZ6R(8'
        b'aa)MnLflqAZp(06hTAgSmf^Mxw`I63!)+OE%WzwU+cMmi;kFF7Ww<TFZ5eLMa9f7kGTfHocIgU=UB!yBOL1F@+fv+?;<gmGrMNA{'
        b'Z7FU`aa)SpQruqpytc)-EyiszZi{g{^?7Z}aa)eta@>~Vb~3vlw*|Q^$ZbJx3vxTS%S&=wlG~Ep4rUkSwkWqnxh=|VQErQJTa??P'
        b'+!p1wD7V>?+p^r2<+d!hWw|ZO?ZE?`Ft>%dEzE6UZVPku4Raq4tE9P-=1Q6?X|ANXlIH3OKHjODD7!dU;#`SyCC-&NSK?fWb0yA|'
        b'I9K9aiE|~+l{i=8T#0if&XqV<;#`SyCC-&NSK?fWb0yA|I9K9aiE|~+l{i=8T#0if&XqV<YRQ#aawW`_FjvA{33DaPl`vPrTnTd}'
        b'%#|=#!dwY+rIuXDawW@^ELXBzsU=sUT#0fe%9SWrYRQ!(SCU*wawW-?Bv)$5l^|DwTnTcemR!kkCC8N<S8`m*aV5u<99ME&$#Ese'
        b'm0EHo#+4XXVqA%FCB~InawWx;6jxGQNpU5`Ew$v95VwT5CB!WuZV7Qqh+AsOEg5dfa7%_;GTf5kmJGLKxFy3a8E(mNONLuA+>+sz'
        b'47X&sCBrQlZpm;<hFdb+lHryNw`8~_!>z3j#S-F{5VwT5CB!WuZV7Qqh+9J365^H+w}iOGmfVu!mK3)tcwfpc#w{^!iE&GeTVmW2'
        b'<CYk=#JDBKEirD1aZ8L_V%!qrmKe9hxFyCdF>Z-*ON?7$+!EuK7`Mc@CB`i=Zi#VAj9X&d662N_x5T(5#w{^!iE&GeTVmW2<CYk='
        b'#JDBKEirD1aZ8L_V%!qrmKe9hxFyCdF>Z-*DaNH3mttIsaVf^77?)yPig78%r5KlDT#9ii#-$jSVqA)GDaNH3mttIsaVf^77?)yP'
        b'ig78%r5KlDT#9ii#-$jSVqA)Gsa9M{aVf>66qiz5N^vR0r4*M^TuN~%#ibONYQ?1xmqJ{s6_+wx%5W*e<<eJmQ;16;E`_)h;!=o9'
        b'Auffu<o8umic2XjrMQ&hQi@9{F25=6yT_~SVqA)GDaNH3mttIsaVf^77?)yPig78%r5KlDT#9ii#-$jSVqA)GDaNH3mprSQlRLa1'
        b'mx5dhaw*8AAeVw%3UWDuLA2Q=xsc@I(Qz(Bxe(>zDc;x8O;UwWVXC0)vRue=A<Km<7qVQ)av{rwEElp|$Z{deg)A4cT*z`E%Y`f#'
        b'vRue=A<Km<7qVQ)av{rwEElp|$Z{deg)A4cT*z`E%Y`f#vRue=A<Km<7qVQ)av{rwEElp|s2>-iT!?ZZ%7rKwqFjh_A<BjNaUsct'
        b'Bo~rgNOB>`g(Mg1$Auslf?No4A;^Uw7lK>}av{itAQysM2y!9Fg&-G#TnKU@$b}#mf?No4A;^Uw7wX4_`f(w~g%}rNT!?WY#)TLc'
        b'VqB;n7gAhEaZ`$$>c>qXZVGWz{kSQ^O&M;=a8rhxGTfBmrVKY_xGBR;8E(pOQ-+%|+?3&_3^!%CDZ@<}Zpv^|hMO|nY<)_fLfjPM'
        b'rVuxUxGBU<A#MtBQ;3^F+!W%b5I2RmDa6gvRj^Zvn^N4A;-(ZgrMM}@O`cYrV%!wtrWiNHxVdyTeadlDj+=7al;frxH|4k~$4xnI'
        b'%5jrlb(pzXQFcjgN^(<@o08m=<fbGyLmi6ApC6gB+?3^}EH`DjDa%b+Zpv~~mYcHNl;x%@H)Xjg%S~Bs%5qbdo3h-L<)$n*Ww|NK'
        b'O<8Wra#NO@vfPyArYtvPxgpC9S#HR3gAKVM%ne~~2y;W28^YWW=7umggt;Ni4PkBwb3>RL!rTz%hA=mTxgpFAVQvU>Lzo-F+)zVq'
        b'$Z|uL8?xMx<%TRbWVs>B4Owo;azmCIvfPm6hAcPKkQ<`h5aos_H$=H1$_-I&s3A8bxgp67Np47TLk+nh$PGbm2y#P^8-m;r<c1(O'
        b'1i2x|4K?J38gfI78)DoL<AxYF#JC~G4KZ%0Ave^J8$#R=;)W16gt#HZ4Iyp_aYKk3LfjDIh7dP|xFN(1A#Mn9Lx>wf+z{f15I2Ol'
        b'A;b+KZU}Khh#NewI;1#Ebx2KhNKJJ}O?Ak7s>5fYR9Gsc3ZcSOLCKGjA0<CZew6&Y`!zMyAvM(@HPs<C)gd+2AvM(@@2L)-kGGY*'
        b'mA$pqThUw5ThUw5ThX_oZ${sWz7>6Qvu|bJ%D$C-EBjXVQufjwUy5FeUfSxV<du4|O3_QvYv{g~vM+A;rSMDPm%=ZFU)t>W0g}{U'
        b'hr9<nd=ygIQ`<ciJrz9_Jrz9_J+;|W$y3Qg$wSFQ$wSFQ$wSFQ$wSFQ$wSGfl20X{N<Ni*D*06MspM11r;<-4A4)!yd?@))@}cBI'
        b'$%m2;B_B#I%N<$n$Z|)PJF?u7<&G?OWVs{D9a-+kaz~asvfPp7jx2|{4ym~gsksiRxelqh4ym~gsksiRxelqh4ym~gsksiRxelqh'
        b'4ym~gsksiRxelqh4ym~gsksiRxelqh4ym~gsksiRxelqh4ym~gsksiRxelqh4ym~gsksiRxelqh4ym~gdCztDFb)Pgqy{^r20Nq%'
        b'JER6Xqy{^r20Nq%JER6Xqy{_WJ=o#%Ig#Ry6nCV!BgGvl?nrS*iaS!A20P?E*x|Dx#~nHD$Z<!GJ96AL*x|c}rSO8>5#){_cTINq'
        b'{Ogk3HQM33n`~B;UY0wu+>zxp+aWdEAvN0}HQOOI+aWdEAvN0}HQOQY*$$sUMzEqSFV3|%*N;B3wmjGJT+4GU&$T?)@?6VvEzh+)'
        b'*YaG;b1l!cJlFDE%X2NywLI7IT+4GU&uO?rYPdscxI=2VLu$A~YPdt*!yUeRSjsNZwM5qvT}yN=(X~X^5?xDlEzz|^*AiVzbS=@f'
        b'MAs5sOLQ&KwM5ry%(Xz*0$mGqEzq?<*8*J&bS==eK-U6Yt1H*?T+4GU&$T?)@?6VvEzh+)*Xqi(x^gYewKUh#TuXB;&9%C6EzGqr'
        b'*TP&2bFHpi%W|!*T&pYBl3Yu2Ey=Yc*Xqi(AlHIi3vw;UwIJ7mTnloou3W1t*J50Yaa)Yr>dI{?ZcA}nirZ4$mg2S)x23o(#ce5W'
        b'OL1F@+fv+?;<gmGrMNA{Z7FU`aa)SpQruQoZVPc+h}%Nk7UH%Lx7C%~GTfHoHqWnaA?_OQ@ZC+6U5eXM+?L|5`3~Rxb%ht>witH}'
        b'c=-Gya-0S{<UQcwyGpVWtZ3WIa$A<$vfP&Cwk)@0xh>0WS#I+b>ow!yyNX&=*OZ6vD%|r0x-HOcfo><ay+pSqx-HRdiS8Qn@cGw8'
        b'x-HUa&O>U>Lu$@LYR*I6a~?hm$}ZDwnQqH;Tc+DG-InRLOt)paEz@n8Zcpy>LfsbXwotc)x-Hagp>7LxTd3PY-4^Ofs4Jnagt`*y'
        b'N~kNLu7tW0>Po09p{|6w66#8*E1|B0x)SP2s4Jnagt`*yN~kNLu7tW0>Po09p{|6w66#8*E1|B0x>94VWV%vgu0*;L=}M$4k*?I3'
        b'D~YZox{~Nhjkyx&N{zWvW3JSgD>ddym@8qfgt-#tN|-BQu7tS~=1Q0=HRei<xl&`U)R-$Z=1Pt$Ij-cmQe&>fxDw+^j4Ls&)R-$N'
        b'uB5n9W3Gg_65>jTD<Q6gxDw(@h$|tkgt!voN{A~Vu7tP};!21sHRei&D;ch2xRT*YhASDaWVn*yN`@;LZpm;<hFdb+HRIuf$V-S@'
        b'LfjJKmJqjuxFy70LmoaKuN1eWxFy9cDQ-z|*OZ6v9#*iT@N(Re;}*aDumrg!$X$aTzPm4_m*kctw<NhG$t_84Npee)Taw(8<d!72'
        b'B)KKYEjH(tD7QqpCCV*PZi#YBlv`}gEm>~Ka*LPEEn#j6b4!?8!rT((mN2)3xh2dkVQ#T8x1_lx%`IteNpnk@ThiQ;=9V<Kq`4){'
        b'Eop8^b4!|A(%h2fmNd7dxh2glX>Lh#OPX8K+>+*&G`FO=CCx2qZb@@Xnp@J`lIE5)x1_lx%`IteNpnk@ThiQ;=9V<Kq`8#lQkqL?'
        b'F4d7sVJ?Na6y{QxOJOdBxl~6kWx15)QkF|uE@in?M=sToOGz#zxs>Em9k~?bQjkk^<Wi1HIWFb6l;cv4OF1s(xRm2kj!QW%<+zmN'
        b'QjSY?<Wh`FF)qcp6ys8iOEE6hkxMBqrMOf_E`_)h;!=o9Auffu6yj2dOCc_WxFq6Ih)W?Zd2ThOxRl~jic2XjrMQ&hQi@9{E~U7X'
        b';!=uBo>@&XF2%SM<5G;fMm>D@*Ogt4OF1s(xRm2kj!QW%<+zmNl3#pCK`sTk6y#EnOF=FLxfJA5kV}5=Atkw#<U*1QN$#5T@bTA!'
        b'jk%EJLY50zE@Zio<wBMVSuSL`kmW*_3t28?xsc^TmJ3-fWVw*#LY51D@gan{5avRd3t=vVxe(?;m<wSpgt-vrLYNC-E`+%d=0cbY'
        b'VeXps@VTX=xsc{UnhR+zq`8phLYfO{E~L4T=0chaX)dI>kmf>~3u!K-xsc{U=T}3R3t=wQkqcQaWVw*#LY50zE@Zio<wBMVSuSL`'
        b'kmW)hxe(<-lnYTVM7dB$E+n~-<U*1QNiHP0kmN#=3rQ{{xll(g1i298LXZnVE(Ey{<U)`OK`zvh3pp<2xGBd?Ic};WH`S4wQrwi{'
        b'rW7}&xGBX=DQ-$}Q;M5X+?3*`6gQ=~DaB1GZc1@eiknj0l;Wm3a#M(#LfjPMrVuxUxGBU<A#MtBQ;3^F+!W%b5I2RmDa1`7Zt~3P'
        b'R7-9OaZ`w!LfjPMrVuxUxGBU<A#U=->g4arO)+lrcjcxWH|4k~$4xoz8uswrUsrlTZVGZ!keh<s6y!AQAvNqF?_m$$-9%enl$-ob'
        b'k}1ngS#HX5Q<j^u+?3^}EH`DjDa%b+?i%*+`J4!IQ<$6lMUpAaO=)gQb5oj|(%h8hrZhLDxhc&}X>Lk$Q<|I7+?3{~G&iNWDa{RO'
        b'Zb)-Onj6yGkmiOoH>9~C&0W(TK6b?+&JA&Hh;u`n8{*s$=Y}{p#JM5P4RLOWb3>dP;@lADhB!CGxgpLCac+onL!2Ap+z{u6I5)(('
        b'A<hkPZisV3oEzfYP)}}1b3>XN(%g{dhBP;%xgpIBX>Le!LoK<XmfVo#hFWq%lpCVl5aos_H`J0FlH8Ewh9oy6xgp67Np7emHw3w%'
        b'mfVozh8#E4k{e>&5aWgzH^jIh#tkuUh;c)V8)DoL<Az#tLy8+x+)zty2ysJ*8$#R=;)W16gt#HZ4I%Ct_wZ>0DQ-w{Ly8+x+>qji'
        b'6nBk#`0nw(W)IDK2+ey4&3g#VdkD>Y2+ey4&3g#VdkD>Y2+ey4&3g#VdkD>Y2+ezld)~umLCNuzhtRx-(7cDxyob2wJ$(L=R`yo*'
        b'R`yo*R`%9bZ$)oK@1cZSn|&+#R`jjtThX_oZ$;mVz7>6ItCy0Ol9!T~lH<onLh~Ntp7-!sQ1nvtQuL+hOVKfcH8kxZH0>cY?IASn'
        b'A?|4pp9Mu<ik^y|ik{l)spP5TspP5TspP5TspO&Lq2!_Dq2!_Dq2!_Dq2!_Dq2yD^r;<-4pGrQJd@A`=@~Pxg$)}PJB?sj~(;h<8'
        b'9zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X'
        b'(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|Lem~X(;h<89zxR|'
        b'Len1Np7!urP;x2mNO2hU5cjZ$kGF#(#T_Z`NO4DsJ5t<{;*Jz|q_`u+9VzZeaYu?fQrwZ^judyKxFf|KDeg#dM~XXA+>zq0Ne`d<'
        b'j~I9GeTUGjhqz}wd=`{ljyrPPk>idWcjUMu#~nHDOnpqh20eUNQFcM@2y#b|JA&L1<c=VB1i2%~wIJ7mTnlpj=rGrkTuX8-$+aZc'
        b'l3Yu2Ey=Yc*OFXIaxKZVB-fH$OL8sAwItV)TuXAA^AMWz5SsH4n)48v^APu(hwuKnvdeNU%e5@mvRun@Ez7km*Rov8a=mqgZ(**5'
        b'xfbSHm}_CKg}D~yT9|8Lu7$Z4=31C*VXlR_R!6R7xt8TxmTOtAWx1B+T9#{Bu4TEF<yw|&S*~TdmgQQOYgw*kxt8TxmTOtAWx1B+'
        b'S{=C-<ysxNmgHKJYe}vpxt8Qwl50t>CApU5T9RvZ<XRoMmg8EEYdNmvxR&Euj%zut<+zsPwj8(RxGl$RHRQG!x5c<E#%(cfi*Z|w'
        b'+hW`n<F**L#keiTZ82_(aa)YrV%!$vwivg?xUGiVmg2S)x23o(#ceg@wh*_4xGltOA#MwCTZr32+!o@t5VwW6O~h>>ZVPc+h}%Nk'
        b'7UH%Lw}rSZ#BCvN3vpYB+d|wH;<gaCg}5!mT@xNYx0DpOrMNA{X~09=10FtZBF1eoZi{hSjN4+|7UQ-Ux5c<E#%(cfi*Z|w+hW`n'
        b'<F**L#keiTZ82_(aa)YrV%!$vwivg?xGlzQF>Z@-Ta4Rc+!o`u7`Mf^EyiszZi{hSjN4+|7UQ-US7Ka=aV5r;7*}FkiE$;yl^9oI'
        b'T#0cd#+4XXVqA%FCB~H)S7Ka=aV5r;7*}FkiE$;yl^9oIT#0cd#+4XXVqA%FCB~H)S7Ka=aV5r;7*}FkiE*X#swKsh6jxGQNpU5`'
        b'l@wP}TuE^y#g!CSQd~)KCB>B#S5jO_aivyV32`OFl@M2I#gz<KGF-`UCBu~rS2A45a3#Z)3|BH-$#5mZl?+!hT*+`H!<7tIGF-`U'
        b'6@q&SaV5l+5LZH632`OFl@M1#TnTX{#FY?NLR<-PCB&5wS3+C~aV5mn&}a0L;+7P54R`psuO-GUF>Z-*ON?7$+!EuK7`Mc@CB`i='
        b'Zi#VAj9X&d662N_r_m0f(GH=}4x!NwagTQREGW7hx8%4b$1OQ-$#K_UhtKCkkXwS>66BU3x3;b#Sd!e5<d!72B)KKYElF-ka!Zn1'
        b'lH8KymL#Xi4xz~op~(($Pj>h$D7q-OM7br(Em3ZXa!Zt3qTCYYmMFJGxh2XiQErKHOO#up+!E!MD7QqpCCV*PZi#YBlv|?Q66KaC'
        b'w?w%m$}LfDiE>NrxFyLgNp4AUOOjiX+>+#$B)25FCCM#GZb@?2T!)XNk0iGwxh2UhNp4AUOOjh^$1OoF1-Vo^F6FqC<5G@GIWFb6'
        b'l;cv4OF1s(xRm2kj!QW%<+zmNQjSZt<5G-EF)qcp6ys8iOEE6RxD?}3j7u>t)s9OkE~U6@uEPg2QjAOW<5G%CDK4eBl;To~ODQg;'
        b'xRl~jic2XjrMOf-E`_)h;!=o9Auffu6yj2dOCc_WxD?`2h)W?Zg}4;rQiw|-E`_)h;!=o9Auffu6yj2dOCc_WxD?{9u@0X{04Xk|'
        b'xNEG#clV|2VqA)GDaNH3mttIsaVf^77?)yPig78%r5KlDT#9k2hFpqqDaNH3mttIsaVf^77?)yPh;bptg%}rNT!?WY#)TLcVqA!E'
        b'A;yIm7h+t9aUsTq7#Ct(h;bptT_YVnju|1xg&Y?x&4eHqf?No4A;^Uw7lK>}av{itAQx)Kg&Y@hT*z@D$Augha$NB8xDez*kPAUB'
        b'1i298LXZnV?waZF8AOr`NiHP0kmN#=3rQ{{xsc>Sk_$;LB)O2}LXrzfE+n~-<U*1QNiHP0kmN#=3rQ{{xsc>Sk_)xtLXZnVF4T?-'
        b'IWFY5kmEv*3pp<2xRB#Qjte<1<hYRILXHbLF66k7<3f%LIWFY5kmEv*3pp<2xRB#Qj+=7al;frxH|4k~$4xnI%5hVUn{wQg<E9)p'
        b'<+v%wO*wALaZ`?)a@>^TrrL2+jGJQIR6A};aZ`$$Qrwi{rW7}&xGBX=DQ-$}Q;M5X+?3*`6gQ=~DaB1GZmJtMg}5oiO(AXyaZ`w!'
        b'LfjPMrn+%chMO|nl;NffH)XiV-;SF?+!W%b5I2RmYoNn-e_h$7xGBX=Deju+@cGxpxGBa>G42}Z@ZBHLc9-L(95>~-DaTDYZpv}j'
        b'K!?vDg52aUk4#B!N^(<@yCyn({t;15BOO8`9YP}=LL(hQBOO8`9YP}=LL(hQBOO8`9YP}=LL(hQBOO8`9YP}=LL(hQBOO8`9YP}='
        b'LL(hQBOO8`9YP}=LL(hQBOO8`9pWD8@L5oDS#HR3LzcTHI()nihA=mTxxpU;9@5;9=7uykq`4u@4QXyjb3>XN(%g{dhBP;%xgpIB'
        b'X>Le!Lz)}X+>qvmG&iKVA<YeGZfsoxJ;b>o&JA&Hh;u`n8{*s$=Y}{p#JM5P4RLOWb3-k;A<YeGZb)-Onj31#4PkBwb3>RL!rV|x'
        b'Zpd;&mK(C%P)lxzazm6GqTCSWhA1~gxgp98wd95*H`J0Fg4__~h9Ec8k{fc|P)lxzaYKw7V%!kph8Q=*xFN<3F>Z))*Gz}cqoo`-'
        b'<hUWn4LNSeaYK&7OoyqN4pTE7re-=!&2*TW=`c0ZVQQws)J%tY&vf{*L8gW}ObvCI8tO1L)M09<!_-iRsi6+@9_sK}Q1o}kP0e(e'
        b'_e_V+f|9qA_tx{?ir$Leir$Leir$L86@4rEZYA8>>RZvbqHjgtioO+nD|#t<9VIMn_EPjx^iuRv^iuRv^iuSt=!?;pqAx{XioO(m'
        b'Df&|M#lzfE_Eh%N9-oSyik^y|ik^y|ik{l)spO&Lq2!_Dq2!_Dq2!_Dq2!_Dq2yD^r;<-4pGrQJd@A`=@~Pxg$)}PJB_B#Ylzb@p'
        b'Q1YSVL&=Ad4<(o6jwE*^xg*IPN$yB;N0K{|+>zuk(_w0+!_-WNshJK_GaaU8I!w)Un40M@HPc~gro+@shpCwkQ!^c=W;#sGbeNjy'
        b'Fg4R*YNo@~OoyqN4pTE7re-=!&2*TW=`c0ZVQQws)J%t|nGRDk9j0bFOwDwdn&~h#(_w0+!_-WNshJK_GaaU8I!w)Un40M@HPc~g'
        b'ro+@shpCwkQ!^drJ=5W{pyWc_5#o*zhmj6bBORtjI!ujpm>TIYHPT^fq{GxmhpCYcQzIRwMmkK5beJ0HFg4O)YNW%|NQbGB4pSo?'
        b'<~`Ekv!LWM+>zmq40mL>Bf}jT?#OW0K!=ad8P`OI?<&eJ#T_Z`NO4DsJ42t+j~I8vxFg0LG46<QM~pjST#Ir2=%Z@OaV^KS9M^JO'
        b'%W*BowH$X%bodM+$h9EXf?Nx7Ey%SX*MeLNaxKWUAlHIi3vw;UwIFv*bohMUCApU5T9Ru?t|hsa<XVzzcH~-=Yf-L6xfbPGlxtD0'
        b'MY$H`T9j*1u0^>P<yw?$QLaU~7Uf!$Yf-L6xfbPGlxtD0MY$H`T9j*1u0^>P<yw?$QLaU~7Uf!$Yf-L6xfbPGlxtD0MY$H`T9j*1'
        b'u0^>P<yw?$QLaU~7Uf!$Yf-L6xfbPG?YLGut_8Uk<XVtxL9PY4Ry(fcxR&Euj%zut<+zsPT8`Ut+*Uhoi*Z|w+hW`n<F**L#keiT'
        b'ZMEaJ6t|_gEyZmqZcA}nirZ4$mg2S)x23o(#ce5WOL1F@+iJ&cA#MwCTZr32+!o@t5VwW6t#;g&;kFESO?3G1;1=Sx5VwW6EyQgh'
        b'Zu7M2mg2S)x23o(#ce5WOL1F@+fv+?;<gmGrMNA{Z7EJ89i~P)OpSDy_eh8D?n@t5w;;C#xh=?DGabJBBT6sHZAorRa$AzylH8W$'
        b'wj{ShAKA7jw?(-v%570@i*h@--DSBg%WYY1%W_+m+p^r2<+d!hWw|ZOZCP&1a$A<$vfP&Cwk)@0xh>0WS#Ha6Tb3(Xu4K8A<w}+-'
        b'S*~QclI2R4D_O2&xsv5dmMdAVWVw>%N|q~Gu4K8A<w}+-S*~QclI2R4D_O2&xsv5dmMdAVWVw>%N|q~Gu4K8A<x2gy66H#iD^adQ'
        b'xf10{{kW3kN|Gx{t|YmV<Vun&Nv<ThlH^K~D@m>-xsv2c{kRh3N{}n{<4XOw65~pYD>1Irk1Hvzq_~peN{TBfuB5n<;!27u_2Wv2'
        b'D<Q6gxDw(@h%5EuN`@;Lu4K59;Yx-p8LniwlHp2*D;ch2xRT+nsSY0!E+MXjxDw(@h$|tkggA|Lm>TOaHP&I?V;w#Vt-@BJR9Gsc'
        b'3ZcSOLCK}KCB-c%Zb@-Vid$0LlH!&Wx1_iw#VsjrNpVYxTT<MT;+7P*q_`!;Eh%nEaZ8F@Qrwc_mK3+7xFy9cDQ-z|ONv`k+>+v!'
        b'6t|?fCB-c%Zb@-Vin}H{d>qM^7`Mc@CB`i=Zi#VAj9X&dVnc4paZ8R{a@=A+ZV7TrkXwS>66BU3w*<K*$Spx`335x2TY}sY<dz_}'
        b'1i2Nucw|X(OOjiX+>+#$B)25FCCM#GZb@=Wl3SA8lH`^ow<NhG$t_84sT;Qhxh2RgL2e0hOORWF+)_7g$#E&ir5u-XT*`4N$E6&X'
        b'a$L%BDaWN8mvUUnaVf{89G7xj%5f>jr5u-XT*`4N$E6&Xa$L%BDaWN8mvUUnaVf{89G7xj%5f>jr5u-XT*`4N$E6&Xa$L%BDaWN8'
        b'mvUUnaVf{89G7xjsvDPLT#9ii#-$jSVqA)GDaNI`aVf>66qiz5svDO=Tnce1#HA3ILR<=QDa54^mqJ_$aVf;55SKz+3UR4!T*`1M'
        b'!=((DGF-}VDZ`}<moi++a4Ewjzvz%c+%?nTyPGKc)Tfpd<5G-EF)qcp6ys8iOEE6RxD?}3jJu{fd^{&1$Augha$Lx9A;*Or7jj(i'
        b'r$<7N3qdXfxe(++kPAUB1i298LXZnVE(Ey{<U)`OK`sQj5adFT3qdXfxe(++kPAUB1i298LXZnVE(Ey{<U)`OK`sQj5adFT3qdXf'
        b'xe(++kPAUB1i298LXZnVE(Ey{<U)`OK`sQj5adFT3qdZ}jtfaHB)O2}f|tjIC>Nq!h;kvyg(w%IT!?ZZ%7rKwqFjh_A<Bg)7ouE<'
        b'av{ovC>Nq!h;kvyg(w$l$Au&pYR82j7i!0a92at2$Z;XZg&Y@hT*z@D$Augha@<rqZi;bJjGJQI6yv5CH^sOq#!a>3rW7}&xGBX='
        b'DQ-$}Q;M5X+?3*`6gQ=~DaB1GZc1@eiknj0l;WloH>J2K#Z4)0N^w()n^N4A;-(ZgrMM}@O(||laZ`$$Qrwi{rW7}&xGBX=Deju;'
        b'@Iinn#!WG9igDLihtEGE$4xnI%5hVUn{wQg<F2s|-#uPs7v!coa#N0*a@>^TraE#{jJu{fd?u9RrW`lrxGBd?Ic~~vQ;wT*+?3;H'
        b'FuEW&1-U86T|*r{|A-_vB{>asm>TLZHPm5hsKeAyhpC|s^B(H(-NVv$m*u7`H)Xjg%S~Bs%5qbdo3h;GHF8s!8^YWW=7umggt;Ni'
        b'4PkBwb3>RL!rTz%hA=mTxgpFAVQvU>Lzo-F+z{r5FgJv`A<PY7ZU}Qjm>a^}5axz3H-xz%%ne~~2y;W28^YWW=7umg_^s6;%?)X8'
        b'NOMD)8`9j6=7uykq`4u@4fW)PFgJv`A<PZ+<c2IaWVs>B4Owo;azmCI>d6gJZisS2lpCVl5aos_H$=H1$_;hoh9oy6xgp67Np47T'
        b'Ly{Yk+>qpkBsbKN8-m;r<c1(O)R7x<+>qmjI&wpd8)DoL<AxYF#JHi3+>qji6gQ-}A;k?TZb)%MiW^egkm80qa+vAx{CVQR{Q3Rl'
        b'qi|H{UkYE|i=IFKn*8fcN`<9Dst_tn6$T3DQIMnTr&Un&Q~V>!ew6(v`%(6z?5)k-M~}Cay_LPS*;~<D(Oc15(Oc2CqHk^Wt>jzD'
        b'w~}uq-%7rfd@K1@@>23r@>23r@>23r@>23r@>23r@}=ZU$(NEZC0|Oulzb`qQu3wbOUYBoQ^`}wQ^`}wQ^`}wQ^`}wQ^`ZgL&-zQ'
        b'L&-zQL&-zQL&-zQL&>L-PbHs9K9zhb`Bd_$<WtF~l20WcN<Nf)DEUzGq2xo!hmsE^A4)#nlOH+m$Z_X(+(%)nP%11HQiV`qs-WaY'
        b'$&ZpBB|l1jl>8|9QSzhYN6C+px01J#x01J#x01J#x01J#x01J#ZzbPKzLk6{`Bw6+<Xg$Nl5Zv7N?uA{N?uA{N?uA{N?uA{N?uA{'
        b'O1_kQDfz-D)g#0mA?^rqM~FK@+!5lA5O;*QBg7pc?g(*5h&w{u5#o*zcZ9ej#2q2-2ysV<J3`zM;*Jn^gt#Nb9U<-raYu+dLfjGJ'
        b'ju3Z*xFf_JA?^rq=SJMei*PFXcyyDY>_gdyvJYh+%085RDErV>2jg0dYca0HxEAADjB7Ej#kdyZT8wKkuEn?(<64YsF|Nh97UNos'
        b'Yca0HxV{<pWv^~IuI0GC9rt-(%HGP}%HGP}%HGP}+Ui@;x1w)t^{wPv$+wbUko$Ogx1w)F--=#}UW#6dUW#6dUW#7Y=%wVP<fY_G'
        b'$(NEZC0|Oulzb`qQu3wbOUaj#r;?|Vr;?|Vr;?|Vr;?|Vr;?|VhmwbqhmwbqhmwbqhmwbqhmwbqPbHs9K9zhb`Bd_$<WtF~l20X{'
        b'N<Nf)DEUzGq2xo!hmsE^A4)!y9E{sy+!o`u7`Mf^EyiszZi{hSjN4+|7UQ-Ux5c<E#%(cfi*Z|w+hW`n<F**L)s5Ry+?L`t6}P3h'
        b'EyZmqZcA}nirZ4$mg4rUxQ`%P*|)a(R`jjtThX_oZ$;mVz7@R~y%fC^y%fC^y%fC^y%fC^y%c?EhhIv*lzb`qQu3wbOUaj#FC||}'
        b'o=Todo=Todo=Todo=Sc(?qeU#$%?XvvWKz<4|A8~J}XKeN*_ue+U}w49?CwIeJcCZcAwhrQ^}{2PbHs9K9zhb`QTx0DEd(Jq3A=='
        b'hoTQfABsK{9hNIuu4K8A<w}+-S*~QclI7~Q+?Op?!dwY+CCrsDSHfHgb0y4`FjqI`J`zH6CC!yISJGTbbER{uCCrsDSHfHgb0y4`'
        b'FjvA{sU=skT*-1J%atrwvRuh>CCim8SF&8GBUhqaiE<^%l_*!DT#0fe%9SWrqFjk`CCZg3SE5{rawW=@C|9CfiE<^%l_*!DT#0fe'
        b'%9SWrqFjk`rH)*wBUgf4334UKl{#{zj$DaxCB~H)S7Ka=aV5r;7*}FkiE$;yl^9oIT#0cd#+4XXVqB>sS5jO_aV5o-6jxGQNpU5`'
        b'l@wP}+>+v!6t|?fCB-c%Zb@-Vid*W)Eg^0RaZ89>LfjJKmJqjuxFy6bb>!A<xG%a}H{w1k%HGP}%HGP}%HGP}%HGP}+Ul*X-b%ie'
        b'd@K2GJ-u7ex1w)F--^Du*|)N9WiMqfWiM^@QuI>v%W)q;O4&=By%fC^eJT1<^rh%a(U+nxMqi4)6n!cBQuI{xRP<EzRP<Ez)DE9Y'
        b'o=Todo=P5%p7&7nQ1nprQ1nprQ1nprQ1nprsf|9Bd@A`=@~Pxg$)}P}C7((@m3%1qQ1YSVL&=Ad4<#Q;K9qbYIVhK+T#9lj%B3im'
        b'qFjn{DaxfNm!e#Xaw*ECD3_vKigGE+r6`x8T&f+HYR9D@mx5dhaw*8AAeVw%3UVpPr68ArTnch2$fY2cYR9D<mvUUnaVf{8+Hony'
        b'r5KlDT#9ii#-$jSVqB^nmr`6xaVf>66qiz5N^vR0r4*M^T&f+HLR<=QDa54^mqJ_$aVf;55SKz+3UMjKr4W}wTncfic3jGEDZ`}<'
        b'moi++a4Ey343{!o%5W*er3{xcT*`2H8}18{Q;16;E`_)h;!=o9o>t9Uai0~19|}LT*@v=^<X>0#q402A$Z;XZg&Y@hT*z@D$Augh'
        b'a$Lx9A;*Or7jj(4aUsWr92at2$Z;XZg&Y_Bu0!0A`>ZH?D|;(@D|;(@D|;(@PX^Iu-`eb3(YK;+Mc<0P6@4rER`jjSzLmU`yp+6@'
        b'yp+6@yp+6@yh7Wy6ulIEafe^ZzLb3_`%?C$>`U2~w)#@^rRb@xo=Todo=Todo=Todo=Todo=P6v=%MJL=%MJL=%MJL=%MJL=%MIS'
        b'8+|JIRPw3hQ^}{2PbHs9K9zhb`B3tq<U`4ak`E;xN<Nf)DEUxwNN!4UQ<9sK+?3>|BsV3wDalP<A~!|3DauVzZi;eKl$)a56y>HU'
        b'H$}NA%1u#jswFqolAD6u6y&BLHwC#V$W1|R3UX5|xhcm@Ic~~vQ;wT*+?3;{95>~-DaTDYZpv{}j+=7al;frxH*d#%ygEzKOVLZw'
        b'OVMlSuP<d^%D$9+Df`k^Uy8mIeJT1<^rfwyN}fudN}fudN}fv_Bo#dsJrz9_Jrq3@Jrq3@Jrq3{Jrq3@Jrq3@eJc7?^r`4m(WjzM'
        b'?e9~`r;<-4pGrQId?@))@}cBI$%m2;B_B#Ylzfc;&Tr+;__2^LKP3BFXce{!rNUAnRR|TP3gf+Sl>8|9QSzhYN6C+pA0<CZew6$u'
        b'c`JD<c`JD<c`JD<c`JD<c`JD<`Bw6+<Xg$Nl5Zv7O1_nREBRLPt>mTTrR1gLrR1gLrR1gLrR0CU(0S%dVZ0aqe3|TP;i%9mY!ym{'
        b'r9!F@DohoSJe53^Je53^Je53^Je53^Je53^Jd`|?Jd`|?Jd`}{<R7iT)J;N#sRE);MW2d36@4oDRP?EhK9zhb`Bd_$<UhaD&)7c}'
        b'*7t9OZ-rK2t57N|6;g#zVY(3e$B#2V70zAwX`;jF>>t<bUjL~{_hVsw-cevZvA(cgSl?LhKd<NK`oa2x^`FO({aE<i1Ad-0`??-j'
        b'PpmJj7uGk{8|w$_&+B^JdG7;CAaEdXAaEdXAaEdXAaEdXAaG&>C-NrpCh{ioCh{ioCh{ioCh{)iUC6tTcOmaW-gW1_cdUiL3xO8`'
        b'F9a?GE(9(FE(9(FE^OdJ-a_6&-a_83SMWyQjldg$Hv(^M;f=%_i8m5&ByMcsM&L%^M&L%^#ujemZRBm_ZR9=3dyw}a??K*!ya#y?'
        b'@*d<p$a|3YLEZ;>ALM<I_d(tVc^~9`koQ5}2YG*<7xv?SJ{xF1Pf&Yb53DEF7uE~w8|#hrgZ1ZqeIoBf-if>uc_;Et<ekVnk#{2R'
        b'MBYH&K;A&!K;A&!K;A&!K;A&!K;A^&MBYT+MBYT+MBYT+MBYT+MBatG3wamvF63RvyO4Jw??T>%ybE~?dFxx=8w5X{IK>SRxDdDy'
        b'xDdDyxDdDycq8z}2HwcKk#{5SM&6CQ8+kYKZscv`ZRBm_ZRBm_ZRG8Z9jg(zk8Xg(2Z@h%>4V4zk<Zc{A@jjDK8Sp<jSmt(Nc><M'
        b'KY;)b5<l3;4+1|3{2=gyz|U*ofU*M03Mea}tbnot$_gkepsawh0?G;~E1;}^vI5EqC@Y|>fU@FN)<*{lfhPh70tW&I0tW&I0$<Gf'
        b'8X%B3u!RGG1A!BP6M++f6M++f6I(cuH<34yH<5QC??T>%ybF03@-F0E$h(ktA@4%oLf%5&Lf%5&Lf%5&Lf%5&Lf%5&jl3IqH}Y=e'
        b'-N?I<cO&mc-i^E)c^i2fc^i2fc^i2fc^i2fc^i2fc@Od)<UPoHkoO?(LEeMB2YC<j9^`$H_d(tVc^~9`koQ5}2YDaleUSHESP5Yz'
        b'gq09hLRbl5C4`j_Rzg?_VI_o>5LQB131KCKl@L}!SP5Yz`dA5IC4iLxRsvWFU?qT+09K-pmGD)<R|#Jwe3kH3-oCoQmjG7Y!1}lW'
        b'5+@QT5+@QT5+@QbBwk9qka%GeF9co)ybyRH@Iv5)z=go2z=gntz=gntz=gntz=gntz#D-#Uc(!CH}Y=e-N?I<cO&mU*f1Bez7`sZ'
        b'TbsBMxe>V$xe*!Ntc0@?&Pq5d;jC0MD*>$pv=Y!tKq~>Q1hf*+N<b?Ct-PW2HSmMLj}Fui5<f`%An}954-&sqYk^wprq<W1F5a!R'
        b'V66pfEm&*8TI;sf#{)XJ0TNFne#h1Vw$_cUuMcP<@kHW@#P8Z#(AI*s7PPgXtp#l@Xlp@R3)))H)`GScw6&nE1#K;Kvlg(m(9K$~'
        b')`GPbthHdR1#2x>Yr$Fz-K+&_El_KLS_{-#pw<Gl7O1sAtp#c=bh8$uwIHnpX)Q==L0SvaT9DR)v=*ebAgu*yEl6uYS_{%zkk*2<'
        b'7P?sr-K+&?Ep)RMn6<#H1!gTUYoVL9psWRDEhuY2Sqt5)1!OHCYXMmc$XY<w0<sp6wScSzWG!^F7L2vf%~~MV0<ji|wLq)|Vl5D>'
        b'K&(PHt01hpg>{3YK&%3>>L%959W`!nbOS`5h&&N_2L8Z%N2_jQecS+<-;q^1X;`<iJ_^Vj$Q;NV*vz5L9N5e_ZCHR-0a^uU6`)ms'
        b'RsmWCXceGUH?+P6OKjvs;6&imHclj7NW74EA@M@ug-yH=cp>mY;Ds%`koTh2*C(hDxDdDyxDdDyxDdDyxU_}qL1AMPZzSGGypeb#'
        b'@kZi}#2bmX5^rqdM&L%^M&L%^M&L%^M&L%^#y;K1dyw}a??K*!ya#!Y4$}vL4+0+qJ_!6E@Poh)0zU}+An=0?{2=dxybtoeduzj6'
        b'8{XRR)<#EbgIgQi+URI)XltXPwSlb-Y;9m`qoK87t&M)x2DLV*wb9So=x1#}YXe#v(At33Mn7xASsVSV4Q6dHYlB%E%-Ueq2D3Jp'
        b'wZW{7e%6MvHk7rYtc`xw2C_DgwSlY+WNjd816do$+CbI@vNn*lfvk;w)`qb*jJ08`eH-iR*Xs&_3xNxP3xNxPHv(@2-hlu+Rlp|R'
        b'NW77FBk@M!jl_+_jl_*j+}Om8yp6n#yp6n#yp6mEc@Od)<UQKJ2Z0X)9|S%Kd=U5`@Poh)0zU}+U;{tM`xKs_2Z0|1eh~N_Sq)@0'
        b'kkvp|16d7ZHIUUnRs&fLWHpf0H?r<0x}mIwvKq>2D665YhO!#UYACCrtcJ20%4#U9p{$0o8p>)YtD&rhvKq>2D665YhO!#UYACCr'
        b'tcJ20%4#U9p{$0o8p>)YtD&rhvKq>2D665YhO!#UYACCrtcJ20%4&478pvuOtAVTrvKq*0Agh6_Mjxxu$7&F(L97O`8pLW4t3j*='
        b'u^Pl`5UW9~2C*8%Y7nbItOl_f#A*<$L97O`8f~nGuo}W@2&*BihOipKY6z<#tcI`}!fFVs(Zy;2s{yPAuo}QR0M-Gp4uEw4tOH;j'
        b'0P6r)2f#W2)&Z~%x>yIkI`GwjuMT{5;Hv{)9r)_NR|md2@YR8@4t#ass{>yh`0Bt{=l0bN-veMB0P6r)M;F1LTUZ|jL{3CbL{3Cb'
        b'Y~n=XMB;_S3!8Xh6EEal$h(ktA@4%og}e)S3waB93waB93waB93waB93wd#_@W5CH#yT+8fw2yZbzrOmV;val=v?6evJQ}SfUE;#'
        b'9U$ufSqI2EK-K}W4v=+#tOH~nAnO2G2go`=)&a5(kad8p17sZ_>i}5?$T~pQ0kRH|b%3k`WE~*u09gmfdVs74$a;XR2grJWtOv+?'
        b'fUF0|dVs74$a;XR2grJWtOv+?fUF0|dVs74$a;XR2grJWtOv+?fUF0|dVs74$a;XR2TiO8#(H3^2gZ8P#Cp)gdLXO^!g?UA2f}*('
        b'EUfdgez5+$um3EpxAlqjz<OeRVZE@vv5ve8c^C2)@)q(I@)q(I@)q(I@)q(I@)q)L<lV@-k#{5SM&6CQ8+kYKZsgs_+sNC<+sNC<'
        b'+sNC<+sNC<+sNC<dyw}a??K*!ya#y?@*d<p$a|3YAn${`5Ar_9`ylUwybtm|$onAg^DXcFf#J`;Nc#N^e)>hypME0$$MyNS9#~JT'
        b'FRT~VH`W{L2kX!K`p;9E-qw+K{*w3WkDiT*z!QNd0?(~KfW-6Y1`i4`RG2C#b)*VQg;GJmqg7DyQ1VprRPx0BoeG``o(i7W#S<Gm'
        b'v5O~m@r7M{VHaQ8*bBS(!Y;nBi!TIT2)qz@{Sx@=<17R&1TJjjLf%5&Dm<Qz^~U<aI`Zy;^@;Ug^8WNora%9WKd&d&f64pPPrh91'
        b'g@3=X-ngHG_2+%P2i7On1M9!!U4J$1^@a7q`o?-={b2q1wtlYlGq66f{!8Ba^8BB~`oj7zd4IZ?^IG3n|J<MdvVQH)UmoZ4zWxlX'
        b'|C0Cl@;INs`n5lgb^QwY!uqv8$Nj728|#hrbFH8I*Uw+qe<sLZ)^lI$6YJOOcdctZv3}*PxYi5n8|zozeLwz#^(${*kADuVPpn^g'
        b'&y{N?))&?b>sQ|A$~znD2kX!KdJL@pk~hBox_w|hvA(cgSl?Jj-Z<~uC7$>FB(Xo|eBb}`{q;M)8SBe&X6AZ5?#*w)`f{9~`TqVp'
        b'0jvpNO#o{GSaZF<|IYRP{yW$E`_WwQ$LG1~^*A}#`|)b7em_1eY~OFb`nsQd^VLs3A$zS~`}4l;zpnlH-1l?s&*$17>hi|C@Bi!e'
        b'{kFcbetYA-|H~Wq{#noa`W{%H@Atps{qcYERo?GAbFcU3VPCJ`zVGLH{jPgzMSokL_v1Y8$Gwf?bL05j^ZhvAfq#$VbK`i`SKhe)'
        b'{eIuCy5FC-KJU+eWBZ<K``)>3UwxVO5#QGD$9b;T@4D~jI$qt|@f6lK)-O-q&%?PqS$D3>llSZN8Sjt(x$|C+{|T&L`}4hhpY{H{'
        b'J@x)PKiBJb-~Ya`e(-mn*T)%SV12$mKQZF<d5&@UdEWp2IzHFEUf<TweLt6<zdWAC`oa41?f$R#>-grYuYdQOuj)Qd#*EkfHs<^G'
        b'jk#Wrw=u8R@0a^|_f_1!x}UfC?&tXCt1nK7+gI0dHR9bH5%GE)3-~H-U!8lsUXLdc`*r`1==bM2uGjB#-_Nyu?_4?F_rsj{_t>BF'
        b'I$mAx7m@4rbr-j<zTE$Q-|qSD$;kK53z5&OPh!1$GS<X;yzf^lU*D%<t=IR>SoQio8*9J5FUNX+KZ<qUpSShAzE4Gs_kVx;>fR4;'
        b'U)|gH_SJoStM~WcsQ1rDQSYAzquxCkh1YMrfBua<u#VU7-M_lG`+ooY8~gq9Z+z)r-S@xW_mkc4&kuYR`~CA^^uYT2=fSvr_2u6u'
        b'*55z>M%R0;-tW&tWBc~OI=1f_ub)Ta%=hE=dH*~Z-+cA4zFt2E#VM?BtT*oeVEuW2{15Uz*q<Nl&v*ao%lg84VSQu0@puj%|9js2'
        b'=BqEyPmcN856o}A`trFx$9sP=$NT5~9Pj<u92@J6`#D&D@5kottNU?&^VQc}-+cA;dHCk5kM)K1!urN~WBuT9KJV+_eD$$D-=BvF'
        b'tYiDed%ryc+c)-WpC_ZSez5+2ej#&UeZD`Q+gD%K6YI#E^}fI5#(Lv^4%U%(A@A+0dwZ>T?Q12@59Y!t(fkIi-+cUqt-`l+!5@$H'
        b'sPLfhO<A9XZ|8!)K7Zew^_#C7;v2L+3rmGkVXN>h`PWy^H*I|u9u&TL>o;Ft(BHuISqK%rI}QGq!oAJ!9J^H5Dzpkmg$IRi^7_r!'
        b'x2kXS`Ygn?C*E&*-|+QWSXUwL;i9F&y|-U)#Q6<ipM|5sa}}=l`ut|F-+Vz^enZ%2Ayl}NU!KmhR45g`yw0y2=T6>tj-$eZ!gD9T'
        b'IGoR&{PJ@?p+c&#R45g;3g42)*IBbih3CDnzPao-Uq0BbZ!r5TgbH~TuAbrgX0y**ly0$AxRYO>SJyZy+{xEHXLqPDRk)L1O^h{D'
        b'g**B6v3t!@;ZA-v4%WPrpZifA74GC$uVKZ&En?~xp+a77kk=7z#nLTGx7aGQ?&_$Z<T+HBcuF!<NZpl^XQ`m%`5~Eag<CS$!E)s-'
        b'nSD=#z9n-VMb|gLeHKD@b#L?Q{bOB*`*N~tt$UkaU97b>o|4sg8LWes+S2R1296~)RR|Rl2bo&><9eOf#(q_eouLkPhI+6w?13$~'
        b'?`?j)-K`y8S^KSU>-9c_?_00;bG2{1-p|#(^?Gl?%WxmJIJ(7y!ezLxchysc&|Repck=6$+Uh&`z3=u`;ZFYCcXd>FP`C{DrNO_>'
        b')GY#A@FezK8E$<s+?PMT-#=!o)9;_7*1>1Bb#UG8dWH(`mvpX2!Jovd;JFWQmmk+tc%?lXhs39GNPP5m{)KSY=BO`(yY}f4;c5V{'
        b'HS!(EHP$=AYt;KEfExS#Q$`JZV5xE5Kl{{0xGx{PYL55Ma5d-qN6wn~kXjev>g(UY$Mc%?zU$W9DzxtEsPLc=0}nUw@w~pjJpV6+'
        b'Q~@8)>muCO?GgL^^<U8{;6rK!KBQLSLu%#3E}HTF;jR)N)#?l3t^+`2=@wgs)?FPH9^BO$xW&@zd<BjSE8o3RYrXTTR=tC))>ff$'
        b'yb|F`gsajKu9Un2yH6!{YPI32>iwOlYGc1DG$yK!Zh^)`?V*D9_8sqU&b1RSoW1ZG+-OGCMYyjAfQxV+g@c34MYylWs~#%o2zL?g'
        b'>oKu)Y;SGz*0H^HY;QE3tB(rLd;hu)AX8_m5Gte!ONCNltI*1ER8aB<?e==63ZX(`Lp@6cZSx21_Il86uji<My?u}I{`%Nssu1r7'
        b'vV9ru%WH67hWjX#3R{I%;iv$^?KxDKDufD3KG!=x_AC{)3Q9heeBwmIj-i6KxeT{2!+qVaF2j8k>U|g8u~mRxcN`U-_gB=;p#om#'
        b'J41zhKbCBL$8zgCmK)!(-1rK>##aD#zJCR8uYvW6_4ls;?3MWU3+sjZ*;sGf&%wV(-a_6&UVO!J<13aMU$NY!4P09VyoK#LD&Q?_'
        b'Z@h)=tzG-G=k~3TDl8Rn#(3|of?m`cZ()1mjPdTF0>HKfw%t0MwGL+&*ggtdg;qhyTgi`-Uts$vDEUbhK-GP~mVLGgtpcdJ53Y~c'
        b'0^9y_-t^A#<-F;af)3=54&;ws=Z{|J502yyj^xk#k*&wT`uoK&Jp${A^@aN{{QHgd#`?kiBkx4sIk6sCPpl&^e*dE9#(HD@U>&{W'
        b')~{c5V88AN6;g$z0vZne<qYYULaT68z$>~%v(BkPsE{h)Sk_s(#a5wpS4X!1N_#2!n)n+F9PfoDQ?I2$sjyXO{c*g1^+Gebb(LEe'
        b'8V+4Jpw;`;j9u?n=XSmSdO+9v)v8^M^@H{2ef_&%!}!bk`_;L<->+8fjlB3ZjK;5F^u_^g<7aJp1E#(8!ftKko+^Y2I;1t;s=IL@'
        b'@Aq$AcjJw#->=eb{3=GD_p7k`U^}1ptFZed))&^_zg5uqRzc%eG5TP?KJQmy_cO5meie4#-`cz%&y!exzbd((_ixqrv$2lt`y8x4'
        b'uj}XjRg5p|@4xbLMqvH@b$rIc`ukPb=NrW8>*MLy?Q+KZ_C4eMxOL|Hub!M6VqezZza?{KVtrw~u)eY0cs$5Ekr&@8I5DyQew;q>'
        b'zCWK>SbzUk!HJFa#`?kf^S+(~>+iqfb2700{w<1=3+wOi!zVY^-;Y-(v3(QUcVT~C*q`rj6?|Dw-2cLQVI6tDVe9Mj_RU%!>+kPJ'
        b'rv}#Fzcq3yu#V$Zy}RQS_UFR>T#fs|{#@9fH}>a^yc_#7oOL#i&l|_*y|La{N8XL?+sNC<+t|MN)v<$L9XtL0{(E|39ox6x-+xcz'
        b'{rBMg_q;a-&Izn1))&?b>l^Eh^@H{IM#y;v){*x?-UoRf<b9C$LEZ;>ALM=Bf3@ft1M3s(fpwgRe#TOvRKVB1p3y2C70|?g=1^g('
        b'5Gte!O9ixlo~h(h$>-5s;Zx!hN*+_U2o+L=rGk>H@%02+*^e4ukIqRxYJ5Fvd_8J>J!*VCxm3W3n<uvlt-?_O?VsmM<LgUdst_uq'
        b'3QGkgUrN4|d?~paUr#A{DS6?<&7-e(KDw^#snT6-6<UR(0!~>yd#Er~Q1YEBEESY|EBRJ(ou+(rn)1<U%BPiFrzsy@SN61$x035L'
        b'<<m;8)0B@+Q$7b*)IDda5GvrB;pZ&fqI8R`LaT68cu;tT3iG|Y{yd>uDEarNDgRPX@<-eJX%&tNuh*6Rbd~Vy-X8aF^?w#Zh1Vlo'
        b'+&|a<Stu2@3a{_yasOohXW>;casO!lXJM)kD!dvl?%(bIER+hbr>5in<^IpYQQ<-1{p4lnr~Bjn>HdrDasP7v$NIv0=?`q&ps}vd'
        b'q0p(&sn8iJqzc;Hsm+~Rg~q;?N4L<9zO<t+?dVH8`bxaY)>5HV*ed8HzVs4b4+=F@m^i+?|2jw9KjZ&-S4v(=UP@j{UP@j{zLk7y'
        b'n{RFNohmF9N(JrhyH(KMz8@4?uk+UHy!AS7z0TiHW5)gC{$CGQjl)~x@YV<GZ*8Zzf876LeZJq%!SVdH%@+5M`+uw#);HFX_qELy'
        b'_mBI3{rAtnI`Tfq`{4cS!TZ;P?fYQ+K6=xB^#1kW{cDb)!c-yNUm|m)Zn1QWQemsmDjXH?ik|Pywzz-l|FfXv6UVaa*Zw~XN<O!4'
        b'q2yD^r;>+~hmwbqN2;K09!ehC=7GI^N^4VEo6_2p)@HuH#IN7~|5#sGFRX8@H`Wi<-w)w)4XjVB2i6no3+sjTjrGR*!8-ER`yqVR'
        b'#5%TbVfz-gZ(;iuwr^qk7PfD_zhzy&|NpV$Ztc-KRM7EzFBM9Kt%6?5Td!q_?3x7dS<q3ebrfqI#ac(P)={i=6l*1K9lu-2kCGoH'
        b'KT3X-{AimWZS$jTew6%Zn;#v&AMNdr_V!16JB&6T1qae5kTxH@!XLcCpZ8aIjDhuu^}u>!ePR9m2p(hO-#69|){%E2?|gs9x+VmC'
        b'{Cnh`$U6(`$g9p-OaxZr?3xkq^#<yjMT82e!cw7B*eWQuI2Phqh+`p+g_>uf=2@hYr;?|Vr;@9Ac1;cVxW!WPrQ}P=my$0fUrN4|'
        b'eBrgdw8<A<+~8S&X91oCcoyJUfM)@o1$Y+VS%7EP_<)af<lV@--}!xw5BONe!FzA4BkxAu{m$=ee89&#@^;|gWBWGpqC*zwkOewq'
        b'fezU<KH%&6!S+RmEYKml#s_??V}Cx#dyp3$vTJ<6$2#%~U?G5o02YsqTM}42I&M8WZmCfgYLtZ<WvND4s!^6BR7e$;3Z=qUp;dU_'
        b';@3cdj~h${p9&6<B}A4GSwdt9k>zZxBl3LT$rB#CW(s_)BQH*4CQf5sGX*}@-w)b}vzXURfsgg~gZ4F3;OoCn<V8;`(GyGb#1cKR'
        b'L{BWy6HD~OGO>LZ@-F0E$h(ktA+Ow(3xStj#0!a+Uc{vrap^@|I(n6kUZtZ~DR=4URSI5^cIuR6>XhX@P~gis+1#N*s<8AoN(Hp}'
        b'Q>QFb4YF&Xz}J1D->#95dz`@6zx?{0fVhVVd=1?T>xK1=^~U<a`t$xc&v?&u=KJ$--szd3Sb}1yhS<jdf%`9O8z1f62d%)b0|dSl'
        b'rV4spqal{hJGt|z_viK5Snv1k^PKnX`8==Ne~p3liS@vGVtu_np7lD>Dejp8Uvga|13uQz>+`e5`@8d+1MBbaRtx&NhXs7x(ZUT%'
        b'H`usAWBnij636?I>lzdA^%;%8dSV@0IIx9ZzkM0^fPk-iP#mnkAGy}+w=d%!4)C##yotQI-k;AbtZ%HppQBre?F)Y`_-o;u<ia`0'
        b'g>#Y%=Oouc-i5rYv3}m4=k>g=*T6cqZw1y9>kIdf{kd?&Djcy2dFy$9el}jedtx2i7fr8)rq@E#YoY12(DYhpdMz})7MfnykN$tG'
        b'BX1)wnqI51KR5R0#`bNzBQ>_~!S;o;u3!59SWm1YFD?sPxGZcP9Iw#yS_j(~mxV1{7PfF%*n+SYE(=>Y<G66fap8>P!WqZ)yyvYk'
        b'us*RKSWm1ktQXcd*5C6MT3m%Qj)gOh1!3Jk=Ku8pp}}?kmjA~MN`<WgK0B1I*DsxREM2c(I`3FI?^rtTSUT@m=yjD&JC;s67P?*6'
        b'FZq96N4Klc?JAveEVR2yXB<ms97|^$OJ^MKU-ADaXlv_?W9f|J{VV=ocQu`Hynn_2;})gDRzdsuIx65Za;f)K>V1`ZU!~qxsrOat'
        b'eU*A&rQX;5EB+q^ZF5~5R=POs>#z9VPogdk`}!;XUv8mouFLgHy|2>c`lXA*N*9NfE)FYQ99FtGtaNeM{VV=opDSG)R;}c^IPCrv'
        b'|F3`JXqz8x^P|087l)ND4y&VWu8YG;y|7X*tkerD^}<TMuzIx3A0>aZ%^xLyw9OwSf0X<|^7~i(KMGTYP$5-VDwGOa1svhFF5=(1'
        b'h=1!M{;iAnw=Uw}`nKiPw=K6W;@?xrrMIn%__r?N-@1r@>mvRg_$t=cMf_VA@o!zkzjYD+4kZsI4<%0}PbE)n^Hg$u<#OvQms^J0'
        b'GTfHowhXsrxGlqNT{X64xUH+kw$5H|oW0z*Xl(1Ev5kwyHZB_5xM*zSqOpyO#x^b*+qh_K<D#*Ri^euC8r!&NY~!M_jf=)ME*iUj'
        b'$p2$qn^@QLZ(Yy7bv^&q>B_yebL$(HTi>wU`iAA!H!QclVYx-N`*-|53QFEeuIu@?$hJkcEwb&S<VOcIk!|br<<|B5Ti5e%UC+OD'
        b'IoTH3woYGek!|a8vaQR>wk{{zx}0q5a<Z+<$y%3_wJs-XT~5}zoUCzWf9q<p#-;s@Mq8uN*1DFg(P?XR+8XCF8;!Qcsm#WeWQ{YK'
        b'jWd~zE6EyHlC@fFjZ4WImy$IuC2L$t)@ZLaeh{N^DOsbr)@ZJ^uIO)F(cfyXwXW!I_19YcwU*6VHf!0eWwVyeS~hFhtYx#7&002V'
        b'UD4mVw5+AG*0p7={#wIlt@c{0z1C{4wc2Z~_FAjG)@rXcnATuggJ}(>HJEn&nE%H*UfUb5ZJg$8oaSs?P<H*2|HnG=HuB=y{_9u#'
        b'Kh}{K*Y-EA?QdM$-?+BFaczI&+WyA1{f!IC8W)r`uI+DJ+uyjhzj2zgaczI&G-u;9XX7+y>%#t)#2P0$TM%nOtZ}Nd<**}%9XagC'
        b'VFzEZJi4m?$YDoU^&fT4t{?ROEGW1x>p!sA(OJ)<3(Jm9dmeD?fMW+7J38libk6e#WJl*b4@7n#vICJFi0nXQ2O>KV*@4IoPIn%h'
        b'?mW1-|KM!r!NvUt7xy1r+<$O!|G~xmM-V%L*b&5zAa(??BZ&Qfb=^sF+(-}vz%NZzX65q#kIfZoqhCRx$q*=`Gk^|_AU1;72x23M'
        b'4QjGMO*UZIXXO9i*GEnE8TtSD^AE=h#~a57$NzpzW#F&R$p6ou$9FdTHSpKKUju&){57h`MitqpA{$j?ql#>RumQpb2pb@5fUp6='
        b'1_&D<?C+)ji>3dIrT>ei|BI#ni>3ejUHbp|I*}I$8z5|eumQpb&zcRMH5(9Y@T}S3S+fDa2J04!b&JKi#bVuJ@vPaR16w?6wy3~9'
        b'EB`-#esCOl(SR-1Efx*f9^^#>wjkGnTzilg4cMXqTfCQYQGhMDwcyr*TMKS2-b?ve^#3`I>x=hNe&+mtj^p~q#c^EUNF2xab09CC'
        b'<zGC@zt|UB?29d)<zKv)a`9fu#d|3~`~5%1@%@aOk&AayE|9fA)&g0Jdy${r{(t{|a4&L!ti`>^#l6S{vKGkN!u18R7SHl8p5<R4'
        b'Yk{l<vKGi%lw6CFYf*A7O0Gr8weZ!#R|{V)e6{e^!dDAlEhb(UCD)?lT9jOil50_NElRG1uNJ;q_-f&+g|8OATKH<=tA(!?zFPQd'
        b';j8`ASMeR<|2cke{Ni}vc;fiO@xt-O@xgKAJ;-~I_aN^<-h;dcc@Od)<UPoHkoO|*Mc#|N7kMx8UgZ7z`TfA;etakQe~u$?AaEdX'
        b'AaEdXAaEdX-~tEo2J!~-Ch{ioCh{ioCh{ioCi4E>W5su0|K|h*ehB>c)B5op*Z(;Ii60U_Bz{QzaD^WNKLjoWE?nV4-a_6&-a_6&'
        b'-a_6&-a_6+-bUU=-bUU=-bUU=-bUU=-bUU*-a+0$-a+0$-a+0$-a+0$-a+0)-bLO;-bLO;-bLO;-bLO;UJ&bmSO>&9Al3n~4v2L?'
        b'tOH^l5bJ<g2gEuc)&a2&h;=}$17aNz>ws7X#5y3>0kIB<bwI2GVjU3cfLI5_Iv05%tOH>k2<t#t2f{iK)`74NgmoaS17RHq>p)ls'
        b'!a5Mvfv^sQbs($*VI2tTKv*Y{7r;6I)&a2op2!bO<Oe450~7gyiTuDseqbU$Fp(da$dB(t{?Bpbg|7~Lb>OQ5Umf`Bz*h&pI`Gwj'
        b'uMT{5;Hv{)9r)_NSEunS;Q_D?fOP<@17IBh>i}2>z&Zfd0k95$bpR~v;Kz3d|KAfJtOH>k2<t#t2f{iK)`74NgmqSdL97d6T@dSn'
        b'SQo^)AlCi+39Y~geqaPYFoGW#!4Hh!$9Dw(=QslY{cKizSMYy7@bF4V3}sy?>q1!<%EAbKd`IwqPQWz=vo4r*!K@2rT`=o{Sr^Q@'
        b'VAch*E|`Vk`}hvu|D1rpaMp#hE}V7YtP5vdIP1b$7tXqH)`hb!oOR)>3uj$8>%v(V&bn~cg|jZ4b>XZFXI(h!!dVy2x^UKovo4%<'
        b';j9a1T{!E)Sr^W_aMp#hE}V7YtP5vdIP1b$7d_Smvo4r*!K@2rT`=o{Sr^Q@VAch*E|_(}tcxD&LRlBex=_}IvM!W$p{xsKT`229'
        b'Sr<Ll1+oIj3Lq<htN^kC$O`mW0b>P>6);x7SOH@Nj1@3e$XJ0QD?qFOu>!;j5Gz2e0I>o^RzO$*VFiR05LQ4~0bvD%6%bZHSOH-L'
        b'gcT50pvVdUD*&tjumZpe04o5j0I&kU3IHn-q6`QtAgq9}0>TOiD<G_ZumZvg2rD40fUp9>3J5D8tbnir!U_m0Agq9}0>TOiD<G_Z'
        b'umZvg2rD40fUp9>3J5D8tZ4k&A0SqMSOH=Mh!r4K41Ng?7%O0`fUyF`3K%P<^#vg@kQG2ytP(?60c8c06;M_{Spj7Qloe1`LRkrA'
        b'C6tv=Rzg_`WhIoA|KSQJn3aFO(=fgR_Wzy$XC<7Ka#q4w31=mom2g(VSqWz)oRx4^!dVGtC7hLTR>D~cXC<7Ka8|-u31=mom2g(V'
        b'SqWz)oRx4^!dVGtC7hLTR>D~cXC;cP1hW#%N-!(ItOT<X%t|mT!K?(c63j|4E5WP;vl7foFe|~V1hW#%N-!(ItOT<X%t|mT!K?(c'
        b'63j|4EB}5|VSESde@;MPG+7B|C76|9R<?d5!C8qaD@Tdptc0@?&Pq5d(Pbr=m0(taS&1$yp{#_m63R*_E1|4}vJ%QlDC<F456XH_'
        b')`PMhl=Yyj2W35USr5p1=&~M+^<b<AV?8{R^>n-N!B`K*dN9_5u^x=|V5|pYJs9i3SP#Z}FxG>y9*p%aZh=1_>j7C0$a+B51F{~F'
        b'^?<AgWMQ&CzLWJoCm=AC^`NW=Wnry8zH9YACm=DH^}wtLW<4<LfmsjCdSKQAv#?Pg+Nh83M*Yt#A@S2UHKg?*tp{m6Nb5mb57K&&'
        b')`PSjr1c=J2WdS>>p@x%h1LVK9-#FAtp{j5K<fco4}BKK=;J#^|8oKY!&wi`dT`c*vmTuF;H(E{Jvi&ZSr5*7aMpvf9-Q^ytOsX3'
        b'IP1Y#1!onURd80pSp{bmoK<jE!C3`o6`WOYR>4^XXBC`PsIm&oDln_StOBzN%qlRez^nqZ3d|}ntH7)RvkJ^AR9OXO6_iy_RzX<>'
        b'WfhcFP*y=%g(|CntOBwM$SNSKfUH85RWMe;SOsGhj8!mJ!B_=j6^vCdR>4?>DyvXs6@*m~R-wu&0ILA30<b<)^8fz8E(oh2tb(u#'
        b'!YT->AS}$t$9G2l-xENr0<j9j!i0Q$C**%lKw=oHV61|%3SCx#Sfwrt+wt+;j{kWjBnGky$SNQUv+?nrjsN!q-895%e0*2qe_jcZ'
        b'_vwX7A3$hGt01k0v>MWCNUI^OhO`>eYDlXgt%kH3(rQTSGZp{uUvz_74QjP2t%kK4)@oR*VXcO>8rEt(0pGw@16vJjHL%sdR^Qea'
        b'gv8KRLt71PHMG^xRzq7DijVJ5{LcwW3~x2O)$mrsTMchDyw&hl!&?n+HN4gER>NBjZ#BHt@K(cH4R1BP)$mrsTMchDyw&hl!&{9)'
        b'tHG@Xw;J4PaI3+s2Dci8Rzq72Z8fyj&{m_+YGA8@tp>Il*lJ*_fvpC%8huuy&uUPsL9GV08q{h~tI=mQq}AxN8huv7Sq*13`m9Eu'
        b')lgPLSq)_klr>P+Kv@H24U{!d)<9VUWet=yP}V?MgFb73tO2q<6Y&51+#DEdV61_$2F4l~YhbKFpEV%XfLH@!4Tv=$)}YTC2x}m$'
        b'fv^U`8VGA3tbwowebxY2gFb8EtAVcuzQXo<e7E0!PC#G)YXGbPum->y0BZoO0k8(Z8USkmtkF%w(M`hvu?EB%5NklJ0kH<e8W3wh'
        b'tO2nG#2OH53coW9j5VsX2FMy9Yk;f)vPPBGKv@H24U{!d)<9VUWet=yP}V?M17!`AHBi<-Sp#Jalr>P+Kv@H24U{!d)<9VUWet=y'
        b'P}V?M3uP^owNTbVSqo(?l(kUSLRkxCEtIuT)<RheWi6DoxNEpj)<RheWi6DoP}V|O3uP^owNTbVSqo(?l(kUSLRkxCEtIuT)<Rhe'
        b'Wi6DoP}V|O3uP^owNTbVS&J5Hfvg3x7RXv4Yk{l<vKGi%AZvlF1+o^%S|DrDVl7&%1+f;yS`ceNtOc<a#99z*L97L_7Q|W*YtdpY'
        b'gtcg~7Qk8nYXPhUuof-W!dDAlEqt}`)xuY+$Fdf{S^#SStOc+Zz*+!n0jveE7Qk8nYXPhUuol2t0BZrP1+W&tS^(?s)%V2edt&uH'
        b'vHG6h)%Wi=N@DgsG5emFeNW83CuZLhv+s%7_r&abV)i{T`<|G6&+qK}&tK3%;Df*if&c!W3=_leiQ)Ie@OxtTJ-@^6zn>VF_#*K|'
        b';)}!=mpBkO5I7Jx5IAs&19|`cPo)#f?}_F2#PWM$`8~1xo>+cQEWam~-xJI4iRJhFF2DaAN8Uu<hrADYAM!rreaQQe_aW~?-iN$T'
        b'|AY&H3xNxP3xNxP3xP}D(S^i?#Dy!|2;2zV2;2zV2;2zVxWbLRjl7M#gS>;hgS>;hgS>;hgS>;hgS>;hi@b}xi@b}xi@b}xi@b}x'
        b'i@adg0kaO6b-=8@m){f1?}_F2#PWM$`8~1xo>+cQEWam~-xJI4iRJgi@_S<WJ+b_rSbk3|zbBU86U*<3<@dz$dt&)LvHYG`eorjF'
        b'Czjt6%kPQh_r&shV);F>{GM2TPb|MDmfsW0?}_F2#PWM$`8~1xp5Nv7e}51j5bJ<g2gEuc)&a2&h;=}$17aNz>ws7X#5y3>0kO`*'
        b'uSo~SIxyCOvHsqEPi(&@w%-%m?}_dA#P)lBx8HvP7XpJ=2gEuc)&a2&h;=}$17aNz>ws7X#5y3>0kIB<bwI2GVjU3cfLI5_Iv~~o'
        b'u?~oJK&%5|9T4k)SO>&9Al3n~4v2L?tOH^l5bJ<gpXK*|{~=u%>ofiSdqUuZ#PO#WDxA<bJ~)oZP}YUAE|hhltP5pbDC<I57s|R&'
        b')`hYzly#x33uRp>>q1$d;rE{l3}#(0>w;Mq%(`IK1+y-gb-}C)W?eArf>{^Lx?t7?vo4r*#jFcvT`=o{Sr^Q@VAch*E|_(}tP5sc'
        b'FzbR@7tFd~)<u(bp{xsKT`229Sr^K>P}YUAE|hhltcxP+0$CTxx<J+ivM!KyfvgK;T_EcMSr^E<K-LAaE|7JBtP5maAnO8I7s$Fm'
        b')&;UIkadBq3uIj&>jGI9$htt*1+p%Xb%Cr4WL+TZ0$CTx3iMb3V+DGw0I>qZ3iMb3VFiR05LQ4~0bvD%6%bZHSOH-LgcT50Kv)4`'
        b'1%wq4RzO$*VFiR05LTea3IHnrtN^eAzzP5>0IUG80>FyEz0ruL4<Ip!6(ClCSfA<lpHGaNg#ls(h!r4KfLH-y1&H;Te*X=C#4uLC'
        b'SOH^ww%>m~@zX_yvI5EqC@Y|>fU*M03Mea}tbnpU>+e5-Rc3{57zUvA8G!#i0huAKfV2YA3P>wjnL(`pwF1=oEWrPM0^Ai2U@L&F'
        b'0JZ|y3ScXMtpK(H*a~1PfUN+w0@#X0!4uF{KwANA1+*2=RzO<;Z6&mo&{jfQ32h~`mC#l~TM2C?w3X0SLR$%KCA5{$Rzh0|Z6&mo'
        b'&{jfQ32h~`mC#l~TM2C?w3X0SLR$%KCA5{$Rzh0|Z6&mo&{m?(O4L~iYbC6euvWrai8?Dmtpv3a)Jjk*L9GO}64XkxSqW(+q?M3X'
        b'LRyJ7E74{poRx4^!dVGtC7hLTR>D~cXC<7KXtNS+R-(;HAS=;kC5)9YR>D{bV<n80XtNT;N)RhStVElYXtNT)N&qVXtOT$Uz)Aos'
        b'0jvbD62M9TD*>zouo7)n!dD4jC480eRl-*ZUnP9?;Hw8;J^1RuR}a2=x?%VkfB*eJ<CO--kr={y2RB$g^Y6bOK<1M=;o*eB@zx6='
        b'GnDn9tOsR1DC<F456XJFYxuydca<2<dT`c*vmTuF;H(E{Jvi&ZSx*nZKS1lV0RMXe5<^-K()vum|9k>a>w#LI3HaX^!bOI)9<23X'
        b'tp{s8SnI)B57v6H)`PVkto2~62Wvf8>%m$N)_SnkgS8&4^<b?BYdu)&!CDX2da%}mwH~bXV66vhJy`3(S`XHGu-1dM9<23Xtp{s8'
        b'SnI)B57v6H)`PVkto2~62Wvf8>%m$N)_SnkgS8&4^<b?BYdu)2V6B3+3f3xEt6;4{msRMp3eqY_t01j{v<h8Tq01^btI%Z?m{nj_'
        b'q01^LtDvlcvI@#7D662Xg0c!-RsmUsE~{Xyg0Tw5Dj2I^tb(x$#wr-A&}9{fRp_z`!YT->Agn@{RRC52ScNXD;H!eK3cf1%s^F`F'
        b'uL`~@_^RNmg0DXN@Bc!x0IUMA3cxA=s{pJ5unNE`0ILA30<a3eDgdhhtOBqKz$yT%0IUMA3c&iTzyDld2&*8h(*42$u?oa05DW9~'
        b'`JI3N{QxopSp{SjkX1ld0a*oP6_E8AfBy*vWfhcFP*y=%1!aA<-~auB8q8`itHG=Wvp(bRzn}QwgwhF(69&iszQzq`HK5giRs&iM'
        b'Xtf3%8q#V=t0Aq1v>MWCNUI^OhO}CH4h?EGsMYGT8rEu9t6{B%wHnrHSgT>JhP4{jYFMjbt%kK4)@oR*VXcO>8rEu9t6{B%wHnrH'
        b'SgT>JhP4{jYFMjbt%kK4T~>oy4Qe&0)u2{`S`BJ7sMVlWgIW!0HK^5~R)bm%YBi|UpjLxg4Qe&0)#$Ps(rQSn(PcHD)qqw5T8%EN'
        b'(PcH5)nHbGSq)}2nAKobgINt`HM*>ZvKq>2D665YhO!!6)&N-pWDSrtK-Qqk8W?NPWetcmAl86b17Zz`H6Yf2SOa1Wx~ze)2ErN$'
        b'YapzFum-{!2x}m$L6<cE)&N)|U=4sZ0M=*x{pW=stbwow!Wsx`AgqC~2ErN$YapzFum-{!-7g#vYe1|4u?EB%5NklJ0kJ;o?>_-x'
        b'tbwrx#u^xFV61_$2F4l~YqXg)K-Opd{r3bUhO!3A8YpX^tbwuy${Hx^GyeV)td$tf8aQk84Ez9E185DPHGtNr(i%u>AgzJ42GSZe'
        b'S_5hgs5PM0fLa4;4X8Ds)___AY7MA0pw@s|18NPZHK5jlS_^6|sI{Qhf?5k|EvU7i)`D6KYAvX>pw@y~3u-N>wV>95S_^6|sI{Qh'
        b'f?5k|EvU7i)`D6KYAvX>xNW$Q)<Rl~Dr*6)1+*5>T0m<7tp&6e&{{xi0j&kJ7SLK$Sqo<^oV9S)!dVMvEu6J*)}qQ<Fl)iA1+x~+'
        b'S}<$DtOc_c%vw}g3uP^;tOc?b$XXz4fvg3x7RXv4Yk{mqm9?m{7Q|W*Yf)t_gtZXXLRbr7Evl>quol2t0BZrPMU}Pi)xuW`UoCvK'
        b'@YTXs3tugKweZ!#R|{V)e6{e^!dHLqzyA-v-e>F'
    ),
    'DS0004.CSV': (
        b'c-n;B-_Jb9l^^DN1O6X$Z*fs|epFRoCRrN}U?WC$<6P5TW<x+LEwqXg=dTY*t;Op3zK_ETL7d{uneLwVnbR}pGyQM=@W+4n{!hQ1'
        b'e>MKc-~P*g@$Y~5hd=!Bzy9Xm{qXz0|I`2Z+c|#v=cil$=imS7AOHHtfBMt+zyHra{PyP?{`QYQ{{7$o@W<c$4}bS}fBfN}{^_^R'
        b'Pybvu|GOXl%MZW%?f9#o?)|U-)i00mH-GrQ{`k{<|EGT&KmE&p{Nay(`td)1|GVG(Z@>TXPyh7W|M<;KzxnGQ|I06RK5qWk|IeR('
        b'`2FAgvO3pK%m42C|MXLx-~9Cd{=*;t@SlG8&ENjlfBf_F`}4p4uiyXa$3OhMWxqU)c|YFY{PaIR{I;)W_?M;s@P~i;@u&O!{`LI-'
        b'?H~U5<Ny1I-~Z_^Px^1(H~#tY;+K1T_e-g3@&E4;|NS5S?uXz0uYdU6KmWgf`7eIj*Z=st?|(WB-~ZE}e)HFV_|sqZ^M5-a|NHm<'
        b'<%j>{rvvxxfAP!C{_*$!`CtF-KR@gbKNa}fAOGR!hy9!H|KT5h_v7#X{x|>Ohd=)Khrj#F)2;W@`CoqbcmMtE`T0Nia-06mkAM7!'
        b'|NHwt{_xxXIR7etzSA#H;je%A)Av9A?ta#P{r#W5|LNbq`{mI5^v@4J-~HzY`r*%y^WF1T&vSlW___Ay!qP%%A+#{Ca8&%L_)+nr'
        b';zz~zUy6VC^8xtb&-?$E6;<D=zEypz`qHK^6<;d8RD7v;t9Yw;t9Yw;t9Yq+sd%Y)sd%Y)s(7k+s(7k+s(7e)sCcM&sCcOORPm|e'
        b'Q^lu>PZb|3K2&_D_)zijm*Vd~=Og0Iuf%=cqjZnZ0=E2!xFh0@h&v+gh`6&=T*4g*cO=}Aa7V%&33nvik#I-C9SL_N+>vlc!W{{B'
        b'B;1j3N5UNmcO=}AaOYRyKHl$+h&v+g{7T&CW68K9<Bp6wGVaK@Gxe5!gxnEwN5~x^cZA#--1AcI{HokHcR~66Xydn9x&5ni-`pj1'
        b'7ajS1SYOm_QMXU<v9$TJZp*qY>$a@hvTn<|E$g<d+p=!Ux-ILrtlP3~%epP=wyfK-Zp*qY>$a@hvTm!J+oEoZx~*<*OS&!TcJR+l'
        b')NN6>)y{26w<X<{bX(GGNw?L`Z9%uy&TToj)Xpt2x75xpDYw+lEwyv&$oM7XmioCR<CctDGH%JZCF7QiTQY7fZn}_LLT(AUCFGWn'
        b'TS9ILxh3S5kXu4-3ArWYmXKRQZV9<1<d%?ILT(AUCFGXcxh3P4j9YBymXKRQZV9<1<d%?ILT(AUCFGWnYa!P{u7z9+xfXIQ<XVkf'
        b'%ea<tE#q3owTx>S*D|hUT+6stAJ-zTMO=%x7I7`&TEw-8YZ2EXu0>pnxE66O;#$PDh-(qoBCbVTi?|kXE#g|lwTNrAaXs~>+9Iw+'
        b'T#L9Cam_Q<mT@iPTE>-(D;ZZZu4G)vxRP-t<Lan;FCkY#u7q3(xe{_E<VwhukSifqLavt1(@V;glq)G$Qm&+2Nx718CFM%Wm6R(f'
        b'S5mH|TuHf-awX+T%9WHWDOXaiq+Ch4QXf}Bu7q3(xe{`vKCWb3$+(hnDdSScrHo4%mohG8T*|mqAD6!p_x?Li5tkw^MO=!w6mcoy'
        b'QpBZ*OA(jq<5I$<gv-{uYKpiNaVg?b#AUI%j7u4prB8uV$fb}=A(uifg<J}`6mlu#Qplx{OCgs+E~h$*lyWKMQp%;2%faemF2!8%'
        b'yN8f-A?HHQ1;2U-K^KB91YPi(hmdq3=|a+lqzg$Gk}f1&NV<@8A?ZTWg`^8g7m_X{T}Zl+bfIQ01YHQaP%{^DF63OuxsY=q=R(ef'
        b'oC`S@axUat$hnYnp=K_`T!^_4b0OwJ&0I*ikaAPa+!S(C$W0+P)yz#9H)Y(EaZ~53Q^ZXXH@E)K=Ps@US5$xL{blmYhbiZ#oSSlP'
        b'%DJg?)+y$un44m5in%G~W^vccxhdzSoSSlPCaVj&Dd?u4n}Ti%x+&<Upqqkj^0T=q>87NcQ%%hjbyL($Q8z{16m>(?4N*5l-4Jy{'
        b')D2NLMBNZ|L(~mXH$>e~H#gMH4Rv!v&JA^QL*3kvazn}uDL16tka9!H4JkLI+>mlZ$_+JhL&yywH-y{}azoABka0uC4H-9N+>miY'
        b'#tj)aWZa-}L&gmmH`L7y5jRBK5OIS8bK`5{j-n4o(TAhx!?~jmuj)tHhokJnxw8+Sk9Ac3Yvqo@59ba)d@g9)w~B8S-$NU+RDE&V'
        b'm&z}dU)uIo_13nxinoflikFHPH@#H7RJ~L^RXw%osp6^Psp6sHq2i(9q2i(9Q^lu>PZggkK2?0E_)zho;zPy7+!1p}%pEa@_`^~B'
        b';VAxa6n{91KODs$j^YnT@rR@M!%_U<DE@F1e>jRi9K|1w;txmhhoktzQT*X3{%{n3IEp_U#UGC14@dEbqxi#7{NX76a1?(ycl_aV'
        b'LB&&V>PN&K5qG5@J|9cQ9T|6I+>vo-vbvBvLhcB;Bjk>dJ3{UVxg+GZklR9T3%M=iwvgLGZVS0B<hGF8LT(GWE#$V4+d^&&xh>?j'
        b'klR9T3%M=iwvgLGZVS0B<hGF8LT(GWE#$V4+d^&&xh>?jklSkGwv5{{ZmW&kB5sSgE#kI_+iK&sgxeBsOSmoJwuIXfZcDf=;g*D3'
        b'5^hPjCE=EYTM}+bxFzA1gj*7BNw_89mV{dpZb`T$;g*D35^hPjEA;U3_Pj*g5^-1R;hV=&c^S84+>&ui#w{7QWZaT*OU5l3w`AOs'
        b'aaZi&b1w?HCFGWnTS9ILxh3S5kXu4-3ArWYmXKRQZV9<1<XXtJkZU2=Lav2e3%M3@E#z9rwUBEe*Fvs^Tno7taxLUq$hDAbA=hf-'
        b'TE?}EYqfDL;#$PDh-(qoBCbVTi?|kXE#g|lwTNpG*J|Th!nK5J3D**?C0t9mmT)cMTEex2YYEp9t|eScxRP)s;Yz}lgewVG60Rg%'
        b'Nw|`5CE-fKm4qt^R}!uyTuHc+a3$eN!j*(830D%XBwR_jl5i#AO2U<dD+yN;t|VMZxRP)s;Yz}lgewVG60Rg%Nw|`5CE-fKm4qt^'
        b'R}!uyTuHc+a3$eV!li^u36~NsC0t6llyE8GQo^N#O9__}E+t$_xRh`y;Znk-gi8sR5-ufNO1PA8DdAGWrG!ffml7@|TuQhr@9=S^'
        b'm?ADkT=ENtOnpL-JX1}6<&a`7#oQHo_<XF%`hrfehojiTQS9L;_HYz?ICt#fb3w%=T}Zl+bRp?N(uJf8Nf(kXBwa|lkaQvGLehn#'
        b'3rQD}E|y+$A?iZZg{TWr7osjiU5L66bs_3P)P<-EQ5T{vL|v$v3rQD}F4WA0nz>Lj7i#80%7v5*HFKe6E@WJ&nF|pYA}&N+h`11O'
        b'A>u;BO%XRm+!S$B#7z-5Mcfo|Q^ZXXH$~hOaZ|)8@o?_M!{=R8UESOiaZ|)i5jREL6me6;O%XRm+!S$B#7z-5Mcfo|Q^ZYnb5q7m'
        b'88>CzlyOtWO&K?XFF7GMh1?W!Q^-vrH-+33a#P4nAvcBG6mnC@4Iwv#+z@g@$PFPkgxnBvL&yywH-y{}azn@s^>IVS4H-9N+>miY'
        b'#tj)aWZaN(L&gmmH)Pz9aYM!p88>9yka0uC4YhGY#Eq0PL&gmmH)Pz9aYKFF5OG7q4G}j)+z@d?#0?QQMBET@V{pIAID{Rx!Vdcm'
        b'JG_c)r5&~|^sp6n*mv0Bv&h%V?K|x7&0SRfsQlK(Z*Kh7#&1>Ms=m1KOB=sbeX06V_13nxinoe)XhT}ni`!l*Un*bP_EPoKwx^1x'
        b'il>UFiie7aiie7aiie6%6`v|TReY-WRPmwWL&b-R4;2@5N6;beu<y9T`z3nh+>vvLJ8Z=rw&D(3afhwA!&cm3EAFrrci4(MY{eb6'
        b';tpGJhpo87R@`AL?ywbi*or%B#T~Zd4qI`Dt+>Nh++i#3uoZXMiaTt@9k${QTXBc2i#=>!>|rbJuywJAeWxA1c`TKeaYx1}?ywbi'
        b'*or%B#T~Zd4qI`Dt+>PH#U8fu4qJJLt-Qll-eD{6u$6b%$~$c39k%igTX~1Ayu()BVJq*j@4Ul%?`|o#rQDWsTgq)Ix24>ca$Cx6'
        b'DYvEEmU3IlZ7H{<+?H}%%55pPrQDWsTgq)Ix7EjOA-9FxRv))z+?H`$#%=X+Tf}V<w?*6*aa+V~5x3RHZ3(v|+>&rh!Yv86B;1m4'
        b'OTsM)w<O$>a7)51o~<r5a!bT55x01@x@6pvacc!1OXY>!5^_t(Eg`pr+!At2$SonagxnHxOUNxDw}jjha!bf9A-9Cw5^_t(Eg`pr'
        b'+!At2$SonagxnHxOUNxDw}jjhaxLUq$hDAbA=g5#g<K1{7IH1*TFAAKYqfDL<66eGjBB-VE#g|lwc5Cra4q3l!nK5J3D**?C0t9m'
        b'mT)cMTEex2YYEp9t|eScxR!7&;abAAglh@c60Rj&OSqPBE#X?i=_(IffrqWY!&cy7EAX%tc-RU&>^t!A@qA0h6~A~WAy@q3p$_Yd'
        b'xe{|F=1R=fW_3ANa<1fD$+==HSAwntUA10!CFx4im82_4SCXzIT}ir<bS3FZ(v_quNmr7tBwb0ml5{2MO4602D@j+9t|VPax{`FI'
        b'UakaP3Az$=Dd<wrrJzgoaw+Fhy<Cd9R4<q6<x<F{kV_$#LN3+IrHo4%mohG8T&kB#5tkw^)yt)XO9__}E+t$_xRh`y;Znk-gi8sR'
        b'srS_saVg?b#3}T!6?)ir=;52YsJxI%e)FKv!&c~FEA+4xde{m*Y=s{79eVhDT5>MrTyT;k1YHQa5Og8vLePbv3qcoxE(Bc&x)5|B'
        b'=t9tipbJ44f-VGI2)Yn-A?QNTg`f*T7lJMXT?o1mbRp<M(1oB2wQ?cnLe7Pp3$=0~=0dGpNV$-5p;j)`%7u&z85c4xWL(I&P%9TA'
        b'Zi=|6R&GkTDdDDsn-XqHxGCW#Pgtjjn<8$CxGCbMh?`pj1yjaN88>CzlyOtWU9pGHrzPa3kefno3b`rdrjVOLZVI_6<ff3DLT(DV'
        b'DdeV*n?i01xhdqPkefno3b`rdrjVOLZVI_6<ff3DLT(DVDddKb8$xafxgq3+kQ+j72)QBThT6Cx<A#hIGH%GYA>)SHxFO<(h#Mkq'
        b'h`1r*hT6Cx;f9185^hMiA>oFE8xn3vxFO+&gc}lWNVp;4hJ+guZb-Nx;f9185^hMiA>oFE8xn3vxFO+&gc}lWNH}C3mNE}Z7kF5A'
        b'<l)r<EM*>+G7pQHho#KJ(ghxtA`k11JbV_}+W4*NTh+I!Z*BWh@ulKR#g~dN6>k-96>k-96>k+U6)zPp6)zPp6;BmU6;BmU6;Blp'
        b'6%Q2;6%Q2;6`v|TReY-WRPm|eL&b-R4;3FOF654oJ3<bZcUa0hEae@R@(xRRho!v3Qr=-H@3542Sjs!BJMZwJ<q>g5#2pcLMBEW^'
        b'N5mZwcSPI~aYw`*5qCt~5pf5XuvtnxEF~V+op|_stUO%N#!I;)<&=3?$~-J(9u_kXOPPnI%)?UVVcnUB&joF}oI7&v$hjluj+{Gk'
        b'?n*p-?5r*5wxHXBZVS3C=(eESf^G}CE$FtO+k$Qjx~*1j%egJ*ww&8?Zp*nX=eC^Na&F7HE$6nJ+j4Hpxvf@ii@7c4wpzKZR&EQq'
        b'E#$V4+iK;ujN3A9tCia#Zi~1r;<kv}B5sSgE#kI_+ahj@xGmzgTDc|RmV{dpZb`T$;g*D35^hPjCE*s&R+ordB5sMeCE}KdTOw|W'
        b'xFzD2h+86ViMS==bcu(h#KXE151)4ta!bf9A-9Cw5^_t(U6F^+=PTuwlv`3x7kO9;JuHPD)*X8I=CM>>&Mi5&<lK^TOU_-XhtKCL'
        b'=$4>cf^G@A7IZD>TF|wiYeCn7t_58Sx)yY;R<7k-%ej_wE$3R!wVZ1?*K)4qT+6wZb1mmutz3(_Rx8(1uBBW{xt4M*<yy+MTDcZ-'
        b'E#z9rwOYBBam~}!Rxj5wu4P=yxR!BO?&0%k3At7?*D|hUTn|<ka#!%-<1QuTieEn{`mhvzSd2a_MIRQU4@;MOSV}%DB_GzEeE2LP'
        b'>J)ugiaxA6`tW&|#rlhzzxV{?UCFzWcg4Q01YQZe5_l!>O5l~iD}F>*60am)NxYJHCGkq)mBcG`btUjh;FZ8DfmZ^r)YO%{D|uJ)'
        b'uH;>*rz>%n>giJ2rFyzlPnWVT)zhV@OZ9Xq=~B|Aq)YX5Dd<u?UCOysPnYWHQaxP~xfF6K<Wk6`n!0RO7jh}&Qplx{OCgs;E`?kQ'
        b'xfF7`@WbMTAC@8zixG&W2*hFpV)4Qcix+-ay70rgGZ3E(+WVp|MO})z6m==;Qq+a03sDzr>_XOstP5EevMyv@$hweqA?rfcg{%u%'
        b'7qTv7UC6qSb)oau5OpEyLezz*3$=40=|a+lqzg$Gk}h=q8iFnaT?o2RHy7&WLd=D_xsY<9ZZ6c#g}S*AaUtSD#D$0p5f>sZL|llt'
        b'5OGt)O%XTM%}oh6CES#7Q^HLNHznMZa8trf2{$F&lyFnRO$j$8+~hgyWIH!y+>~)s#!VSFW!#i;Q^rk>l}sTwh1?W!Q^-vrH-+33'
        b'a#P4nAvcBG6mnC@O(8dh+!S(C$W0+Ph1?W!Q^-vrH-+5fNXd|LL&^;)H>BK<azn}uDL16tka9!H4JkLI+>mlZ$_*(uq}-5lLw(#('
        b'A2(#&ka0uC4H<VuAU<AKGH%GYp+0VixFO;O&sc|y8!~RlxFO?)j2kj;$haZnhKw6BZpgSH<A#hIGH%GYA>)RO8!~RlxB=t(4ncgg'
        b'@@4tfg&+D3L45OAtp7E0trSEn1<^`Dv@ZP6cL?J1X>C=<jBqOk(TYK|Vi2tuL@Nf-ib1qu5Um(QD+bYb4B~UsTh&|DOVvx&OPgMN'
        b')s?EJs;8=_s;4$RRXkKYR6JBXR6JCCs`ym#sp3<`r-~01A1Xdne5m+9aReb+L5Nlmq7{T_1tD5Nh*l7y?;yncMSY~)k#Yz^w1N<='
        b'3qQ1S5Um_UD+kfaL9}uZtsF!v2hqwwv~m!w97HPz(aJ%zauCfNL@Ni;%0aZQ{Ls4cLn{T*N<p+z5X}@sD+SR?LG+!1_*_tR8Fysd'
        b'k#R@H9T|6pAU-!;$eqE%F6DIPht`!JS}}-L45Af-XvH8}F^E<Sq7{Q^#UT2QL40ibmUCOqZ8^8)+?I1&&TToj<=mEYTh47cx8>ZH'
        b'b6d`BIk)B9mUCOJ+!k|N%xy8Z#oQKiTg+`Sx7EpQDYvEEmU3IlZ7H{<+?H}%%55pP)yZukx7EpQ8MoERZ4tLb+<GMJl5tDMEg83D'
        b'+>&ui#x1pSOT;anur3+5WZaT*OU5l3w-$H2kXu4-3ArWYmXKRKVO>&gNx3EEmXy1K5TDh>+!Aw3%q=mu#M}~dOUx}Xw}OXV&M64d'
        b'3PQAk5Un6YD+tjFLbQSq%^*Z82+;~cw1N<=AVez&(F#Jef)ITNAwD*wC0$FpmUJ!YTGF+oYc+E%=vvUVpld;QB_Tfcm72Mhb1mmu'
        b'&b6FtIoE3DTFkYWYcbbiuGP%7lxr#1Qm&<3OSzVEE#+FuwVJsWa;;{rWn9a+mT^7x?$Sc8g<PwfYZ=!vu4P>F8;BBeCFDxTmAbi-'
        b'aY{k7QV@NoAU+pVUC0&Nxsq}v<x0wxlq<D!CFM%WDFD$5K(qo7tpG&p>JP2_Lo5H#%0IO753T$|EC0~C`a>)J(275_;t#F(Lo5E!'
        b'ia)gC53Tq^EB?@mKeXZxt@uML{?Lj)wBiq~_(LoH(275_;t#F(Lo5E!ia)gC53Tq^EB?@mKeXZxt@uML{?Lj)wBiq~_(LoH(275_'
        b';t#F(L*Ma-53o|kr8>D3aVg@mSzX4Zj7u4J#UDP42)Pt;Ddd!YXyqUJ&Odx}7nK)tDdsY`@p3Msco&rybSda`1&CJqp_P7Ur5{@9'
        b'hgSNbm40ZYA6n^$R{EiperTp2TIq*Y`k|G6Xr&+ePCtBBmvtfQLe_<>3t1PkE@WNEx==e8qAo;Th`JDUp>8fDT}Zl+bfIqUia&fD'
        b'T0s|rF4WD1oC|exA?8BNg_sL<b0OtI-CPK{5ON{pLfu@*xR7xn<3h%Tj0+hTGA?A?lyOtWO&T|4+>~)s#!VSF)y_>3H;>-eC(l`@'
        b'kefno3b`rd=3;dzH>KQ^a#sN2vxu0RVs47LDdwh_n__ONpPOQCin+;tZpyjIer^i7Dd?u4o9ySNq??j%O1dfO=45qIH$~kPbyL($'
        b'Q8z{16m?V7O;I;Q-4Jy{C$2-%4M{g7-H>!c(hW&B)XWVvb3@JzIXC3okaI)L+z@j^%ndO&)XWVjH>BK<azn}uDL16tka9!H4JkL&'
        b'%nczo)XWVTH)Pz9aYN1A5OG7q4G}j)+)y(&c*Z(J+?c_;sJx6DGH%GYA>(i%h|+~1N)d=s1fp~uh|+Z+N&$$v0}!vhSV};Y5)h>X'
        b'L@@zTN<fsZ15pY<lmZZ?07NMOQ3^nm0uZGDL@5AK3P6;u15wI9l=2Uy{6i`KP|81)t^-kuKa}DRrT9ZB{!ofPl;RJi_(LiFP>Mg4'
        b';t!?xLn;1Hia(U%52g4+DgIE3Ka}DRrT9ZB{!ofPl;RJi_(LiFP>Mg4;t!?xLn;1Hx(-As{!ofPl;RJi_(LiFP>Mg4;t!?xLn;1H'
        b'ia(U%4|T^MJ`6u1?ufX9k-1U;q7;BA1t97UKzuG}+hyF5ak>md-2sSiR=DwEP7#Pw1fuQ;#5Z@*<_o$b=#HQ}g6;^qBj}ExJAzIT'
        b'h*AWibR~#V0-}_FC?z0D35ZexqLhFrB_K)(h*AQgbRmdR0HPFtC<P!&0f<rnqV531`v`7Hw<X<{bX(GG^>SO#Z9%uy%WXNg<=mEY'
        b'Th47cx8>ZHb6dUKRxh{J%WWaI)yr)ex7EvS5w}I$7I9m|Z4tLc+!k?L#BC9`)yr)Ow<X+`a7)513AZHNl5p$M8`~0bOT;Y^w-E9w'
        b'B_N6kh*AQg?gYd)chS~wZoQaWVs0&#mvc+bEjhR3++stw1l<yJOVBMrw*=i1bW6}JLAM0m5_C(@EkU;g-4b+5&@Dl?1l<yJOVBMr'
        b'w*=i1bW6}JLAM0m5_C(@wV-Q3*MhDEU8|LAIoEQo<y@<kYcbbq<yy+MTDcZ-t&`T4aaa7|ZAq(@YZ2EXu0>pnxK=CI60Rj&OSqPB'
        b'E#X?iwS;R4*AlKJTuZo?a4q4Qr>iaETEuneZM9`w%ea<tx)Ma)0f^5PDc3w>Z86tkuEm@#1yPDX6fXr)%0QGd5OrrDJ{Pp{CFx4i'
        b'm82_4SCXzIT}ir<bS3FZ-CRk!l5{2MO4602D@j+9t|VPax{`Dy=}OX-q$^2RlCC6ONxG7BCFx4NTnV}obfsRd<Xp+Ql5?eAuGGtw'
        b'lq>aeCFDxTm5@vIaw+3dy<Cd86mcoyQoUSCxRh`y;gYAUDdJMZrHD%rmm*FPh*AWi6oDv3AW9L4(iI@;PC$G<Eg_e!_w|%=Ddkei'
        b'rIb?yq7;EBMj+~rKzy^J`uqwaC0$ColytfTMDY?3r4U5v3J|3XL@5JN%0QGd5Ty)6DFac;K$J2Nr3^$V15tMd;&UOipyIMFWL?O*'
        b'kaZ#JLfu@5x)60C>O$0os0(#-A?ZTWg`^8f7m_a2&4r)~b#o!-Le7Pp3pp3+=0ePcy17s{7eX%7&4r8$85c4x)XjyuxsY%n;X=ZN'
        b'gbN855-ucMNVt%2A>k%ZS*M7bB5sPfDdMJxn<8$CxGCbMh?^pAinuA_rihy&Zi=`m;--k3B5sPfDdMJxn{4K$jGHoU%D5@xri`01'
        b'ZpyeR<ED(8GH%MaDdVP$n=)?7xGCeNjGHoU%D5@xri`01ZpyeR<ED(8GH%Map)PKSxFO<(h#Mkqh`1r*hKL&?Ziu)c;)aMDB5sJd'
        b'A>xLJ8zOFqxFO<(h#Mkqh`6CHZb-Nx;f9185^hMiA>oFE8xn3vxFO+&gc}lWNVvgoABKn<B5sJdA>xLJ8zOFqxFO<(h#Mkqh`1r*'
        b'kbFp9{UH^7$UFM*&0RuwQTeZtOI`dS6@5rWA5zhWyrU1F3o5=<e5?3Y@x>j#RDG%XQuU?ktxazgZxwG9Zxt^UFBLBpFBLBpPZduU'
        b'PZduU&o4(LQ_+W1^dS{}NJSq~(T7y@Ar*Z{MITbphg9?-6@5rWA5zhWRP-SgeMm(gQqhN0^dS{}NJSq~(T7y@Ar*Z{MITbphg9?-'
        b'6@5rWA5zhWRP-SgeaJid@L}H(aYw`*5qCt~5phSv9T9g#+!1j{#2pcLMBEW^N5m=kkh=6kD)*3g?%`uW#vK`VWZaQ)N5&l)cVyg='
        b'aYx1-8Fysdk#R@H9T|6I+?H`$#%&q5W!#o=TgGh}w`JUxaa+c18MkHJmT_CgZ5g*^+*TL2Mcfu~Tf}V<w?*6*aa+V~5w}I$7I9m|'
        b'Z4tLc+!k?LUEG#%Tf%J#w<X+`a9hG{3AZKOo;q*XB5sSgE#kI_+ahj@xGmzgh}$A=iMS==mWW#-Zi%=h;?~id>XLCw#x4BlQr@YD'
        b'kGlxDCFGWnTS9ILxh3Qjc}PVbQjv#L<RKM#NJbv=jy!y`!d)-umY`dLZV9?2=$4>cf^G@ACFqu*TY_#0x+UnApj(1&sg+xDZppbN'
        b'=a!sXa&F1FCFho$TXL@DT+6wZb1mmu&b6FtwQ?=yTFkXtxt4M*<yy+Mlxr#1Qm&<3tCedZ*Fvs^Tno8YE7xk}TCH45xGuf7wTNpG'
        b'*CMV(T#L9CaV_Fn#I=ZP5!WKFMO=%x7I7`&TEw-8YZ2EXu0>pnxE66O;!4ETqgPzXxRP-t<8;l3)HNScd56674&OYM%1gPDawX+T'
        b'%9WHWDOXaiq+Ch4l5!>GO3Ia#D=Ak}uB2S4k1HWpLau~d3Aqw-CFDxTm5?hTS3<6YTnV`nawX(S$d!;QAy-1Kgj@-^5^^QvO30;<'
        b'OZ9On<5I?@j7#-#DdJMZrTVy(a4F$Z!li_}@(v%qrie=smm)4jT#C39aVg?b#HEN!o~)*f(<L8Ld52WqA(eMX<{eUbhrII+p9{su'
        b'Qh70_OFpFH4ym|9>WUA^v_mTGkV-qG(hjM#Ln`f%N;{;|4ym+5D(#R;JEYPMskB2X?T|`4q|y$lv_mTGkV-qG(hjM#Ln`f%N;{;|'
        b'4ym+5D(#R;JLH{qc)vtL&V`%{ITvy+<Xp(PP%9T=F4W3}lnW^rQZCfWg^&v&7i#4~#)XUvwQ?ciLd1oL3$=0~;X=ZNgbN855-ucM'
        b'NVt%2A>l&8O$j$8+~kzp6me6;O%XRm+!S$>r>j%OO&K?3+>~*X&D<1nlXG&DQzVn$KFq}@u=Kt@CEb*CQ_@X#bW_w#c63wLO<6Z('
        b'-IR4x)=gPAW!;o@Q`SvcH)Y+FbyL<&SvO_flyy_qO<6Z(-IR4x)=gPAW!;o@L)HyhH`L7yQ8z^05OqV;4N*7L%?(L6B;Am7L(&a('
        b'b3@P#b#p^!twY`1P&YT!%?%khWZaN(L*3jEaYNnQkZ?o74GA|S+>mfX!VL*GB;1g2L&7QbkV-wIQV*%rLn`%<N<E}f52@5cD)o>`'
        b'J>;Ew_}p|EH)I@Q51|V_#GQJ0RS#Y9A(VOur5-}5hfwMvlzIrI9zv;yQ0gI+dI+T+LaB#P>LGN!hfwGt6nY4S9zvmqQ0O5PdI*Ic'
        b'LZOFH=pht(2!$R(p@&fDAryKDg&snohfwGt6nY3<?;#X=2!$R(p@&fDAryKDg&snohfwGt6nY4S9zvmqQ0O5PdI*IcLZOFH=pht('
        b'2!$R(p@&fDAryKDg&snohfwGt6nY4S9zvmqQ0O5PdWbvp@bNxzM4VC&q0~bt^$<!ugi;T2ryf453%MiYj*vSzSq<eLLb-=f?je+W'
        b'2<0B)&OLlCsJN6nQtn8(Bjt{iJ5ugQxg+I{lsi)HNVy~Bj+8r6?nt>K<&KowQf^DRE#<b9+fr^zxh>_kl-u}RE|hx+<sL%0hfwYz'
        b'lzRx}9zwZ?Q0^g=dkEzoLb-=f?je+W2<09^xrb2hA(VRv<sL%0hfwYzlzRx}9zwZ?Q0^g=dkEzoLb-=f?je+W2<09^xrflj9zwB)'
        b'Q0yTTdkDoILa~QX>>(6;2*n;kv4>FXAryNE#U4VjhfwSx6nhB99zwB);Kd$77kda^>>=*l!#69czgeH%+>&!k&Mi5&<lI`UF6fq^'
        b'TY^r>htSm?Lcxc)gAbpNCF+)_TcU1>x+UtCs9U0LiMl1~mZ)2zZi%`j>XxWmqHc-0CF+)_TcU1>x+UtCs9U10MO}-!7IiJ^TGX|u'
        b'Yjtxi>00NkE$CX!T&tODG1qG5TFqPwxfXIQ<XXtJnz@#7E#q3owVJsWaV_Fn&0I^kmT)cMTEex2YYEp9t|eScxR!7&;X3rz(ju;>'
        b'-q(9BuBg6{Ya!P{PFH&fg&#uUhfw$-6n+R@>>-qWh&%i6@qA0nm6$6rcV!<wi^#c>b0z0W&Xt@iIahM7<Xp+Ql5-{JO3sy>D>+wk'
        b'uH;<Fxsr1w=St3%oGUq3a<1fD$+?nqCFe@cm7FU%SL)<S%#}L1QYTkJu7q3(xe{`vPOfBJ$+(hnDdSScrHo5;aw+0c#HEN!b#f`;'
        b'Qo^N#O9__}E+t$_xRh`y;Znk-gi8sR5-ug2F7*())I%ux5W3PsDESadK7^7FaVH->7E=98%D9wqSMcGR$5MGAm!W<prCdrmB_Be`'
        b'hfwk%lza#!A4188Q1T&^d<Z2ULdl0v@*$Lb2qhmv$%jz#A(VUwB_Be`hfwk%lza#!A4188Q1T&^d<Z2ULdl0v@*#Achfwk%lza#!'
        b'A4188Q1T&^d<Z2ULdl0v@*$Lb2qhmv$%jz#A(VUwB_Be`hfwk%lza#!A4188Q1T&^d<Z2ULdl0v@*$Lb2qhmv$%jz#A(VUwB_Be`'
        b'hfwk%lza#!A3~RT2n8QP!G}=rAryQF1s_6}d5Amr@bR`bMcfo|a~B^gxT1|0a#P4nAvcBG6mnPS;q&=&kYtLv$?qPfoSSlP%DE}$'
        b'uGGV4bwM`;-4t|F&`m)%1>F>MQ_xL8HwE1kbW_kxK{o~66m(P2O+hyW-4t|F&`m)%1l<sHL(mODHw4`fbVJY$K{o{5P%Ag&+>moa'
        b't=v#6H>BK<azn}uDL16tka9z<+)yhwWZY0IH$>bJaYMun5jWJz4GA|S+>mf1_1-o_+z@d?#0?QQMBET@L&OacH$>bOd-&XR88@c7'
        b'iYxf=&5G(vxgq7Q;KQrPRPtf!dJj{<hj|Adz9}+r7yR<lRP<pg`Y`Y4!#9tm@>?6fReh`a*V0X8AEvSoQ`v{9?88*{VJiDDm3^4X'
        b'K1^jFrm_!H*@vm@!&LTRD*G^%eVEEVOl2RYvJX?)hpFttRQ6#i`!JP#n94p(Wgn)p4^!EPsqDj4_F*dfFqM6n%05hGAEvSoQ`v{9'
        b'?88*{VJiDDm3^4XK1^jFrm_!H*@vm@!&LTRD*G^%eVEEVOl2RYvJX=ie3*(p%scw<xuD`pxE~RBTJJ4K#vK`VWZaQ)N5&l)cVyft'
        b'ZhPr%{YbeZ<&Km)Qtn8(Bjt{iJ5ugQxf9%UF?TRZGL?RqN<U1cAEwd|Q|X7P^utv8VJiJFm428?KTM?`rqT~n>4&NG!&LfVD*Z5Z'
        b'(TAz<!&LZTD*P}NewYeBOobn&!Vgp7hpF(xRQO>k{4f=Mm<m5kg&(HE4^!cXsqn*8_+cviFcp573O`JRAEv?&Q{jiH@WWL2VJiGE'
        b'6@HisKTL%m<{f_c0Bg&*tyXS}xGmzgh}$A=i?}QO@VOx}Zp*kO<CctDGH%JZCF7QiTQY7PZTb>&OUNxDr|`p6_+c{qFd2TB3O`JR'
        b'AEv?&Q{jiH%RbCI`|$a+np-dFlzx~>KTM?`rY`$16@HisKTL%mros<X;fJa4!&LZTD*P}NewYeBOobn&!Vgp7hpF(xRQO>k{4f=M'
        b'm<m5kg&(HE4^!cXsqn*8_+cviFcp573O`JRAEv?&Q{jiH@WWL2VJiGE6@HisKTL%mros<X;fJa4!&LZTD*P}NewYeBOobn&!Vgp7'
        b'hpF(xRQO>k{4f=Mm<m5kg&(HE4^!cXsqn*8_+j4RhY!PB#I=ZP5!a#jmX>iX<66eGjBB2;wvfBR58vEH<)vIpxn@6ir60aoLHUw%'
        b'CFe@cm7FU%SNsB^1YHTb5_Bc#O3)P>x{`Dy=}OX-q^rg1qOL?;Ep?M6>q^#@tSebpvaV!Z$-0ttCF@Gom8>gSSL)_U)Rm|!QCI5b'
        b'O4602D@j+9uGGzypeuEACFe@cm7FVebER&sq+Ck5lyWKMQp%;2OLcQ8<Wk6`kV_$#LN3+KrHo4%mohG8T&kPPrT4a!ajAALMO=!w'
        b'6mcoyQpD-n4^s(<sRYEl6A<4#mV8esmr^dJTuQl=aw+9f%B7S`DVI_%rCdt6lyWKMQp)AvVHa~L=2Fb1m<ur%VlKp7h`A7RA?8BN'
        b'g_sL57h*2NT!^_4b0OwJ%!QZ>F&FCOLdu1d3n>>;E~H#YxsY<9MlOV02)R%r7i#1}#D$0p5f>sZL|llt5OE>mLd1oL3lSG0E<{|2'
        b'xDaunMlK{=NVwqnYKXWg;--k3B5sPfD*^HGw5E)kGH&u4h$-Zzkefno3b`rdrjVOLZVI_6<ff3DLT(DVDdeV*n?i01xhdqPkefno'
        b'3b`rdrjVOLZVI_6<ff3DLT(DVDdeV*n?i01xhdqPkefno3b`rdrjVOLZVI`nHg3weA>)RO8!~RlxS=*~h`1r*uJFTKxFO+&gc}lW'
        b'NVp;4hJ+guZb-Nx;f9185^hMiA>oFE8xn3vxFO+&gc}lWNVp;4hJ+guZb-Nx;f9185^jXf*M^82B5sJdA>xLJ8zN5OhpF(x)I}er'
        b'F8VMPeVB?qOhq52uKDo%ITa8;EyU--(n4t=v@o!6RQ&u}{Iz?ZKM#9;c^6ebs=kkkY;F54-bLlN$}esFQuU>6Un;&-yj8rp=bPK!'
        b'!Ihzv;?6HVgW@wNK7-O8NIrq&6G$C^)B#A94;_He(GL|56%Q4kDn5TL{@(IgS_mx+ER3TC6(1@-el1?_h4JOWxw-pT__g?XFO(L3'
        b'Ek3WEcy8|A3m11kpIKiE>sqLLp|o%pU-!btZvV@|zIXdweBZnMu5n?P?}v(O;q_1h3zv2Idb*c&`O|a17V0`suZOyi`Tca)ea!C~'
        b'eIN6?#%0~d!mq{qwdI#|9}5EumvjAkx~1Z!Eidi%(lO72S7Ltsr*>}PzWZ4DTA6bb_x=NkgV$&*ys)BhY~$YVuZQ}z_<HTdi@5z>'
        b'xc|`a8W(ZT>sejIeJtEte%H8&JFg>k5x3t9b#2GJffsS>eUI_w!oG>?>sjraxOHvEzKL7!d&Kp0*S#>l+`xSk_q=Lc#68z;-*4i6'
        b'`al2b&o7_-ChoN`u(0pq_p{pT%LBD<;@;b_wdEIa`?Y}=aqpeDmKL<zSE$DL^5wH{;yxA@w&No1;~pyB+Vawt*ZT5By>H^)uf&Qk'
        b'wcIyx^?Im0Uy`wJ;_~&Zvc7yT+P84|eyFeC*7hx2zP2N7U+)@GU%uJyTex^X)cCRm*tc-=b&vUVE3?(a?OV9<ekk<s`zG$YdaqpV'
        b'^Sl0q+;{V}BQNE?dvE2%+;{i8$hw{T?%rkVhVFCa_4XF`%IgfG?v?8q<m=Hc?7qv_quuAP`FgaAyYKR~J(qXi-LKengZEv%9&Nvl'
        b'!u@C$dEfQxS>NZd{eHB#S6+{HJ%janw2Qs(_IqXBEBE$X@O|DVSZ8Q^`n3=4qtHwHx_K1tbJ>qadmV*)d+u}Dey^;1dtNK`@mKQe'
        b'%JmGM_sY0e?jv`f%>FY5^PgANYbCCg*HO6lL55!OIllf2UN?m=57uqr%aElr+NCqvr8C-fWB9ItGukB#m(FOH&S;m;XxEFv_s$=$'
        b'!+*apZVLBn7mxAv%dB4%-VF4M!hGKce**oY@Mf)F6vpd5=X?!ofBRA1uPep-0ONj9_<nBtXiIMj-+Sk#@M9&o0tWg;VZZLPUJSfe'
        b'>g%K3FADp$-`3C`?Qa?QZ!7ETKhNgJdd-jZnjh;mKh|r0tk?WlulccF>tnsXCHz=eIyza}(-~jC@%9VC_g_yO{f)dKypMI%*G*2}'
        b'5`M0XuUn?REqo!}H-?`p#g$;4q31TJ!C3ms9`*H`PQOHa|ILr_b(>W$5%1$u6tO-;e664<sF#TKdIksWX}v_e8{~S4Snn15C6vD2'
        b'D1ulo5Z^mz*VntcULeN%KI7|yT`v&l>plxF#Cn1F-Z?jjA1gz<t&etFFYLB1Erb?^iWJM&(o-((fQqkmuUBLE-ktBp@P2*18^inH'
        b'Wa)rraAkb`cU>PIzE&1jiYvhtz6|H#%F^HHEbX1Qhp$H)IzG3D?_)anIvAzb&d>{SeBGd?zV*l(!;gipzxBwQ!jF3l+~cVDQSnpQ'
        b'4t&R(ueR`gmyPq1d@1;T#=E$JgP$)2-zy8B668z4+u?jEc-x&X1wS4Qe*pP$hJ0^NFC4V4FF%gTTfzJ9FK-3owJr7a7f*Q`__6Ty'
        b'w_bS*_}-2h+QK|KE&?#W{SxHY6*NQnab|qq2jBGO?cc{e>dP-q^8F#<T|Kn(U+2a5J*Ia4)Xu-)d%srZzy=P!?gsz%2LJX(=x-eE'
        b'#qHjF{SxT2zTCrcyLa!FxZS&VWZdq(7tX;$9Tk6l6%wD9IRAFn$2Z^hzORq{?J)_Bw#1Fz$3uPnM;AAF^W&1X*pBt}_uO%N_puPz'
        b'S8L!Nt>V4#bi1&3TU*`(J5dc^@bous?Y_8x)|U%9>Ur>lb7?`_o_fjQ>)yDXd)ElP<f61sBEFU*UTxmH{ybb+Tq&;b8O))T!J|KT'
        b'^auBWyo=XqZ+-h4zhAFlAI#Tj@BH>RRKIxdoUhZ~`R#9TetDm>xWZ>}f-6JYvkzCe57hC^SI76U!hNs~SGW&WafL@=;Ul2=@j+mG'
        b'{^-irD|`mcXV82Gb-04QYral>KkgG;8G3HXeURJ-$yZ|Xm6&`b2LBpkaRnaCk5k`oLR#MfJ2mGwZxO%9zkIn6T2S$!;zPxUivL1k'
        b'ezobpP?%p13jh3A{nNka=gPWP>YHKFuUFz)Ip2&LezkI5&)~hXxKdmRuIL$TK7-9?u(=QR>lu7}<&Txcqp)7j;Js2@xt_uM>%Z~!'
        b'Lf`erH(#QBxlmfTiod^$7+){cT|B<|!19Y4rQ)S6uh*{s_^^a4iz~&I;EE1@yq>{(WpSmr5?s+Un0yA4&tUQyOzwlpqwvcgXY|je'
        b')#r;@EA{7<{B|Y&J1ZS4=Fg9I?ia|<7oB)^?O*P3-JZC;ar?&Y&wcy#gnrKM{<QGT3-;$<XZg$Rum8UB-*4Rhe0h7IZrpY6Jr{T3'
        b'_rlh~^YudBwciVO@wyjU_t;uM@lx^9mY0f`ikFJFinoe4j(6jDH;(th@m_b~*X{k~>u=q4U$;MB-oEd;uiL+_yRp3+bvL&6V0#bh'
        b'9v$R^9es3+A5?yHh@U@~t}hn07M`yce$LMQyz)((_j9073$2B%h378|FKOn)?TOnPw{P5zx`DcZx`DcZx`Dchx{11px{11px`n!h'
        b'x}_srC|nQj(0X;XUR|x)tyfp8_)_tu;!DMsiZ2!4D!x^GtN2#&jpKcAybq4|!Rzeca39otQ1?OI2X!CReZH(4w_)$w6Ssf8r{fmv'
        b'eFqet6Av^|d8+c<T6nM!*yup%uMrC%7B^z=!imC(!imC(!ik+-s9UI8s9UI8sN1M}6IQRj_d3YuwQwEe`Ff}a_qYjr)mYf*OR(2L'
        b'y8wF^__gxu8C`t6Z%^FbUqAk3SACuWeDypJcJ@iU8ZW!vd+5UJV*$tcLFr4c_sL3sy=f12^`+Oj&pR%?-nZvhkDs?*@7woR&ohBn'
        b'qWj9*u8)(j*22akq4Gu7>u~~G9N6O6U+=k#uGfEm$@RV+b+f;Io{2r3*xtgP{#vetTzSj&J^+QnjUC;I+p(h?8{F8@3w0OjE^P3^'
        b'3u|F}H+J;aOM0X5M&XUZ2OE5_!3T8@cJx8r2OIq0mGz+RqgU4Rbr(=NZ!DcR*6r3+yU=tM+$y+L=(!4KjRmzz9ar6KeJp5`qqVTL'
        b'pyH|Gsp6^Psp6^PrQ)UHrQ)UHrQ-c{4^(Qv>eYa~7q%82EG!jYD!#t%{_1rO`Cd@*t>Rn7x3+w1w{PwCqho&XcXo8h5B|`ON<a7u'
        b'e^B|sU-*N<U!O-d&LbP=kqyxrqBTTookq4=u~sYAI*)9r)>5rss?}?uwXn6I;yH1T)Rw2VJhkPi;-xJw6)zm{!T~QF@Wu<R@j`3V'
        b'ZPe{A&#wB`?DgL-)Lp2%P<LT_H|lQG-8#ry2YKrtACx{i#z(b}Y9G~p^a6WS{88}-d;Dw47ARYwY=N=`$`&YFXwDXzvxVktp*dSP'
        b'rCj(lY~j<eg-^p4J`G#=NPpqOu!Rr9miSuYYl*KVzLxk}YRZ<HvZbbM>4W{H5B8Tn*kAf!f9Zq$r4RO(KG<LSV1Gff1<95^)?c7('
        b';S6)(40GWObHTBNZfxNUbD<eqIKy1%#TI(8CBYU1TROwsI>X#BZ0lqFtuH3`t=M~Es`f<fsoGydwgK4&WE+reK(+zd24owMZ9um1'
        b'ss6^N`Wv5(ZG1Mi@!8l$3%1dMZM0w;E!aj2w$XxZv|#(j>vj7=-G#afbr<R`)ZKbTZxr4ryis_g@WBQj)IIu3Iw*Wl_(9<Zg&%D2'
        b'gSub0UI(q$0b2*H*Fo!b(0U!2b#xYagx1kn<k4B=(OKkyTL*3(xOHx~UXP#H+nM<98@HowVs97fmUeb&XP0(%X=k@~c4KEZc6MW<'
        b'8ymf_(F;3!q3%N6g}NJcH@0`9?nd2>x(9U+>K@en`eXfb)AhRjLEQ&+AJl!Yy<gMyfUXC0J?gj~*!94!2d9k>+N}qtjSsr52i?|#'
        b'Zz3P4^+2r$YCY(-9;o%8+j`J#Jy7d`S`XBE&}}{FwjOj_54x=f-PQxJ9)R@#tOsB{0P8`w^#H6#^Is3ddem+`AnO5H56F5z)&sI0'
        b'koBnDdh~JEqmR2Decbiv<E{s8J^DKG(aGZL@(iB~UtQ2aSAftZBXo@kU6kT=8H<nG7oO-u;XvWg7Ds6Tn;fW|sGQj3#3m=|Ch8XI'
        b'7U~x27U~x2HtII&HtII&HtH_aU8uWIccJb=-Ho~%bvNp6)ZM6iQ1_tjLEVG82X!CReNgv7-3N7_uj_t&6)asgOV`=bg}HPkFJ1ae'
        b'*9z0c#dI|?UA|1$JJSWybVW5?f=$<KdtK!14c`P<6JSk%H38Nv6^2+7VoiuOA=Y(4zmLKoYl5r^vaYKVeiT@$474WDnm}s;tqHUy'
        b'(3(JN0<8(OCeWHdYXYqav?kD+Kx+c63A85Anm{XnRsgL4S^=~IXa&#;pcOzXfK~vl09paG0%!%$3ZNBevI1rW%nFzlFe}hx1;`4J'
        b'6(B1>R-nlWh!qekAXcEs3N%^Q<)~kd1-uG)74RzHRluu&R{^gAUIn}gcopy};8nn@fL8&p0$v5Y3V4<9D&f_2P5Sp6*mV{A&jnP5'
        b'SP8Lmsxrt*kd=Y=;DlMJGs6U03A7SuCD2Nsm8pFVwGwJ2)Jmw8P%EKULal6^geTZau$5pd!B(QpO0bn+E5TNRtpr;Mwi0Y5*h;XK'
        b'U@MPa(r_!`R>G}>TM4%kZYA6*xK(hg&}S9;tb$qvwF+t#`m6$41+)r%R>7=7pH(2MKvsdQ0$ByJ3Vl{Vtb$ktu?k`p#43nY5UbE<'
        b'6~HQhRRF61RspO6SOu^OU=_fs4WSES6~rotRS>HnR-w_VgZILMSOu}_(H1|Y1#B|RYM9k9t6^5dtcF<)vl?bK%xaj`Fsor!!>oo`'
        b'4YL|%HOy+5)iA4JR>Q1@Sq-xqW;M)enAI?=VOGPehFJ}>8fG=jYM9k9t6^5dtVWa7Age)EgRBNw4YC?!HF~UuSPiinJyxT~YIxP~'
        b'YQd`ouNJ&o@M^)U1+NyoS~@da0BZrP1+W&tS^#SStQGiJctNZMu@=Ny5NkoK1+f;yS`ceNtOc<a#99z*wT>{zS|DqItOc?b$XXz4'
        b'fvmOhnZtrv3udjo^gw!TL9GR~7Svi$YeB6AwHDM`P-{W01+_NR+8TA(`bcZTtqr#}+}dz!qtDu4YlE!~wl>(>U~7Y|jXrBbtqrv{'
        b')Y?#ML#++9Hq_csYeTIKwKmk+P-~;l+CXaqtqrs`(Aq$21Fa3THu|g$vo_4y=(9HZtPQa?#M%&RL#z$4HpJQxYeTFJu{Ojy5bH>+'
        b'gGTEBtOKwPz&Zfy0IUPB4!}A9>j10+unxdF0PASd;Xtedv5r1~KOpOXtOK$R$T}eFfUE<u4#+wn>$El)W*wMyVAg?I2WB0Zbzs(k'
        b'SqEkv^;rjK9iVl9)&W`vXdR$+fYt$82WTCjb%53bv>u!sJ}~QnSr5#5VAcb(9+>sOtOsU2FzbO?56pUC)&sL1G+7VGdO+5LChLJ%'
        b'55#&P)&sE~i1k3M2Vy-C>w#Dgnyd$4Jpk(gSP#H@0M-Mr9)R@#tY_nF<w2MA0IUaKJpk(gSP#H@4o*lPi1k3M2Vy<XQh~3w_@A#*'
        b'^yTe=+Y7gUSvW8R9~gp<dkFsHfhH<{iB@0;J}?9y_YnNY?Wh~58>pM8o2Z+po2Z+pTc}&8Tc}&8Td3Qp+o;>9+o;>9yHNMb>8!vA'
        b'd|(7VFajSKfe(zp2S(rnBk+L{_`nE!U<5ue0v{NG4~)PEM&JV@@PQHdzzBR`1U@hV9~glTjKBv*-~%J@ff4w?2z+1!KJF3tkDq4-'
        b'SQB7PfHeWu1XvSbO@K84)&y7+U`>EE0oFAE|FPpB)`VEs2>izbfvgF#Cdir~3nTDxkHEk0uuvFgT@&!11+cHb{tP}a10R@y56r;F'
        b'Jp=zy0BlXLHNn;dTN7+eur<Nf1X~kqO|TVUE5KHOtpHmAwgPMg*ov>eI1Eg|2d3ZyQ}BT)_`no=U<y7k1s|A#4@|)arr-lp@PR4#'
        b'z!ZF73O+CeADDs<Ou+}H-~&_efhqXF6ntO`J}?Cz_Z0lc?Wl`BD?nDD&kBeY5Gx>7pw9|`6#y#$RsgI3SOKsCU<JSmfE55M0agMm'
        b'Ou@%J1^;meyfq}qN|2T4v=U+^#7c;j5Gx^8Lac;X39%AlT_f-xI}T)BBk&)0fLW<l>l%XpTtI24l~608Rzj_8bUX>R5^N>dO0bn+'
        b'E5TNRtpr;Mwi0Y5*h;XKU@O5^f~^Ew3AXalOABr#+)B7raI4@}q0uU^RcN#dY8BKfs8vv_pjJVxf?5T&3ThS9DyUUZtI%f^&?@v<'
        b'1+xl$R-w--h*c1)AXY)Ff>?z<s{mF3tO8gCuuA8K1+fZZ6~wBI_l5#l1+ofc704=(RUoSl-VU!Z_|FAYhFJx(3T8FTYM9k9t6^5d'
        b'tcF<)vsx#I4YV3)HPC9H)j+F(Rs*dDS`D-sXf@Dkpw&RDfmQ>p23ifY8fZ1pYM|9XtASPntp-{Rv>IJj!>oo`4YL|uR)eetSq-up'
        b'WHrcYkk#n28eLWctOi&Ouo_@3fVBYD0$2-REr7KE)&f`yP1b@}*A)C4xCO8lz*+!n0jveE7Qk8nYw6r@L97L_R^h#Hfvg3x7RXv4'
        b'Yk{l<vKGi%AZvlFwY0%2w1CP$YXPmL58)ToT2Kpv@No~qzwZFHt~vOR2ZCGcs50PMfNKG+1-KUAT7YW<t_`?0;M#y|1Fj9YHsIQT'
        b'YXh#0Mr*^Z4YxMj+Hh;btqr#}+}dcgHrU!=YopQHP-{c24YfAZ+E8mltqrv{)Y?#MqtV)Ev^LDzFl)oCjYeyOtPQfZ$l4%lgRG5C'
        b'YeTG!PHXGra6_yOu{Olo5NkuM1F;UoIuPqXtOKzQ#5xe`K&%6?4#YYT>p-jnu@1yK5bHp!1F?=etpl<S$T}eFfUE<u4#+wn>wv5S'
        b'vJS{PAnSmv1F{atIw0$StOK$R$T}eFfUJXa!vnDn#5xe`K&%6?4qB`OunxdF0P6s(1F#-|^#H5~U_Aiq0ay>fdH~h~upWT*0IUaK'
        b'Jpk(gSP#H@0M-Mr9)R_r#d_e?1Fs%<^}wqKUOn*YfmaW_df?RquO4{yz^eydJ@D#*R}Z{;;MD`K9(eV@s|Q{^@alnAk4_4oQ(8b}'
        b'i1q06_XlJ>AnO5HUr)g&rr;A(@QEq-#4o|;Jp}*0Cid4;@Tn>I#1wpD3O+FfpO}JAOu;9n;1g5ui7EKR6ntU|J~0KKn1WAC!6&BR'
        b'6I1YsDfq+`d}0bdF$JHPf=^7rC#K*NQ}Br?_{0yvC#K*NQ}Br?_{0=^VhTPn1)rFLPfWolrr;A(@QEq-#1wpD3O+FfpO}JAOu;9n'
        b';1g5ui7EKR6ntU|J~0KKn1WAC!6&BR6I1YsAA(N|!6$~`6GQNcA^5})d}0Vb^+WJ^Pr-jKpfbdo5NkrL39%-`nh<M3Ec_6BVhTR*'
        b'DfrJjpmO8ga00Cfw5~Du&j(ssz(#|u3AQHKnqX^!tqHa!*qUH#f~^U*CfJ%_Yl5u_wkFsLuoYk{z*c~*09ygJ0&E4?3a}MmE5KHO'
        b'tpHmAwgPMg+N?mE6+kP1RsgL4S^=~IXa&#;pcQDd0%ir;tN>YoHY*@jK&(KU6=<^pUIn}gcopy};8nn@fL8&p0$v5Y3V0RpD&SSX'
        b'tAJMluM%D*yh?bL@G9X|!mET=39k}f<-}WX0;~jB_#ODXhu}XKu*vGQ5@sdLN|=Qy_{0=^;+No4L-2_q_{0!=VhBF<Q}Br?_{2}a'
        b'Cx+k?L-2_q_{0!=VhBDl1fLj!PYl5)hTs!J@QES##1MR92tF|cpBREq48bRc;1j<DpO}G9%)lpR;1e_Oi5d9B418h+J~0EIn1N5s'
        b'z$a$l6EpCM8TiBud}0PZF$15Nflti9CuZOiGw_KS_{0o+Vg^1j1D}|IPt3q4X5bSu@QE4t#0-3520k$ZpO}G9%)lpR;1e_OiC=<G'
        b'jKC*G;1eV8i64SbOu*+o0sn@+09FC40$2sG3Sbq$Du7i0s{mF3tOi&Ouo_@Bz-oZi0IT)+dqb>-SPiinVl~8Sh}96QAyz}IhFA@;'
        b'8e%oXYKYYkt07iHtcF+(u^M7E#A=Au5UU|pL#&2a4Y3+xHN<L&)ex&8R-?mefYku20agR723QTS8elcRYJk-MYXPhUuol2t0BZrP'
        b'1+W%6tOc(Yyjt*T!K($Y7Q9;UYQd`ouNJ&o@M<MaM;E|a0BZrPrIW%1u~y-|e}Sw8vKGi%AZvlF1+o^%S|Drb!}kTV7R*{OYr(7q'
        b'vlh%+Fl)iA1+x~+S}<$DtOc{yqn8%YT0m<9tqrs`(Aq$21Fa3THqhEYYXhwfv^LP%Kx+f74YW4U+CXaqtqrs`nyd}8Hq6>EYop29'
        b'AZw$^+7N3)tPQa?#M%&RL#z$4HpJR!vNpil0BZxR4X`%A+5l?<tPQX>z}f(71FZevquC9yHoB}0ur|Ot0P6s(1F#OjI-1Km5bHp!'
        b'1F;UoIuPpw_VfW+N1wkReg1xc)&W{agANC39kp5qY@O0Z!>t3it_k?h2SR1QbpY1^T=*&Y#0-3520rf@_>TwDsKbF>2X-CUbzs+l'
        b'T?cj@*mYpnfn5i79oThX*Fm#&K-U3X2Xr0K^?<GibUmQ!0bLJztp{>Fkn4e559E3v*MnZ`0bCDytp~l<1GXNp^`O^!pw@$4>j7F1'
        b'daVa$JuvG*ul0be2V^}U>j7C0daVayJrL_bul1nUdf?RquO4{yz^eydJ@D#*R}Z{;;MD`Ko`bgr{1$xTx8M^)@QES#)NjEjrr--x'
        b'@P#S(x~Jgp0);X7!f(MBrr_(Ig8#U^a69Tw)D6@P)D6@P)D6^4)J@b))J@b))GgF4)GgF4)GgF))NRyl)NRyl)Lp2%P<Ns3LfwVB'
        b'8+AA8Zq(hVyHWR`?m^vyx(9U+>OQFZpzedZ59$J}39u%>`g#byFa%#1f-elg7lz>L9)f>EIRVxLSQB7PfHgyfA=ZRg6Jkw>HB*H_'
        b')&yA-WKEDYONGBaH!S@ceBsyN3v=+LIrzHg;6E3z(O>@*P2Gd=p9^?pL#_$ACghrsYi?~b=$fEwg02a==22nTHDT9$w6B5J1YQ$('
        b'P2d&4D}Yx3uK->FyaIRy@Cr0r0lNZr1?&pg6==2sbOq=N&=sI7&};?d3dj|ZD<D^(*$OmUfo3bfR)DQQvlVEz0%!%Atw6ICAS*yt'
        b'fUE#n0kQ(kRzR$PSOKvDVg;J509XOA0?k&ytAJMluL52ryh?bLIy+2&l>jRNR!+PbCd5jJg<<%@FnnPczBCM98ip?n!<UBP3&ZfG'
        b'--NGw8vb)ZTMf7pa3$bMz?Fb2TiXn|5^^QvO30OvD<M}xu7q3(xe{_E<VwhukSifqLau~d3Aqw-CFDxTm5?hTS3<6YTm`ubauqtQ'
        b'0$c^S3UC!Vt%6$xw+fwBfvo~tg-)x`X%)~abXo<o3Y}J=(<*dY1+WTW6~HQhRRF8dX%)OGogfy#Du7i0s{mF3tO8gCunJ&Z<M1CF'
        b'2C)ia6~rotRY!$ER-L6gpfb!Vm{l;VU{=AbYasspL>p)|&}yL7K&ydP1FZ&H4YXR#Rzt08BL4FZsH|qI;a0<~hFh&(s{vO7t_EBU'
        b'xEgRZ;A+6tfU5ylqtR-()o`ofR>Q4^TMf4wZZ+I$G+GU|8f-P#YOvL4v>IwP8m$Ie4YV3)HPC9H)j(^Z(ONKT!K?+d7R*{OYr(7q'
        b'vlh%+XtWl{T4=PE&JPzltp%|b#99z*q0?FbYXPhUuol2t`ZRt)tOc=_&J7pHTI#hH%vz22r3JJW&{{xi0j&kJ7SLKiYw4r-1+~`J'
        b'7K5z?wiei0U~7S`1-2I0T3~B|tp&C=*xF!ggRKp=HrU!=YlE!~wl>(>U~7Y|4YoEKtqrv{)Y?#ML#++9Hq_csYeTIKwKm$U4YW4U'
        b'+Gw*j%-U$PHptp&vo^%q5NkuM4Y4-F+7N3)tc^Bn1FVfUYs0GzuQt5e@M^=W4X^ftGnNgoHo)2d>*(b0K&%6?4#YYT>p-jnu@1yK'
        b'5bHoJjKvqm;tM|tU-(h@!ccr+D8BHc@P(Q9!f(RYJre)1VKD1d=nkk1w2sCej#{k)whq`jVC#Ub1GWy>I$-O7tpm0W*g9bAfUN_z'
        b'4%j+i>wv8Twhq`jVC#UbgEs3xtpl|V)OyfnJwWRLS`W~AfYt-F9-#FAtp{j5K<h!9^}wtLW<4<LfmsjCdSKQAvmTiBpv`(f)`K?d'
        b'fmjd3deCM)XtN%8^}wqKUOn*YfmaW_df?RquO4{yz^eydJ@D#*R}Z{;;MJoK;U9qY04z+zmwp$%FcM!Fi7)&veBBfAcY(%8eBUGS'
        b'pLZzSVd)O%>pM(T{!*>ROnhS|zVDg%k0%r;94H(p94H*v;6&X--9+6)-9+6&-9p_$-9p_$-A3I;-A3I;-A3Jox(js|>Mqn>sJl^j'
        b'qwYrCjk+6k59%J&J*fMuTJOJ?gSrpuKB)Vk?t|@pzOFk#*4IPvjiLC)P<&%3zA+Tv7>aNFGJInuzA+Qun2B%9#5aB!zA+Nt7>RF;'
        b'#5YFb8@~(Rn22vo#5X458x!%3--U1dE_~y6;Tz-djdA$aIDF%0;TzNNjbDXt4Z}Bv;Tyy7jbZr4Fnr&`@E?Ui)&yA-WKEDYLDmFW'
        b'U(dofX5kyZ3f~xoZ;Zk>M&TQy@QqRU#wdJa6uvPE-}fl|`+x<^3YZlzD_~Z@tbkbovjS!X%nFzlFe_kIz^s5-0kZ;T1<VSV6)-Dc'
        b'R=}))S%DraKvsaP09k<^D<D=ttbkYnu>w6-0IUF50k8sK1;7e`6#y#$RsgI3SOKsCU<JTRfRz9%0agO6M30s5D&bX5ya!*S@Sh8)'
        b'46zboCB#aIl@JSq@Qp$E#;?LReigp&G5GiGAS*#uf~*8t39=GoCCEyUl^`oYR)VYqSqZWdWF^Q-kd+`SK~{pS1X&5P5@aRFN|2Qx'
        b'D?wI*tOQvJvJzw!$SROkAge%Dfvf^q1+oe)R-wfzfK>pi09FC40$2sG3N2Q_tAbYruL@ojyt;<qUw#(6DtJ}!s^C?@tAbYruL@oj'
        b'yefEA@T%Zd!K;E-1+NNT6}&2VRq(3dRl%!*R|T&MUe$w>-~w3nEZyPj$_=p^V)a0?(;%xs)-?kE@jx)E)n>KYtk$RR4YnF=b*MDl'
        b'YPi*KtKn9|t%h3-w;FCW`mBaq4YwL@HQZ{r)o`ofR>Q4^TMf4wZZ+I$xYcm0;a0<~hFcA{8g4b*YPi*KtKn9|twx{KV5`w*HPl*A'
        b'YeB7rK5GH31+*6WtOc_c`mBXMYe}pHu@=NyXtY)=6~GoltOc<a#99z*rM4JktqR=%m0{L`Sqo;ZR$-vEfYt(93urBU1izrxf?C%Q'
        b'{Ko@<tp&Cg*jiv~fvp9$7T8)~Yk{o=wiei0U~7S`1-2I0+F)ygtqry|*xF!ggRKp=HrU!=Yvc5AL#++9Hq_c^vo_G$Kx+f74YW4U'
        b'+Gw*j%-U$PHpto_YlEx}vNqbR4Y4-Fx@O>CZfuCP(PnLcwE@;fo3-K9hF2S2ZFsfe)rMCaUTt`_;njv$8(wXAwc*u<S6h==2Vfn5'
        b'bpX}@SO;JofOP=Y0ayoM9e{NJ)&W=tU>$&UO~8NbXoz(n*2&TXp)$xiAnT~jIxy?NtOK(S%sMdZXwu;Tt%Ek}0IdVG@T2gJ5%|Uk'
        b'd}9Q@F#_Khfp3h!H%8zaBk+w8_{Io)V+6i20^b;cZ;Ze<M&KJG@Qo4p#t3}>e|6nUa^pr2K*1{wy6b!Yi+SgO5X>zYni58uM3H{c'
        b'MgpHm;1daaB7sjN@QDOIk-#Ss_(TGqNZ=C*d?JBQB=Cs@K9RsD68J;{pGe>n349`fPbBd9PT>Fk1747|AZtO^f~>`wwGe9|)<Ud>'
        b'Sc^4l0oDSn1z4*zYa!M`tc6$$u@+)2#KN)g`HtZKegVi@khLIdLDug>;R^+P;ZXR}q41>wzEHr|yMX_E!GqC{M*n21a4viyfiEQR'
        b'g#^Bkz}Gv0|9qntgD(bO489nAal?VRfw_UXfw_UXiMffniMffniMfTjg}H^fg}H^fjk%4vjk%4vjk$xlgSmsbgSmsbi@A%ri@A%r'
        b'K<fco577GEz!w_$LIYoD;0q0Wp@A<n@P!7x(7+cO_(B6;Xy6MCe4&9aH1LH6zR<uI8u&s3UufV94Sb=2FEsFl2ENe1*SmrLT*utR'
        b'gW&;KXyEJJz<*yTy?}d$S*YMkXTldU_(BF>$lwbZd?AA`WbpOQ;6L9e*m}U$1GXNp^?<GO4;#;cTL*3(xOL#xfm;V|9k_Mi)`43G'
        b'ZXLLF;MRd#2W}m>b>P;4TL*3(xOL#xfm;V|9k_Mi)`43GZXLLF;MT#Ob->oao^`Ni9qd^LW*wMyuxB0YSqEYrh;<;=fmjD(9f)<X'
        b'XB~ib0M-Fm2Vfn5b+Kn%cy-~`g;y6|U3hij)rD6VUg1ReLIz*R;0qaiA%ibu@b%8%Ki4rAVBs|QLIz*R;0qaiA%ibu@P!P%kii!+'
        b'_(BF>$lwbZd?AA`WblOyzL3EeGWbFUU&!DK8GIpwFJ$nA48D-T7c%%l24BeF3mJSNgD+(8g$%yl8T_B$^MF_Zu>xWR#0rQN5Gx>7'
        b'K&*gRfekAFRsgI3SaASHK&*gR0kHyN1y-y8SOKsCU<JSmfE55M09F9309XOA0$>He3V;;=EA+lFAXY%E(EGvwSpl*FWCh3ykQE>c'
        b'XTcXz_(BR_O5qDBeBm(oLJ40w48D-U7Y>6jl<<WTzHk_PA%ri5@P*Uh>)pZsy$-e#Y$e!A>{$u55^5#XN~o1kE1_0Gt%O<$wGwJ2'
        b')Jmw8P%EKULal^a3AGYxCDcl+SqZcfXeHLHgjor*5@sdLN|==}E3sxJ$V!lvShEshCB#aIRamnMU=_eBfK>pi09FC4!kSg^s^C?@'
        b'tAbYruL@ojyefEA@T%Zd!K;E-1+S367ZUhF0$)hrO9#Oh4uUTv@P!1vbPjx>fiIi`Ur68!34Gxk_(B0+DBuePe4&6Z6!3)tzTO4='
        b'=Pw9m70fD_RWPeyR>7=-Sp~BiW;M)enAI?=VOGPehFJ}>8fG=jYM9k9t6^5dtcF<)vl?bK%xaj`Fzc1T|M|Ub>{t!58e}!dYV24I'
        b'u^M7E#A@tV4X_$uHNa|s)c~sjRs*aCSPifmU^T#M>{t!28eTQLYIxP~s^L|`tA<wtuLfQXyc&2l@M_@Iz^j2*1Fr^N;Sl&j0$)hr'
        b'3kiJT5com?U+)6`^9@0)fmj2vUIqMT0LU7UH6Uw1)_|-5Sp%{LWDUp~kToD{K-Pe)0a*jG24oG$8jv+0Ye3e3tN~dAvIb-g$QqC}'
        b'AZtL@fUE^s3$hkuEy!AswIFLj)`F}BSqrijE7n4+g;)!*7Gf>LTC7+Luohq~z*@X5T)iz^fVBW?0oDSn1y~EP7GN#FT7b0xYXQ~*'
        b'tOZyLuohq~z*>N{0BZr(0;~mCtIysSVlBj4h_w)FSA&0&)#%?F{d?m8_<rZ_pX-?WyMAxf?~McC8}WPV0Qg4z-tYST`vsvFF#6!0'
        b'FD742zPRU$!54!Ag9Crpfw_UXfw`$4a$;~|aAI&`aN&jva|?3|a|?4Da~pFTa~pFTa|d$=a|d$=a|d%5a~E?La~E@=)&sR3sP#at'
        b'2WmY~>vsU(2;dt5d?SEw1n`Xjz7fDT0{BJ%-w5FQ9l-zn2mHH$Zxry20=`keHwySh0pBR#`(40)ZXob@cx2X-M=xM@YBtn*pw<Jm'
        b'5W%-1_<l$5pBDhv1GpZ*^=jb112h?QJ)^&F*!94!2X;NM>sbv3UJvkkfY$+D2Y4Oeb%56aUI%y`;B|o40bU1q9pH6<*8yG!cpcz%'
        b'ux=gLbzs+lT?cj@*mYpnfn5jN*1@**yMk|2@Qn(-QNcGV_(lcasNfqFe4~PIRPc=ozEQz9D)>eP->Bdl6?~(DZ&dJ&3cgXnH!Ao>'
        b'1>f%q{x4()U>$&U0oMHoDKEsj5bOTK!{LIg3$kz^eCt5?#)0sS624KwH%j<^m++s#Q0qdit2c=kY+bN*!PW&^7i?Xyb-~sJTNi9y'
        b'?OPXaUAT4O)`eRaZe6%_;nsy)7j9j+b>Y^9TNiF!xOL&yg<BVHUAT4O)`eRad)5V87i<OC3hY?{wE}7d_N)L}0ki^pR$$KxkQE>+'
        b'KvsaPz@8NlD<D>2&kBGQ04o4i0IUF50k8sK1;7e`6#^>&RsgI3SOKsCU<JSmfE55M09F93Xgq}jVg<wsh!qekAXY%EfLH;s0%8Tk'
        b'3WyaDD<M`wEF1~n2;mzcd?SQ!gz)_i;Xl_g7i1;KN|2QxD?wI*tOQvJvJzw^$V!lvAS*#uf~*8t39=GoCEgk)#7c;j5Gx^8Lac;X'
        b'39%AlCB#aIl~}P7U?spxfRz9%0agO61Xu~M5@033N`RFBs{mF3tip;_@T%Zd!K;E-1+NNT6}&2VRq(3dRl%!*R|T(1Zwd=w6~HQh'
        b'Re?{#QNs7Tg#W&P(J-rER%y*DpjG-D9xZ&oTlnt__&LL^f?K7};R|pT;3}<J1-S}x733<&RgkM7S3$0VTm`ubauwt%$kmXmAy-4L'
        b'hFlG~8gez{YRJ`)t07lIu7+F<xf*gc<Z8&(*t8mOHQ;K%)!4Ken^t4fYN*vvtD#n7(`umA*t8m(R%6p@h}96QAy#A4YJk-MtFdV{'
        b'ylQyW@T&En4I5xJz-oZi0ILC3>r?oKSPiiTVhzL^h&BK4I2@2QAZtL@=tKC?hwuZm255~^t$|vjPvLPad?Si)91GuQ;u}qTqls^v'
        b'3g1ZL`<=vpZUAr%;2OX+TDJyr4dfcgHIQo{*Fdg;Tm!iVat-7f$Tg5_AlE>y!KO8UYXH{(t^r&FxCWcnz^#E>3%6cT{GZ>eg<FeF'
        b'Yr)ontp!_)O>41fEznw^wb-;4W-T_Y#iq3oYa!M`tc6$$u@+)2#9C}x3$PYoEx=lAS_`ihUM;*@y*XTfwE$}Y)&i^rSPQTgU@gE}'
        b'fVBW?0oDSn1z0<H1~0@~h_w)FA=d9Ee$d1Zn)pEzKRObA5XH|sivL{4;0J?$k~KIGevrhElK8=i@Pi_LP{hx>i2n?@xZ{h#7lSVb'
        b'2ktm9H!wFaH!wFbH!(LcH!(Lcw=lObw=lObw=lOcw=uUdw=uUdcQAJ_cQAJ_cQAJ`cQJP{cQF@aJs|4=Sr5qi9mEfU_(2dq2;v7p'
        b'{2+)Q1o49)eh|bDg7`rYKM3MSLHr<y9|ZA(Abt?U4}$nX5I^rA{_~AOtOsH}5bJ?hPiip8dO+3#vL2B2fGouDqZoePG5q%hjD}hd'
        b')Ow)S1GUh?4_f#^3qNS#2QB>IMEF4pKS<#RDf}RXAEfYu6n>Dx4^sF+3O`8U2Pym@g&(BwgA{&{!VgmTK?*-e;Rh-FAcY^K@Pib7'
        b'kirjA_(2LkNZ|)5{2+xNr0|0jevrZsQusj%KS<#RDf}RXAEfYu6n>Dx4^sF+3O`8U2Pym@g&(BwgA{&{!VgmTK?*-e;Rgr84@&rX'
        b'm+*hVy#VV1tP8L%z`6kIK6oTuh;<<rLio{{@PiV5-X;9^3osgJU7&S=)&*J@XkDQ73gJID0JSdEx=`yvtqZj-)VfgXLaht6F4VeE'
        b'>q4yywJy}UQ0qdi3$-rPx=`yvtqZj-)VfgXLaht6F4VeE>q4yywJx@-09paG0%!%$3ZNC(vI1rW%nEE-fh{W_R$$8tfE55M09F93'
        b'09XOA0$>HUtbkVmuL52Lyb5>~@G9U{z^i~)0j~mH1-uG)74RzHRluu&R{^gAUIn}gcop#KmBIggum!-1!ISV6!hc^ldO@?d^uq7E'
        b'PN<boE1_0Gt^9|ZPOz0=3#Y;lYWP77Kd9jcHT<B4AJp)J8h%j24{G>94L_*i2Q~blh9A`MgBpHN!w+isK@C5s;RiMRpoSmR@PitD'
        b'P{R*u_(2UnsNn}S{Gf&()bN8Eeo(^?YWP77Kd9jcHT<B4AJp)J8h%j24{G>94L_*i2Q~blh9A`MgBpHN!w+isK@C5s;RiMRpoX7!'
        b'4gVL!0$2sG3Sbq$Du7i0s{mF3tO8gCunJ&R;8|D@s~}cEtkQ?@1+ofc704=(RUoTCR_QbNf>{N#5W){a_(2Fi2;m1I{2+v%cL@La'
        b'hM-nKt%6zwwF+t#)M}{JP^+O<L#>8d4Ye9-HPmXT)ljRUR%6R*pw&RDfmQ>p23ifY8fZ1pYM|9XtASQy%W9a_Fsor!W6Nrg)gY@u'
        b'R)eetSq-upTUJA?hFFa)s{vL6tOi&Ouo_@BwycI%4X+wrHN0wg)$nTI)%?TOG5<m@6#gVkhFGr<{xbk%4agdO20t)sVAjB_fms8y'
        b'24)S+8kjXOYhc#Etbtjh+F^j!0If-U;4x5Zpw>XGfm#E#25Jq|8mKi;YoOLZt$|ttwFYVp)EcNYP-~#pK&^pV1GNTf4b&Q_wNPuZ'
        b'Wi7U>g;@);7G^EBtOZ#MvKCv`Lac>Y3$YerEyP-gwGe9|)<Ud>SPQWhVlB3;1y~EP7F*WBtA$q!uNGb{yjpm*1)&#UEx=lUwT*}H'
        b'Lac>Y+js~s$XbxK`WSv;*21iXS*s7>7icZe`W?dWJA{7^*mnv4eIf8CmA(L@f4a3u;TI|VzEk+m;0J>b1|JMQ82l5iMGC(-9DY&4'
        b'FG~1D3BM@e7bX0ngkO~KixPfO!Y@krMG3zs;rCs_e}2G;!G*zv!G*zv8!pW4U;lr$MGC)2;TI|VB86Y1@QW0Fk-{%h_(cl8NZ}VL'
        b'{33;4r0|Orev!g2Qusv*zewR1Df}XZU!?Gh6n>GyFH-nL3cpC<7b*NAh2M7y|Mwr*1F;^6^+2o#Vm%P+fmjd3dLY&Vu^x!^K&%I1'
        b'JrL`62)_v77a{z<L-@}R24bOv-**ZB_X`@&{|9D0FzbO?56pUn@Snjz>j7HN=$4_@1GOHg^+2r$YCTZvK&=C{4%9kO>p-mowGPxe'
        b'Q0qXg1GNs+I#BCCtpl|V)~o}x4$wM4>j13-v<}cZK<fal1GEm%Iza0Htpl_U&^kct0Ih>9>s7-4`FDD-Wu3xP>30ghNZ}VL{33;4'
        b'r0|Orev!g2Qusv*zewR1Df}XZU!?Gh6n=3i{Gx<k9SXni6#n}{;ZMS3kaZtThFKS8U6^$bo=O*JU7&?S;rE@we_jAvI2C>o!!Kg^'
        b'#i{U%7JlC?{AY0J#|*hH<hqdSLaqzBF66q9>z01XpzC7Yx}fV~+q#hJLaqzBF66q9>q4#zxh~|okn2LO3%M@jx{&Kat_!&Was^hc'
        b'z^WB+E8teZt-z`kU@O2@fUUr)6;LanR$$c%tXcuH0;^VltN>X7vI1lUR;|FQ6#y#$RsgI3SOKsCU<JSmtXcuD0$v5Y3V0RpD&SSX'
        b'tAJMluL52Lyb5>~@Cr5j;#l~_vG9vy;TOlkFG~1D3BT_W{_i)M5Gx@TQusv*zewR%Df}XZ-**cC`2wJoKr4Y(0<8pE3A7SuCD3}c'
        b'@ShulS_!ohY9-W4sFhGFp;khzgjxx;5^5#XN~o1kE1_0Gt%O<$wGwJ2)Jmw8P%EKULal^ai7o3D!~gv~1X>BS5@;pRDxg)^vI=Gu'
        b'%qnbI1+ofc6}GH`ScNUC09FC40$2sG3Sbq$Du7i0tFUDiyefEAdSh4ss{mF3tO8g#7k+Ur{Gx?lwD5}-e$m1&TKGi^ziQzZE&QT|'
        b'-**fDxsJggt3XzPtO8jDvI=As_N)R~1+ofc707Ck)gY@uR)eetSq-upWHrcYkkufoK~{sT23ZZV8e}!dYLL|+t3g(StOi+)6{{gu'
        b'L#&2a4Y3+4Rs*aCSPifmU^T#MfYku2v0^p6YIxP~s^L|`tA<w%uNq!8ylQyW@T%ce@7>*B*Av$p*Z1f3`S;-Z|8xI7VCG-q3mVrk'
        b'_`%@k;(Fp6dRi}F^1<YzTRs|naL)&mFD75y^Tprw)h%BP4h#+q4h#<5a4f!|#Nfo>#Nfo>#0?kb7UmY_7UmY_Hs&_wHs&_wHs%iI'
        b'4(1N#4(1N#F6M%*0b2vM25c?Z+J6_<6W1HpF&Anr)Y=DgAIybW3$ylM?!jD$wGe9|)?Qe-AZsrML#%~Z3$YerEyUWuT!6IzYXQ~*'
        b'tOZyLuohq~z*>N{0BZr(0;~mC3$PYoEx=lUwE$}ekHQ^1k``p`>Q@?O?N77*136C-)B'
    ),
    'weather_example.csv': (
        b'c-kw;%uS8Y&?!pH%!|*>jnCjR&@nRLGSo3O;4;!NG%(;Y1~E-^47dOfrwU8'
    ),
}

@lru_cache(maxsize=8)
def bundled_bytes(name):
    """Lossless embedded originals; no external data folder or writes needed."""
    return zlib.decompress(base64.b85decode(_BUNDLED_CSV[name]))



"""Eco-Rain repaired Streamlit application. Run: python -m streamlit run app.py"""
from pathlib import Path
import io
import json
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
FILES = ["DS0004.CSV", "4.CSV", "3.CSV", "2.CSV", "DS0001(2).CSV", "DS0001.CSV"]
st.set_page_config(page_title="Eco-Rain", layout="wide")
lang = st.sidebar.selectbox("Language / 語言 / 言語", ["繁體中文", "English", "日本語"], key="language")
LANG_INDEX = ["繁體中文", "English", "日本語"].index(lang)


def tr(zh, en, ja):
    return (zh, en, ja)[LANG_INDEX]


def chart(fig, x_label, y_label, title=None, height=350):
    fig.update_layout(title=title, height=height, xaxis_title=x_label, yaxis_title=y_label,
        margin=dict(l=15,r=15,t=45 if title else 15,b=15), template="plotly_white",
        legend=dict(orientation="h",y=1.12,x=0), hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def line(x, y, name, color="#137D86"):
    # Downsample only the display; calculations and downloads use every sample.
    # Preserve each block's minimum and maximum to retain narrow impact peaks.
    if len(x) > 18000:
        block = int(np.ceil(len(x)/8000))
        indices = [0, len(x)-1]
        for start in range(0,len(x),block):
            window = y[start:start+block]
            indices.extend([start+int(np.argmin(window)),start+int(np.argmax(window))])
        indices = np.unique(indices)
        x,y = x[indices],y[indices]
    return go.Scatter(x=x,y=y,name=name,mode="lines",line=dict(color=color,width=1.5))


@st.cache_data(show_spinner=False)
def parse_trace(blob, name):
    return read_scope_csv(blob,name)


@st.cache_data(show_spinner=False)
def event_csv(result):
    return pd.DataFrame({key:result[key] for key in ["time_s","diameter_mm","mass_kg",
        "speed_m_s","incident_momentum_Ns","incident_kinetic_energy_J"]}).to_csv(index=False).encode("utf-8-sig")


st.title(tr("Eco-Rain：撞擊響應與降雨模擬", "Eco-Rain: Impact Response & Rainfall Simulation", "Eco-Rain：衝突応答と降雨シミュレーション"))
st.caption(tr("使用六份乾燥狀態示波器資料。以實測波形建立線性疊加模型。",
    "Six dry-state oscilloscope records. A linear superposition model built from measured waveforms.",
    "乾燥状態のオシロスコープ記録6件を使用し、実測波形の線形重ね合わせを計算します。"))
st.sidebar.info("TE Connectivity LDT0-028K / PVDF")
source_name = st.sidebar.selectbox(tr("參考實測檔", "Reference recording", "参照する実測ファイル"), FILES, key="source_name")
uploaded = st.sidebar.file_uploader(tr("改用自己的乾燥示波器 CSV", "Use another dry-state scope CSV", "別の乾燥状態CSVを使用"), type=["csv"],key="scope_upload")
area_cm2 = st.sidebar.number_input(tr("水平受雨面積（cm²）", "Horizontal catch area (cm²)", "水平受雨面積（cm²）"), .1, 100., 2.5, .1,key="area")
with st.sidebar.expander(tr("波形擷取設定", "Waveform extraction", "波形の抽出設定")):
    threshold = st.number_input(tr("對齊門檻（V）", "Alignment threshold (V)", "整列しきい値（V）"), .01, 100., 1., .1, key="threshold")
    tail = st.number_input(tr("門檻後擷取長度（s）", "Post-threshold length (s)", "しきい値後の長さ（s）"), .05, .30, .25, .01,key="tail")
    st.caption(tr("保留門檻前 5 ms；最後 20 ms 平滑收尾。t = 0 為擷取起點，不代表接觸時刻。",
        "Includes 5 ms before the threshold and a final 20 ms taper. t = 0 marks the extracted start, not contact time.",
        "しきい値の5 ms前から抽出し、末尾20 msにテーパーを適用。t = 0は抽出開始点です。"))
known_load = st.sidebar.checkbox(tr("此波形量自已知純電阻負載兩端", "Recorded across a known resistive load", "既知の抵抗負荷の両端電圧を測定"),key="known_load")
resistance = None
if known_load:
    resistance = st.sidebar.number_input(tr("量測時的負載電阻（Ω）", "Load resistance during measurement (Ω)", "測定時の負荷抵抗（Ω）"),1.,1e12,1e6,1000.,format="%.0f",key="resistance")
    st.sidebar.caption(tr("此值只用於 V²/R 積分；改變負載後須重新量測響應。", "Used for V²/R integration only; a different load requires a new recording.", "V²/R積分にのみ使用。負荷を変えた場合は再測定が必要です。"))
try:
    blob = uploaded.getvalue() if uploaded is not None else bundled_bytes(source_name)
    name = uploaded.name if uploaded is not None else source_name
    trace = parse_trace(blob,name)
    kt,kv,kinfo = prepare_kernel(trace,threshold_V=threshold,tail_s=tail)
except (ValueError, OSError, UnicodeError) as exc:
    st.error(tr("無法載入波形：", "Cannot load waveform: ", "波形を読み込めません：")+str(exc))
    st.stop()

tabs = st.tabs([tr("實測資料", "Measured Data", "実測データ"),
    tr("連續撞擊", "Repeated Impacts", "連続衝突"),
    tr("降雨模擬", "Rainfall Simulation", "降雨シミュレーション"),
    tr("模型與文獻", "Model & Sources", "モデルと文献"),
    tr("Model 1｜舊假設", "Model 1 | Legacy", "Model 1｜旧仮定"),
    tr("Model 2｜模型優化", "Model 2 | Improved", "Model 2｜改善")])

with tabs[0]:
    st.subheader(tr("乾燥狀態單次撞擊", "Dry-state single impact", "乾燥状態の単発衝突"))
    a,b,c,d = st.columns(4)
    a.metric(tr("最高電壓", "Maximum voltage", "最大電圧"),f"{trace.voltage_V.max():.2f} V")
    b.metric(tr("最低電壓", "Minimum voltage", "最小電圧"),f"{trace.voltage_V.min():.2f} V")
    c.metric(tr("取樣率", "Sample rate", "サンプリング周波数"),f"{1/trace.dt/1000:.1f} kHz")
    d.metric(tr("資料點數", "Samples", "サンプル数"),f"{len(trace.time_s):,}")
    fig=go.Figure(line(trace.time_s,trace.voltage_V,name))
    chart(fig,tr("CSV 時間（s）","CSV time (s)","CSV時刻（s）"),tr("電壓（V）","Voltage (V)","電圧（V）"))
    fig=go.Figure(line(kt*1000,kv,name))
    chart(fig,tr("距擷取起點（ms）","Time from extracted start (ms)","抽出開始からの時間（ms）"),tr("電壓（V）","Voltage (V)","電圧（V）"),tr("用於模擬的單次響應", "Single-event response used by the model", "モデルで使用する単発応答"))
    st.caption(tr("CSV 電壓值保留原倍率。六份內建檔均已由使用者確認為乾燥；撞擊力與機械阻尼尚未由這些電壓資料校正。",
        "CSV voltage scaling is retained. All six bundled files are confirmed dry; impact force and mechanical damping are not identified from these voltages.",
        "CSVの電圧倍率を維持。内蔵6件はすべて乾燥状態です。衝突力と機械減衰はこれらの電圧から同定していません。"))
    rows=[]
    for filename in FILES:
        item=parse_trace(bundled_bytes(filename),filename)
        rows.append({"file":filename,"state":"dry","samples":len(item.time_s),"sample_rate_Hz":round(1/item.dt),
                     "Vmax_V":float(item.voltage_V.max()),"Vmin_V":float(item.voltage_V.min())})
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
    st.download_button(tr("下載擷取波形 CSV", "Download extracted response CSV", "抽出波形CSVをダウンロード"),
        pd.DataFrame({"time_s":kt,"voltage_V":kv}).to_csv(index=False).encode("utf-8-sig"),"dry_response.csv","text/csv",key="download_kernel")

with tabs[1]:
    st.subheader(tr("保留完整尾波的連續撞擊", "Repeated impacts with overlapping responses", "減衰尾部を保持する連続衝突"))
    st.info(tr("此模式假設每次撞擊具有相同響應形狀，並可線性疊加。實測參考檔的撞擊強度未知，結果屬相同條件下的模型情境。",
        "This mode assumes each impact has the same response shape and responses add linearly. The reference impact strength is unknown, so results describe this assumed repeat-impact scenario.",
        "各衝突が同じ応答形状を持ち、線形に加算できると仮定します。参照衝突の強さは不明なため、同一条件を仮定したモデル計算です。"))
    a,b,c=st.columns(3)
    frequency=a.number_input(tr("撞擊頻率（Hz）","Impact frequency (Hz)","衝突周波数（Hz）"),0.,200.,20.,.1,key="impact_frequency")
    duration=b.number_input(tr("撞擊持續時間（s）","Impact-train duration (s)","衝突列の持続時間（s）"),.05,30.,2.,.1,key="impact_duration")
    gain=c.number_input(tr("相對振幅（假設）","Relative amplitude (assumption)","相対振幅（仮定）"),0.,10.,1.,.1,key="impact_gain")
    events=periodic_events(frequency,duration)
    tt,vv=superpose(kt,kv,events,np.full(len(events),gain),duration+kt[-1],trace.dt)
    integral=squared_signal_integral(tt,vv)
    rms=float(np.sqrt(integral[-1]/(tt[-1]-tt[0])))
    a,b,c=st.columns(3)
    a.metric(tr("撞擊次數","Impacts","衝突回数"),f"{len(events):,}")
    b.metric(tr("模型絕對峰值","Model absolute peak","モデル絶対ピーク"),f"{np.max(np.abs(vv)):.3f} V")
    c.metric("RMS",f"{rms:.3f} V")
    fig=go.Figure(line(tt,vv,tr("疊加電壓","Superposed voltage","重ね合わせ電圧")))
    chart(fig,tr("時間（s）","Time (s)","時間（s）"),tr("模型電壓（V）","Model voltage (V)","モデル電圧（V）"))
    if resistance is not None:
        energy=resistive_energy(tt,vv,resistance)
        st.metric(tr("已知負載的模型耗能（含尾波）","Model energy in known load, including tail","既知負荷のモデル消費エネルギー（尾部を含む）"),f"{energy[-1]*1000:.6f} mJ")
    else:
        st.metric(tr("電壓平方積分（非電能）","Squared-voltage integral (not energy)","電圧二乗積分（電気エネルギーではありません）"),f"{integral[-1]:.5f} V²·s")
    st.caption(tr("下一次撞擊不會刪除前一次波形。最後一擊之後仍計算完整尾波。此版本不預設 33.3 Hz 最佳，也不假定行程隨頻率的衰減公式。",
        "New impacts retain the existing response; the final tail is included. No optimum at 33.3 Hz or force-versus-frequency law is imposed.",
        "次の衝突でも前の応答を保持し、最後の尾部まで計算します。33.3 Hzの最適性や力の周波数依存則は仮定しません。"))
    st.download_button(tr("下載連續撞擊 CSV","Download repeated-impact CSV","連続衝突CSVをダウンロード"),
        pd.DataFrame({"time_s":tt,"model_voltage_V":vv,"cumulative_V2_s":integral}).to_csv(index=False).encode("utf-8-sig"),"repeated_impacts.csv","text/csv",key="download_impacts")

with tabs[2]:
    st.subheader(tr("降雨事件與乾燥響應情境", "Rainfall events and a dry-response scenario", "降雨イベントと乾燥応答の想定"))
    st.caption(tr("長時間計算保留雨滴事件與累積量；電壓僅計算選取的短時間片段。這個情境假設裝置維持乾燥響應，不模擬積水演化。",
        "Long scenarios store drop events and cumulative quantities; waveforms use a short selected window. The response is assumed to remain dry, without water-film evolution.",
        "長時間計算は雨滴イベントと累積量を保持し、波形は選択した短時間区間のみ計算。応答は乾燥状態のままと仮定し、水膜の変化は扱いません。"))
    mode=st.radio(tr("雨勢來源","Rainfall input","降雨入力"),[0,1,2],format_func=lambda n:[tr("固定雨強","Constant rain","一定降雨"),tr("合成雨勢","Synthetic storm","合成降雨"),tr("氣象 CSV","Weather CSV","気象CSV")][n],horizontal=True,key="rain_mode")
    weather=st.file_uploader(tr("上傳氣象 CSV","Upload weather CSV","気象CSVをアップロード"),type=["csv"],key="weather_upload") if mode==2 else None
    if mode==2:
        st.caption(tr("接受 Time,Rain（小時、mm/h）、time_h,rain_mm_h 或 time_s,rain_mm_h。最後一列時間是結束邊界。",
            "Accepts Time,Rain (hours, mm/h), time_h,rain_mm_h or time_s,rain_mm_h. The final row sets the end boundary.",
            "Time,Rain（時間、mm/h）、time_h,rain_mm_h、time_s,rain_mm_hに対応。最終行の時刻は終了境界です。"))
        st.download_button(tr("下載氣象範例","Download weather example","気象CSVの例"),bundled_bytes("weather_example.csv"),"weather_example.csv","text/csv",key="weather_example")
    with st.form("rain_parameters"):
        a,b,c=st.columns(3)
        hours=a.number_input(tr("時長（小時）","Duration (hours)","時間（h）"),.01,24.,16.,.5,key="rain_hours",disabled=mode==2)
        rate=b.number_input(tr("雨強／合成雨勢峰值（mm/h）","Rain rate / storm peak (mm/h)","降雨強度／合成ピーク（mm/h）"),0.,300.,50.,5.,key="rain_rate",disabled=mode==2)
        seed=c.number_input(tr("隨機種子","Random seed","乱数シード"),0,2**31-1,42,1,key="seed")
        submitted=st.form_submit_button(tr("執行降雨模擬","Run rainfall simulation","降雨シミュレーションを実行"),disabled=mode==2 and weather is None)
    if submitted:
        try:
            if mode==2:
                bounds,rates=read_weather_csv(weather.getvalue())
            elif mode==0:
                bounds=np.array([0.,hours*3600]); rates=np.array([rate])
            else:
                bounds=np.linspace(0.,hours*3600,max(2,int(np.ceil(hours*6)))+1)
                mid=(bounds[:-1]+bounds[1:])/2
                rates=rate*np.exp(-.5*((mid-hours*1800)/(hours*900))**2)
            result=rain_events(bounds,rates,area_cm2*1e-4,int(seed))
            st.session_state["rain_result"]={"result":result,"bounds":bounds,"rates":rates,
                "area_cm2":area_cm2,"mode":mode,"seed":int(seed)}
        except (ValueError,OverflowError,UnicodeError) as exc:
            st.session_state.pop("rain_result",None)
            st.error(tr("模擬輸入錯誤：","Simulation input error: ","入力エラー：")+str(exc))
    saved=st.session_state.get("rain_result")
    if saved is not None:
        result,bounds,rates=saved["result"],saved["bounds"],saved["rates"]
        st.caption(tr("本次結果：","Result settings: ","今回の結果：")+f'{saved["area_cm2"]:.2f} cm²; {result["duration_s"]/3600:.3f} h; seed {saved["seed"]}')
        if saved["area_cm2"] != area_cm2 or saved["mode"] != mode:
            st.info(tr("設定已變更，請重新執行以更新這份結果。","Settings changed; run again to update these results.","設定が変更されました。再実行して更新してください。"))
        a,b,c=st.columns(3)
        a.metric(tr("雨滴數量","Drop count","雨滴数"),f'{len(result["time_s"]):,}')
        b.metric(tr("實際抽樣落水量","Sampled water volume","抽出された水量"),f'{result["realized_volume_m3"]*1e6:.3f} mL')
        c.metric(tr("入射動能（不是發電量）","Incident kinetic energy (not electricity)","入射運動エネルギー（発電量ではありません）"),f'{result["incident_kinetic_energy_J"].sum()*1000:.3f} mJ')
        st.caption(tr("期望落水量：","Expected water volume: ","期待水量：")+f'{result["target_volume_m3"]*1e6:.3f} mL; '+tr("抽樣平均雨強：","sampled mean rate: ","抽出平均降雨強度：")+f'{result["realized_mean_mm_h"]:.3f} mm/h')
        fig=go.Figure(line(bounds/3600,np.r_[0.,np.cumsum(result["interval_volume_m3"])]*1e6,tr("累積落水量","Cumulative water","累積水量")))
        chart(fig,tr("時間（h）","Time (h)","時間（h）"),tr("水量（mL）","Water volume (mL)","水量（mL）"))
        rain_fig=go.Figure(go.Scatter(x=bounds/3600,y=np.r_[rates,rates[-1]],mode="lines",line_shape="hv",name=tr("輸入雨強","Input rain rate","入力降雨強度")))
        chart(rain_fig,tr("時間（h）","Time (h)","時間（h）"),"mm/h",height=220)
        st.download_button(tr("下載雨滴事件 CSV","Download drop events CSV","雨滴イベントCSVをダウンロード"),event_csv(result),"rain_events.csv","text/csv",key="download_rain")
        st.markdown("---")
        st.markdown("#### "+tr("短時間響應片段","Short response window","短時間応答区間"))
        max_window=min(5.,result["duration_s"])
        # A fraction selector remains valid after changing the scenario length.
        fraction=st.slider(tr("片段位置（全時長百分比）","Window position (% of duration)","区間の位置（全時間に対する%）"),0.,100.,0.,.1,key="preview_position")
        preview_start=(result["duration_s"]-max_window)*fraction/100
        calibration=st.checkbox(tr("提供參考撞擊的有效衝量，試算電壓","Supply reference effective impulse to estimate voltage","参照衝突の有効力積を指定して電圧を試算"),key="impulse_known")
        if calibration:
            reference_j=st.number_input(tr("參考撞擊有效衝量（μN·s）","Reference effective impulse (μN·s)","参照衝突の有効力積（μN·s）"),.001,1e9,10.,1.,format="%.3f",key="reference_impulse")*1e-6
            coupling=st.number_input(tr("雨滴動量傳遞比例（假設）","Drop momentum transfer fraction (assumption)","雨滴運動量の伝達率（仮定）"),0.,1.,1.,.05,key="coupling")
            gains=result["incident_momentum_Ns"]*coupling/reference_j
            response=kv
            unit=tr("假設校正後的模型電壓（V）","Voltage under calibration assumptions (V)","校正仮定に基づくモデル電圧（V）")
            st.caption(tr("須提供與此參考檔對應的衝量，並假設傳力方式相容且振幅與有效衝量成正比；目前並無雨滴電壓驗證。",
                "The impulse must correspond to the selected recording. This assumes compatible loading and voltage proportional to effective impulse; rain-voltage predictions remain unvalidated.",
                "力積は選択した記録に対応する必要があります。荷重伝達の整合性と応答の比例性を仮定し、雨滴電圧予測は未検証です。"))
        else:
            reference_j=float(1000*np.pi/6*(.001)**3*terminal_speed(1.))
            gains=result["incident_momentum_Ns"]/reference_j
            response=kv/np.max(np.abs(kv))
            unit=tr("相對響應（無因次）","Relative response (dimensionless)","相対応答（無次元）")
            st.caption(tr("預設只顯示相對響應：波形峰值規一化，雨滴振幅以 1 mm 雨滴的入射動量為尺度。它不是伏特或發電量。",
                "Default output is dimensionless: normalize the recorded peak and scale drop amplitudes by the incident momentum of a 1 mm drop. It is not voltage or electrical energy.",
                "初期出力は無次元です。実測ピークを規格化し、1 mm雨滴の入射運動量を尺度にします。電圧・発電量ではありません。"))
        pt,pv=superpose(kt,response,result["time_s"],gains,max_window,trace.dt,start_s=preview_start)
        fig=go.Figure(line(pt-preview_start,pv,tr("模型響應","Model response","モデル応答")))
        chart(fig,tr("片段內時間（s）","Time within window (s)","区間内時間（s）"),unit)
        st.caption(tr("片段起始時間：","Window starts at: ","区間開始時刻：")+f"{preview_start:.3f} s")
        if not np.any((result["time_s"]<=pt[-1]) & (result["time_s"]+kt[-1]>=pt[0])):
            st.info(tr("此片段沒有雨滴響應，可移動片段位置。","No drop response in this window; move the window position.","この区間には雨滴応答がありません。位置を変更してください。"))
        if calibration and resistance is not None:
            budget=window_energy_budget(pt,pv,resistance,result["time_s"],result["incident_kinetic_energy_J"],kt[-1])
            if budget["within_incident_upper_bound"]:
                st.metric(tr("僅此片段的負載模型耗能","Model load energy in this window only","この区間のみの負荷モデル消費エネルギー"),f'{budget["load_energy_J"]*1000:.6f} mJ')
            else:
                st.error(tr("此組校正參數使負載耗能超過所有可能貢獻雨滴的入射動能，不符合能量上限。請核對參考衝量、負載與倍率；此結果不可當作發電預測。",
                    "These calibration settings predict more load energy than the total incident kinetic energy of all contributing drops. Check impulse, load and voltage scaling; this is not a valid electricity prediction.",
                    "この校正設定では負荷消費エネルギーが寄与する雨滴の総入射運動エネルギーを超えます。力積・負荷・電圧倍率を確認してください。発電予測として無効です。"))
        signal_key="assumed_voltage_V" if calibration else "relative_response"
        st.download_button(tr("下載片段響應 CSV","Download response window CSV","応答区間CSVをダウンロード"),
            pd.DataFrame({"time_s":pt,signal_key:pv}).to_csv(index=False).encode("utf-8-sig"),"rain_response_window.csv","text/csv",key="download_window")
    else:
        st.info(tr("設定雨勢後按「執行降雨模擬」。相同種子可重現相同雨滴事件。","Choose inputs and run the simulation. The same seed reproduces the same drop events.","入力を設定し実行してください。同じシードで同じ雨滴イベントを再現できます。"))

with tabs[3]:
    st.subheader(tr("計算方式與適用範圍","Calculation and scope","計算方法と適用範囲"))
    st.markdown(tr("**實測響應疊加**：h 是選取的乾燥電壓波形，aᵢ 為相對撞擊振幅；此模型不會在新撞擊時清除舊響應。",
        "**Measured-response superposition:** h is the selected dry voltage trace and aᵢ is a relative amplitude. A new impact retains previous responses.",
        "**実測応答の重ね合わせ：** hは選択した乾燥電圧波形、aᵢは相対振幅。新しい衝突でも既存応答を保持します。"))
    st.latex(r"V(t)=\sum_i a_i h(t-t_i)")
    st.markdown(tr("**降雨抽樣**：採 0.5–6 mm 的截斷 Marshall–Palmer 形狀，以落速加權抽取抵達表面的雨滴。再依平均滴體積、面積與雨強設定 Poisson 到達率，令期望雨量一致；因此不是保留固定 N₀ 的原始模型。",
        "**Rain sampling:** use a truncated 0.5–6 mm Marshall–Palmer shape, weighted by fall speed for surface arrivals. Mean drop volume, area and rain rate set the Poisson arrival rate to match expected rain volume; this is not the original fixed-N₀ model.",
        "**降雨抽出：** 0.5–6 mmに切ったMarshall–Palmer型分布を落下速度で重み付け。平均雨滴体積、面積、降雨強度からPoisson到達率を定め、期待雨量を一致させます。固定N₀の元モデルとは異なります。"))
    st.latex(r"p_{hit}(D)\propto u(D)e^{-\Lambda D},\quad\Lambda=4.1R^{-0.21},\quad\lambda_{hit}=\frac{RA}{3.6\times10^6\,\mathbb{E}[v_{drop}]}")
    st.latex(r"u(D)=9.65-10.3e^{-0.6D}\quad[D\text{ in mm},\ u\text{ in m/s}]")
    st.markdown(tr("**能量定義**：雨滴入射動能使用 ½mu²。只有已知純電阻兩端電壓才以 V²/R 積分；電壓平方積分本身不叫電能。入射動能也不等於傳入薄膜或輸出的電能。",
        "**Energy:** incident drop kinetic energy is ½mu². Electrical dissipation uses V²/R only for voltage across a known resistor. A squared-voltage integral alone is not energy; incident kinetic energy is not transferred beam energy or electrical output.",
        "**エネルギー：** 入射運動エネルギーは½mu²。既知抵抗の両端電圧のみV²/Rで積分します。電圧二乗積分、梁への入力エネルギー、発電量を混同しません。"))
    st.latex(r"E_R=\int\frac{V_R^2(t)}{R_L}\,dt")
    st.markdown(tr("**資料限制**：六份均為乾燥。沒有把單一阻尼比、水膜損失率、100 Hz 共振或 33.3 Hz 最佳頻率寫成已校正結果。改變夾持、撞擊方式或負載，需要對應的參考量測。",
        "**Evidence limits:** all six recordings are dry. No single damping ratio, wet loss, 100 Hz resonance or 33.3 Hz optimum is treated as calibrated. A change in clamp, impact mechanism or load requires corresponding measurements.",
        "**データの範囲：** 6件すべて乾燥状態です。単一減衰比、水膜損失、100 Hz共振、33.3 Hz最適性を校正済みとは扱いません。固定方法、衝突方式、負荷の変更には対応する実測が必要です。"))
    st.markdown("[TE Connectivity: LDT0 datasheet](https://www.te.com/commerce/DocumentDelivery/DDEController?Action=showdoc&DocId=Data+Sheet%7FLDT_with_Crimps%7FA1%7Fpdf%7FEnglish%7FENG_DS_LDT_with_Crimps_A1.pdf%7FCAT-PFS0006)")
    st.markdown("[Rees & Garrett (2021): surface-arrival sampling and precipitation rate](https://amt.copernicus.org/articles/14/7681/2021/)")
    st.caption(tr("版本：4.1 單檔修復版；計算與資料已內建。","Version 4.1 standalone; calculations and data bundled.","バージョン4.1単一ファイル版。計算とデータを内蔵。"))
    settings={"version":"4.1","model":"dry_measured_response_superposition",
        "source_file":name,"source_sha256":hashlib.sha256(blob).hexdigest(),
        "source_metadata":trace.metadata,"dry_state":True,"voltage_scaling":"unchanged_from_csv",
        "kernel":kinfo,"known_load_ohm":resistance,"impact_frequency_hz":frequency,
        "impact_duration_s":duration,"relative_impact_amplitude":gain}
    if st.session_state.get("rain_result") is not None:
        scenario=st.session_state["rain_result"]
        settings["rain"]={"area_cm2":scenario["area_cm2"],"seed":scenario["seed"],
            "boundaries_s":scenario["bounds"].tolist(),"rates_mm_h":scenario["rates"].tolist(),
            "preview_position_percent":st.session_state.get("preview_position",0),
            "impulse_calibration_enabled":st.session_state.get("impulse_known",False),
            "assumed_reference_impulse_micro_Ns":st.session_state.get("reference_impulse"),
            "assumed_momentum_transfer_fraction":st.session_state.get("coupling")}
    st.download_button(tr("下載本次設定 JSON","Download settings JSON","今回の設定JSONをダウンロード"),
        json.dumps(settings,ensure_ascii=False,indent=2).encode("utf-8"),"ecorain_settings.json","application/json",key="download_settings")

with tabs[4]:
    render_water_scenario(tr)

with tabs[5]:
    render_model2(tr)
