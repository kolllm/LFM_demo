
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import windows, find_peaks

st.set_page_config(layout="wide", page_title="LFM加窗、NLFM对比演示")

st.title("LFM加窗、NLFM对比演示")

with st.sidebar:
    st.header("参数设置")

    wave_type = st.radio("波形选择", ["LFM", "NLFM"])

    tau_us = st.slider("脉冲宽度 τ (us)", 1.0, 100.0, 10.0)
    B_MHz = st.slider("带宽 B (MHz)", 1.0, 200.0, 10.0)


    window_type = st.selectbox(
        "脉压窗函数",
        ["Rectangular", "Hamming", "Blackman", "Kaiser"]
    )

    # nlfm_type = st.selectbox(
    #     "NLFM频谱加权",
    #     ["Hanning", "Taylor(近似)", "RaisedCosine"]
    # )

    N = st.selectbox("FFT点数", [2048,4096,8192], index=1)

# tau_us = st.session_state.tau_input
# B_MHz = st.session_state.B_input

tau = tau_us*1e-6
B = B_MHz*1e6

fs = 2*B
dt = 1/fs

t_max = N*dt
f_max = N/t_max

t = np.arange(N)*dt - t_max/2
f = np.arange(N)/t_max - f_max/2

N_tau = max(16, int(np.floor(tau/dt)))

# ======================
# 波形生成（严格参考MATLAB）
# ======================
if wave_type == "LFM":

    K = B/tau
    s_signal = (np.abs(t) <= tau/2) * np.exp(1j*np.pi*K*t**2)

else:

    f_design = np.linspace(-B/2, B/2, N)

    # if nlfm_type == "Hanning":
    win_spec = np.hanning(N)

    # elif nlfm_type == "RaisedCosine":
    #     x = np.linspace(-1,1,N)
    #     win_spec = 0.5*(1+np.cos(np.pi*x))
    #
    # else:
    #     x = np.linspace(-1,1,N)
    #     win_spec = (1-x**2)**2

    win_spec = win_spec/np.max(win_spec)

    S_mag_sq = win_spec**2

    tau_f = np.cumsum(S_mag_sq)
    tau_f = tau*(tau_f-tau_f.min())/(tau_f.max()-tau_f.min()) - tau/2

    inst_freq = np.interp(t, tau_f, f_design, left=0, right=0)

    phase = 2*np.pi*np.cumsum(inst_freq)*dt

    stmp = np.exp(1j*phase)

    s_signal = np.zeros(N, dtype=complex)

    c0 = N//2 - N_tau//2
    c1 = N//2 + N_tau//2

    s_signal[c0:c1] = stmp[c0:c1]

# ======================
# 加窗（严格参考MATLAB）
# ======================
if window_type == "Rectangular":
    sig_r = s_signal.copy()
else:

    w = np.zeros(N)

    if window_type == "Hamming":
        ww = windows.hamming(N_tau)

    elif window_type == "Blackman":
        ww = windows.blackman(N_tau)

    else:
        ww = windows.kaiser(N_tau, beta=6)

    c0 = N//2 - N_tau//2
    c1 = N//2 + N_tau//2

    w[c0:c1] = ww[:c1-c0]

    sig_r = s_signal*w

# ======================
# 频谱
# ======================
S_f = np.fft.fftshift(np.fft.fft(sig_r, N))
spec_db = 20*np.log10(np.abs(S_f)/np.max(np.abs(S_f)) + 1e-12)

# ======================
# 匹配滤波（严格参考MATLAB）
# ======================

# 加窗信号输出
S_sig = np.fft.fft(sig_r, N)
H_mf = np.conj(S_sig)

Y_mf = S_sig * H_mf
y_out = np.fft.fftshift(np.fft.ifft(Y_mf))

# 原始信号参考峰值
S_sig_ref = np.fft.fft(s_signal, N)
H_ref = np.conj(S_sig_ref)

Y_ref = S_sig_ref * H_ref
y_ref = np.fft.fftshift(np.fft.ifft(Y_ref))

y_out_abs = np.abs(y_out) / np.max(np.abs(y_ref))
y_out_db = 20*np.log10(y_out_abs + 1e-12)

# ======================
# 性能指标计算与历史记录逻辑
# ======================

# 1. 初始化 Session State (用于存储历史记录)
if 'history' not in st.session_state:
    st.session_state.history = []


# 2. 计算性能指标的函数 (封装起来以便复用)
def calculate_metrics(y_out_db, t, tau, fs):
    # 限制搜索范围在脉冲宽度内
    search_mask = (t >= -tau / 2) & (t <= tau / 2)
    t_search = t[search_mask]
    y_search = y_out_db[search_mask]

    # 寻找峰值
    min_dist = max(1, round(fs / B / 4)) if B > 0 else 1
    peaks, props = find_peaks(y_search, height=-100, distance=min_dist)

    if len(peaks) == 0:
        return {"Peak Power (dB)": np.nan, "Main Lobe Width (us)": np.nan, "PSLR (dB)": np.nan}

    # --- 峰值功率 ---
    sorted_indices = np.argsort(props['peak_heights'])[::-1]
    main_peak_idx_in_search = peaks[sorted_indices[0]]
    main_peak_val = props['peak_heights'][sorted_indices[0]]
    peak_power = main_peak_val

    # --- 主瓣宽度 (-3dB) ---
    threshold_3dB = main_peak_val - 3
    main_peak_idx_in_search = int(main_peak_idx_in_search)

    # 向左/右搜索
    left_idx = main_peak_idx_in_search
    while left_idx > 0 and y_search[left_idx] > threshold_3dB:
        left_idx -= 1
    right_idx = main_peak_idx_in_search
    while right_idx < len(y_search) - 1 and y_search[right_idx] > threshold_3dB:
        right_idx += 1

    main_lobe_width_us = np.nan
    if left_idx > 0 and right_idx < len(y_search) - 1:
        # 线性插值 (对齐MATLAB)
        try:
            frac_left = (threshold_3dB - y_search[left_idx]) / (y_search[left_idx + 1] - y_search[left_idx])
            frac_right = (threshold_3dB - y_search[right_idx]) / (y_search[right_idx - 1] - y_search[right_idx])
            width_samples = (right_idx - frac_right) - (left_idx + frac_left)
            main_lobe_width_us = width_samples * dt * 1e6
        except:
            pass

    # --- PSLR (旁瓣电平) ---
    pslr_dB = np.nan
    exclude_range = int(main_lobe_width_us * 1e-6 / dt * 1.5) if not np.isnan(main_lobe_width_us) else int(
        fs * tau * 0.1)

    sidelobe_peaks = [props['peak_heights'][i] for i, p in enumerate(peaks) if
                      abs(p - main_peak_idx_in_search) > exclude_range]

    if sidelobe_peaks:
        first_sidelobe = max(sidelobe_peaks)
        pslr_dB = first_sidelobe - main_peak_val

    return {
        "Peak Power (dB)": round(peak_power, 2),
        "Main Lobe Width (us)": round(main_lobe_width_us, 4) if not np.isnan(main_lobe_width_us) else "N/A",
        "PSLR (dB)": round(pslr_dB, 2) if not np.isnan(pslr_dB) else "N/A"
    }


# --- 执行计算 ---
metrics = calculate_metrics(y_out_db, t, tau, fs)

# 3. 界面布局与交互
tab1, tab2, tab3 = st.tabs(["频谱分析", "匹配滤波", "📊 性能指标与记录"])

with tab1:
    # ... (保持原有的 Plotly 频谱绘图代码不变) ...
    fig = go.Figure()
    fig.add_scatter(x=f / 1e6, y=spec_db, mode="lines", name="Spectrum")
    if wave_type == "NLFM":
        fig.update_layout(title=f"{wave_type} 频谱", xaxis_title="频率 (MHz)", yaxis_title="幅度 (dB)", height=600)
    else:
        if window_type == "Rectangular":
            fig.update_layout(title=f"{wave_type} 频谱", xaxis_title="频率 (MHz)", yaxis_title="幅度 (dB)", height=600)
        else:
            fig.update_layout(title=f"{wave_type} +窗函数频谱", xaxis_title="频率 (MHz)", yaxis_title="幅度 (dB)", height=600)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # ... (保持原有的 Plotly 匹配滤波绘图代码不变) ...
    fig = go.Figure()
    fig.add_scatter(x=t * 1e6, y=y_out_db, mode="lines", name="MF")
    fig.update_layout(title=f"{wave_type} 匹配滤波输出", xaxis_title="时间 (us)", yaxis_title="幅度 (dB)", height=600)
    fig.update_yaxes(range=[-150, 10])
    fig.update_xaxes(range=[-tau_us / 2, tau_us / 2])
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 📈 实时计算结果")

    # 获取当前配置描述
    current_config = f"{wave_type} + {window_type}"

    # 显示当前结果卡片
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前配置", current_config)
    col2.metric("峰值功率", f"{metrics['Peak Power (dB)']} dB")
    col3.metric("主瓣宽度", f"{metrics['Main Lobe Width (us)']} us")
    col4.metric("PSLR", f"{metrics['PSLR (dB)']} dB")


    # --- 记录按钮逻辑 (修正版) ---
    if st.button("💾 保存当前参数结果"):
        # 将当前参数和结果打包
        # 【关键修正】这里补充了 Peak_Power_dB 的赋值
        record = {
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Configuration": current_config,
            "Waveform": wave_type,
            "Window": window_type,
            "Pulse_Width_us": tau_us,
            "Bandwidth_MHz": B_MHz,
            "Peak_Power_dB": metrics["Peak Power (dB)"],  # <-- 修正：填入峰值功率
            "Main_Lobe_us": metrics["Main Lobe Width (us)"],
            "PSLR_dB": metrics["PSLR (dB)"]
        }
        st.session_state.history.append(record)
        st.success(f"已记录: {current_config} | 峰值: {metrics['Peak Power (dB)']} dB")

    st.markdown("---")

    if st.session_state.history:
        st.markdown("### 📋 历史记录 (可导出)")

        df_hist = pd.DataFrame(st.session_state.history)

        # 【关键修正】更新列名列表，加入 Peak_Power_dB
        # 并更新显示的中文列名
        display_df = df_hist[[
            "Timestamp",
            "Configuration",
            "Pulse_Width_us",
            "Bandwidth_MHz",
            "Peak_Power_dB",  # <-- 新增这一列
            "Main_Lobe_us",
            "PSLR_dB"
        ]].copy()

        display_df.columns = [
            "时间",
            "配置",
            "脉宽(us)",
            "带宽(MHz)",
            "峰值功率(dB)",  # <-- 对应新增列的中文名
            "主瓣(us)",
            "PSLR(dB)"
        ]

        st.dataframe(display_df, use_container_width=True)

        # --- 导出 CSV 功能 ---
        # 注意：这里导出的是包含所有列的 df_hist，不需要修改
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⏬ 导出所有记录为CSV",
            data=csv,
            file_name='radar_signal_records.csv',
            mime='text/csv',
        )