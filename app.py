import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

# 页面配置
st.set_page_config(page_title="LFM、NLFM加窗处理对比演示", layout="wide")

# 侧边栏 - 参数设置
st.sidebar.header("信号参数设置")

# 波形类型选择
wave_type = st.sidebar.selectbox("选择波形类型", ["LFM", "NLFM"], index=0)

# 脉冲宽度 (tau) - 使用 number_input 代替 slider
tau_us = st.sidebar.number_input(
    "脉冲宽度 (us)",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.1,
    format="%.1f",
    help="输入脉冲宽度，范围0.1到100.0微秒。"
)
tau = tau_us * 1e-6  # 转换为秒

# 带宽 (B) - 使用 number_input 代替 slider
B_MHz = st.sidebar.number_input(
    "信号带宽 (MHz)",
    min_value=10.0,
    max_value=1000.0,
    value=100.0,
    step=10.0,
    format="%.1f",
    help="输入信号带宽，范围10.0到1000.0 MHz。"
)
B = B_MHz * 1e6  # 转换为Hz

# 采样频率 (fs) - 保持为slider，因为它依赖于带宽
fs_MHz = st.sidebar.slider("采样频率 (MHz)", min_value=float(B_MHz * 1.1), max_value=float(B_MHz * 5),
                           value=float(B_MHz * 2.5), step=10.0)
fs = fs_MHz * 1e6

# 窗函数选择
window_type = st.sidebar.selectbox("选择窗函数", ["矩形窗", "汉明窗", "汉宁窗", "布莱克曼窗"], index=0)

# 主页面
st.title("雷达信号处理交互式演示")

# 1. 信号生成
t = np.arange(-tau / 2, tau / 2, 1 / fs)
if wave_type == "LFM":
    K = B / tau
    phi = np.pi * K * t ** 2
else:  # NLFM
    # 简化的NLFM信号相位函数 (非线性调频)
    phi = 2 * np.pi * B * (t / tau + 0.5 * np.sin(2 * np.pi * t / tau) / np.pi)

s_t = np.exp(1j * phi)

# 2. 频谱计算
N = len(s_t)
f = np.fft.fftfreq(N, 1 / fs)
f = np.fft.fftshift(f)
S_f = np.fft.fftshift(np.fft.fft(s_t))

# 3. 匹配滤波
h_t = np.conj(s_t[::-1])  # 匹配滤波器是信号的共轭时间反转

# 应用窗函数
window = np.ones(N)
if window_type == "汉明窗":
    window = np.hamming(N)
elif window_type == "汉宁窗":
    window = np.hanning(N)
elif window_type == "布莱克曼窗":
    window = np.blackman(N)
h_t = h_t * window

# 卷积实现匹配滤波
y_out = np.convolve(s_t, h_t, mode='same')

# 时间和频率轴用于绘图
t_us = t * 1e6
y_out_db = 20 * np.log10(np.abs(y_out) / np.max(np.abs(y_out)) + 1e-10)

# ======================
# 性能指标计算与历史记录逻辑
# ======================

# 1. 初始化 Session State (用于存储历史记录)
if 'history' not in st.session_state:
    st.session_state.history = []


# 2. 计算性能指标的函数
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

    # 线性插值 (对齐MATLAB)
    width_samples = right_idx - left_idx
    dt = t[1] - t[0]
    try:
        frac_left = (threshold_3dB - y_search[left_idx]) / (y_search[left_idx + 1] - y_search[left_idx])
        frac_right = (threshold_3dB - y_search[right_idx - 1]) / (y_search[right_idx] - y_search[right_idx - 1])
        width_samples_interp = (right_idx - left_idx - 1) + frac_right - frac_left
        main_lobe_width_us = width_samples_interp * dt * 1e6
    except:
        main_lobe_width_us = width_samples * dt * 1e6

    # --- PSLR (旁瓣电平) ---
    pslr_dB = np.nan
    exclude_range = int(main_lobe_width_us * 1e-6 / dt * 1.5) if not np.isnan(main_lobe_width_us) else int(
        fs * tau / 10)
    sidelobe_peaks, _ = find_peaks(y_search, height=-100, distance=min_dist)
    sidelobe_peaks = [p for p in sidelobe_peaks if abs(p - main_peak_idx_in_search) > exclude_range]

    if sidelobe_peaks:
        first_sidelobe_val = max(y_search[sidelobe_peaks])
        pslr_dB = first_sidelobe_val - main_peak_val

    return {
        "Peak Power (dB)": round(peak_power, 2),
        "Main Lobe Width (us)": round(main_lobe_width_us, 2),
        "PSLR (dB)": round(pslr_dB, 2)
    }


# --- 执行计算 ---
metrics = calculate_metrics(y_out_db, t, tau, fs)

# 3. 界面布局与交互
tab1, tab2, tab3 = st.tabs(["频谱分析", "匹配滤波", "📊 性能指标与记录"])

with tab1:
    st.subheader("信号频谱")
    spec_db = 20 * np.log10(np.abs(S_f) / np.max(np.abs(S_f)) + 1e-10)
    fig = go.Figure()
    fig.add_scatter(x=f / 1e6, y=spec_db, mode='lines', name='频谱')
    fig.update_layout(title='信号频谱 (dB)', xaxis_title='频率 (MHz)', yaxis_title='幅度 (dB)')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("匹配滤波输出")
    fig = go.Figure()
    fig.add_scatter(x=t * 1e6, y=y_out_db, mode="lines", name="MF Output")
    fig.update_layout(title=f"{wave_type} 匹配滤波输出 (dB)", xaxis_title='时间 (us)', yaxis_title='幅度 (dB)')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### 📈 实时计算结果")

    # 获取当前配置描述
    current_config = f"{wave_type} + {window_type}"

    # --- 显示当前结果卡片 ---
    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="配置", value=current_config)
        st.metric(label="峰值功率 (dB)", value=metrics["Peak Power (dB)"])

    with col2:
        st.metric(label="主瓣宽度 (us)", value=metrics["Main Lobe Width (us)"])
        st.metric(label="PSLR (dB)", value=metrics["PSLR (dB)"])

    # --- 记录按钮逻辑 ---
    if st.button("💾 保存当前参数结果"):
        record = {
            "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Configuration": current_config,
            "Waveform": wave_type,
            "Window": window_type,
            "Pulse_Width_us": tau_us,
            "Bandwidth_MHz": B_MHz,
            "Peak_Power_dB": metrics["Peak Power (dB)"],
            "Main_Lobe_us": metrics["Main Lobe Width (us)"],
            "PSLR_dB": metrics["PSLR (dB)"]
        }
        st.session_state.history.append(record)
        st.success(f"已记录: {current_config} | 峰值: {metrics['Peak Power (dB)']} dB")

    st.markdown("### 📋 历史记录 (可导出)")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)

        display_df = df_hist[
            ["Timestamp", "Configuration", "Pulse_Width_us", "Bandwidth_MHz", "Peak_Power_dB", "Main_Lobe_us",
             "PSLR_dB"]].copy()
        display_df.columns = ["时间", "配置", "脉宽(us)", "带宽(MHz)", "峰值功率(dB)", "主瓣(us)", "PSLR(dB)"]

        st.dataframe(display_df, use_container_width=True)

        # 导出 CSV 功能
        csv = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⏬ 导出所有记录为CSV",
            data=csv,
            file_name='radar_signal_records.csv',
            mime='text/csv',
        )
    else:
        st.info("暂无历史记录，请调整参数并点击【保存当前参数结果】。")