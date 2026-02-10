import streamlit as st
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, stft

# --- Configuration ---
st.set_page_config(page_title="EEG Reconstruction & STFT", layout="wide")
st.title("🧠 EEG Time & Frequency Domain Analysis")

FS = 200 
FRAME_SIZE = 800

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if high >= 1: high = 0.99
    if low <= 0: low = 0.001
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# NumPy implementation of your torch SNR function
def compute_snr_db_numpy(original, reconstruction):
    noise = original - reconstruction
    signal_power = np.mean(np.sum(original**2, axis=-1))
    noise_power = np.mean(np.sum(noise**2, axis=-1))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    return snr

# --- Sidebar ---
st.sidebar.header("📂 Data Upload")
orig_file = st.sidebar.file_uploader("Upload Original H5", type=["h5"])
recon_file = st.sidebar.file_uploader("Upload Reconstructed H5", type=["h5"])

st.sidebar.divider()
st.sidebar.header("⚡ Processing")
bands = {
    "Full Spectrum (0.5-50Hz)": (0.5, 50), "Delta": (0.5, 4), 
    "Theta": (4, 8), "Alpha": (8, 13), "Beta": (13, 30), "Gamma": (30, 45)
}
selected_band_name = st.sidebar.selectbox("Select EEG Band", list(bands.keys()))

if orig_file and recon_file:
    try:
        with h5py.File(orig_file, 'r') as f_orig, h5py.File(recon_file, 'r') as f_recon:
            orig_keys = sorted(list(f_orig.keys()))
            recon_keys = sorted(list(f_recon.keys()))

            trial_orig = st.sidebar.selectbox("Original Trial", orig_keys)
            default_idx = recon_keys.index(trial_orig) if trial_orig in recon_keys else 0
            trial_recon = st.sidebar.selectbox("Reconstructed Trial", recon_keys, index=default_idx)

            ds_orig = f_orig[trial_orig]
            ds_recon = f_recon[trial_recon]
            num_channels, total_samples = ds_orig.shape

            total_frames = int(total_samples // FRAME_SIZE)
            current_frame = st.sidebar.number_input(f"Frame (1-{total_frames})", 1, total_frames, 1)
            
            selected_ch = st.sidebar.selectbox("Channel", range(num_channels))
            
            start_idx = (current_frame - 1) * FRAME_SIZE
            end_idx = start_idx + FRAME_SIZE

            # 1. Processing (Downscale Original by 100 as requested)
            y_orig_raw = ds_orig[selected_ch, start_idx : end_idx] / 100.0
            y_recon_raw = ds_recon[selected_ch, start_idx : end_idx]

            # Median Subtraction & Filtering
            y_orig = butter_bandpass_filter(y_orig_raw - np.median(y_orig_raw), *bands[selected_band_name], FS)
            y_recon = butter_bandpass_filter(y_recon_raw - np.median(y_recon_raw), *bands[selected_band_name], FS)

            # 2. Plotting Time Domain
            x = np.arange(start_idx, end_idx)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y_orig, name='Original (1/100)', line=dict(color='#2ecc71', width=2)))
            fig.add_trace(go.Scatter(x=x, y=y_recon, name='Reconstructed', line=dict(color='#e74c3c', width=1.5)))
            fig.update_layout(title="Time Domain Signal Comparison", template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)

            # 3. Metrics (Replaced NSR with your SNR formula)
            st.subheader("📊 Statistical Metrics")
            snr_val = compute_snr_db_numpy(y_orig, y_recon)
            corr = np.corrcoef(y_orig, y_recon)[0, 1]
            mse = np.mean((y_orig - y_recon)**2)

            col1, col2, col3 = st.columns(3)
            col1.metric("Signal-to-Noise Ratio (SNR dB)", f"{snr_val:.2f} dB")
            col2.metric("Correlation (R)", f"{corr:.4f}")
            col3.metric("MSE", f"{mse:.6e}")

            # 4. STFT Comparison (Side-by-Side)
            st.divider()
            st.subheader("📉 Frequency Domain: STFT Spectrograms")
            
            def get_stft_plot(data, title):
                # nperseg=128 gives a good balance for 200Hz sampling
                f, t, Zxx = stft(data, fs=FS, nperseg=128)
                fig_stft = go.Figure(data=go.Heatmap(
                    x=t, y=f, z=10 * np.log10(np.abs(Zxx) + 1e-10),
                    colorscale='Viridis',
                    colorbar=dict(title="dB")
                ))
                fig_stft.update_layout(
                    title=title, xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
                    height=400, template="plotly_dark"
                )
                return fig_stft

            st_col1, st_col2 = st.columns(2)
            with st_col1:
                st.plotly_chart(get_stft_plot(y_orig, "Original STFT (dB)"), use_container_width=True)
            with st_col2:
                st.plotly_chart(get_stft_plot(y_recon, "Reconstructed STFT (dB)"), use_container_width=True)

    except Exception as e:
        st.error(f"Analysis Error: {e}")
else:
    st.info("👋 Please upload your H5 files to start the analysis.")