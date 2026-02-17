import streamlit as st
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, stft, welch

# --- Constants & Configuration ---
st.set_page_config(page_title="EEG Reconstruction Analysis", layout="wide")
st.title("🧠 EEG Reconstruction & Frequency Analysis")

FS = 200 
FRAME_SIZE = 800

# --- Core Processing Functions ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if high >= 1: high = 0.99
    if low <= 0: low = 0.001
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

def compute_snr_db_numpy(original, reconstruction):
    noise = original - reconstruction
    signal_power = np.mean(np.sum(original**2, axis=-1))
    noise_power = np.mean(np.sum(noise**2, axis=-1))
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    return snr

# --- Sidebar: Setup ---
st.sidebar.header("📂 Data Upload")
orig_file = st.sidebar.file_uploader("Upload Original H5", type=["h5"])
recon_file = st.sidebar.file_uploader("Upload Reconstructed H5", type=["h5"])

st.sidebar.divider()
st.sidebar.header("⚡ Processing Settings")
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

            trial_orig = st.sidebar.selectbox("Trial Select", orig_keys)
            default_idx = recon_keys.index(trial_orig) if trial_orig in recon_keys else 0
            trial_recon = st.sidebar.selectbox("Recon Match", recon_keys, index=default_idx)

            ds_orig = f_orig[trial_orig]
            ds_recon = f_recon[trial_recon]
            num_channels, total_samples = ds_orig.shape

            # Navigation
            total_frames = int(total_samples // FRAME_SIZE)
            current_frame = st.sidebar.number_input(f"Frame (1-{total_frames})", 1, total_frames, 1)
            selected_ch = st.sidebar.selectbox("Channel", range(num_channels))
            
            start_idx = (current_frame - 1) * FRAME_SIZE
            end_idx = start_idx + FRAME_SIZE

            # 1. Processing Stage
            y_orig_raw = ds_orig[selected_ch, start_idx : end_idx] / 100.0
            y_recon_raw = ds_recon[selected_ch, start_idx : end_idx]

            # Noise Canceling (0.5-50Hz) on Original
            y_orig_clean = butter_bandpass_filter(y_orig_raw, 0.5, 50.0, FS)
            y_recon_clean = y_recon_raw 

            # 2. Dual Mean Subtraction
            y_orig = y_orig_clean - np.mean(y_orig_clean)
            y_recon = y_recon_clean - np.mean(y_recon_clean)

            # 3. Frequency Band Filtering
            low, high = bands[selected_band_name]
            y_orig_final = butter_bandpass_filter(y_orig, low, high, FS)
            y_recon_final = butter_bandpass_filter(y_recon, low, high, FS)

            # 4. Main Plot (Time Domain)
            x = np.arange(start_idx, end_idx)
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=x, y=y_orig_final, name='Original (Processed)', line=dict(color='#2ecc71')))
            fig_time.add_trace(go.Scatter(x=x, y=y_recon_final, name='Reconstructed', line=dict(color='#e74c3c')))
            fig_time.update_layout(title="Time Domain Signal Comparison", template="plotly_dark", height=400)
            st.plotly_chart(fig_time, use_container_width=True)

            # 5. Metrics
            st.subheader("📊 Statistical Comparison")
            snr_val = compute_snr_db_numpy(y_orig_final, y_recon_final)
            corr = np.corrcoef(y_orig_final, y_recon_final)[0, 1]
            mse = np.mean((y_orig_final - y_recon_final)**2)

            m1, m2, m3 = st.columns(3)
            m1.metric("SNR (dB)", f"{snr_val:.2f} dB")
            m2.metric("Correlation (R)", f"{corr:.4f}")
            m3.metric("MSE", f"{mse:.6e}")

            # 6. STFT Side-by-Side
            st.divider()
            st.subheader("📉 Frequency Domain: STFT Spectrograms")
            
            def plot_stft(data, title):
                f, t, Zxx = stft(data, fs=FS, nperseg=128)
                fig_stft = go.Figure(data=go.Heatmap(
                    x=t, y=f, z=20 * np.log10(np.abs(Zxx) + 1e-10),
                    colorscale='Viridis', colorbar=dict(title="dB")
                ))
                fig_stft.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Freq (Hz)", height=350, template="plotly_dark")
                return fig_stft

            sc1, sc2 = st.columns(2)
            with sc1: st.plotly_chart(plot_stft(y_orig_final, "Original STFT"), use_container_width=True)
            with sc2: st.plotly_chart(plot_stft(y_recon_final, "Reconstructed STFT"), use_container_width=True)

           # 7. Power Spectral Density (PSD) - Both Solid Lines
            st.divider()
            st.subheader("📈 Power Spectral Density (PSD) Comparison")
            
            # Compute PSD using Welch's method 
            # (Signals are already mean-subtracted and band-filtered)
            f_orig, psd_orig = welch(y_orig_final, fs=FS, nperseg=256)
            f_recon, psd_recon = welch(y_recon_final, fs=FS, nperseg=256)

            fig_psd = go.Figure()

            # Original PSD - Solid Green
            fig_psd.add_trace(go.Scatter(
                x=f_orig, 
                y=10*np.log10(psd_orig + 1e-12), 
                name='Original PSD', 
                line=dict(color='#2ecc71', width=2.5) # Solid line
            ))

            # Reconstructed PSD - Solid Red
            fig_psd.add_scatter(
                x=f_recon, 
                y=10*np.log10(psd_recon + 1e-12), 
                name='Reconstructed PSD', 
                line=dict(color='#e74c3c', width=2.5) # Removed dash='dash'
            )
            
            fig_psd.update_layout(
                title="PSD Comparison (Welch's Method) - Solid Scale",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Power/Frequency (dB/Hz)",
                template="plotly_dark",
                height=450,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_psd, use_container_width=True)

    except Exception as e:
        st.error(f"Analysis Error: {e}")