import streamlit as st
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, lfilter

# --- Configuration ---
st.set_page_config(page_title="EEG Frame Analysis Pro", layout="wide")
st.title("🧠 EEG Frame-by-Frame Comparison")

# Hardcoded constants
FS = 200 
FRAME_SIZE = 800  # Frame width in samples

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    if high >= 1: high = 0.99
    if low <= 0: low = 0.001
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# --- Sidebar: File Uploads ---
st.sidebar.header("📂 Data Upload")
orig_file = st.sidebar.file_uploader("Upload Original H5", type=["h5"], key="orig_up")
recon_file = st.sidebar.file_uploader("Upload Reconstructed H5", type=["h5"], key="recon_up")

# --- Sidebar: Band Selection ---
st.sidebar.divider()
st.sidebar.header("⚡ Signal Filtering")
bands = {
    "Full Spectrum (0.5-50Hz)": (0.5, 50),
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45)
}
selected_band_name = st.sidebar.selectbox("Select EEG Band", list(bands.keys()))

# --- Main Logic ---
if orig_file and recon_file:
    try:
        with h5py.File(orig_file, 'r') as f_orig, h5py.File(recon_file, 'r') as f_recon:
            orig_keys = sorted(list(f_orig.keys()))
            recon_keys = sorted(list(f_recon.keys()))

            st.sidebar.divider()
            st.sidebar.subheader("🔗 Match Trials")
            trial_orig = st.sidebar.selectbox("Select Original Trial", orig_keys)
            default_idx = recon_keys.index(trial_orig) if trial_orig in recon_keys else 0
            trial_recon = st.sidebar.selectbox("Select Reconstructed Trial", recon_keys, index=default_idx)

            if trial_orig and trial_recon:
                ds_orig = f_orig[trial_orig]
                ds_recon = f_recon[trial_recon]
                num_channels, total_samples = ds_orig.shape

                # --- NEW: Frame Navigation ---
                st.sidebar.divider()
                st.sidebar.subheader("🎞️ Frame Navigation")
                total_frames = int(total_samples // FRAME_SIZE)
                current_frame = st.sidebar.number_input(f"Frame (1 to {total_frames})", min_value=1, max_value=total_frames, value=1)
                
                # --- Scaling Adjustments ---
                st.sidebar.divider()
                st.sidebar.subheader("📏 Downscaling")
                # Downscaling original signal instead of upscaling reconstruction
                orig_downscale = st.sidebar.number_input("Original Downscale Divisor", value=100.0, step=10.0)
                
                selected_ch = st.sidebar.selectbox("Select Channel", range(num_channels))
                
                # Calculate start and end indices based on frame
                start_idx = (current_frame - 1) * FRAME_SIZE
                end_idx = start_idx + FRAME_SIZE

                # 1. Extract Data
                y_orig_raw = ds_orig[selected_ch, start_idx : end_idx] / orig_downscale
                y_recon_raw = ds_recon[selected_ch, start_idx : end_idx]

                # 2. Median Subtraction (Accuracy Improvement)
                # Removes the median of the signal to center it perfectly at 0
                y_orig_clean = y_orig_raw - np.median(y_orig_raw)
                y_recon_clean = y_recon_raw - np.median(y_recon_raw)

                # 3. Apply Selected Band Filter
                low, high = bands[selected_band_name]
                y_orig = butter_bandpass_filter(y_orig_clean, low, high, FS)
                y_recon = butter_bandpass_filter(y_recon_clean, low, high, FS)

                # 4. Plotting
                x = np.arange(start_idx, end_idx)
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x, y=y_orig, name='Original (Downscaled + Cleaned)', line=dict(color='#2ecc71', width=2)))
                fig.add_trace(go.Scatter(x=x, y=y_recon, name='Reconstructed', line=dict(color='#e74c3c', width=1.5)))

                fig.update_layout(
                    title=f"Ch {selected_ch} | Frame {current_frame}/{total_frames} | Samples {start_idx}-{end_idx}",
                    xaxis_title="Samples", yaxis_title="Adjusted Amplitude",
                    hovermode="x unified", height=600, template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

                # 5. Metrics
                st.divider()
                col1, col2, col3 = st.columns(3)
                corr = np.corrcoef(y_orig, y_recon)[0, 1]
                mse = np.mean((y_orig - y_recon)**2)
                
                col1.metric("Correlation", f"{corr:.4f}")
                col2.metric("MSE", f"{mse:.4f}")
                col3.metric("Downscale Factor", f"1/{orig_downscale}")

    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.info("👋 Upload both H5 files to begin comparison.")