import streamlit as st
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, lfilter

# --- Configuration ---
st.set_page_config(page_title="EEG Reconstruction Viewer", layout="wide")
st.title("🧠 EEG Original vs. Reconstructed Analysis")

# --- Optimized Filter Function ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Ensure frequencies don't exceed Nyquist
    if high >= 1:
        high = 0.99
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# --- Sidebar: File Uploads ---
st.sidebar.header("📂 Data Upload")
orig_file = st.sidebar.file_uploader("Upload Original H5", type=["h5"], key="orig")
recon_file = st.sidebar.file_uploader("Upload Reconstructed H5", type=["h5"], key="recon")

# --- Sidebar: Band Selection ---
st.sidebar.divider()
st.sidebar.header("⚡ Signal Processing")
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=200) # SEED-IV default

# Updated Bands as per your request
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45),
    "Full Signal": (0.5, 45) # Option to see everything
}

apply_filter = st.sidebar.checkbox("Apply Frequency Band Filter", value=False)
selected_band_name = st.sidebar.selectbox("Select EEG Band", list(bands.keys()), disabled=not apply_filter)

if orig_file and recon_file:
    with h5py.File(orig_file, 'r') as f_orig, h5py.File(recon_file, 'r') as f_recon:
        # Get common trials
        common_keys = list(set(f_orig.keys()) & set(f_recon.keys()))
        trial = st.sidebar.selectbox("Select Trial", sorted(common_keys))
        
        ds_orig = f_orig[trial]
        ds_recon = f_recon[trial]
        num_channels, total_samples = ds_orig.shape

        # Selection Controls
        selected_ch = st.sidebar.selectbox("Select Channel", range(num_channels))
        window_size = st.sidebar.slider("View Window (Samples)", 100, 5000, 1000)
        start_idx = st.sidebar.slider("Time Scroll", 0, max(0, total_samples - window_size), 0)

        # 1. Extract Data
        y_orig = ds_orig[selected_ch, start_idx : start_idx + window_size]
        y_recon = ds_recon[selected_ch, start_idx : start_idx + window_size]
        x = np.arange(start_idx, start_idx + window_size)

        # 2. Apply Filter
        if apply_filter:
            low, high = bands[selected_band_name]
            # Add a small buffer to avoid filtering errors on very short windows
            try:
                y_orig = butter_bandpass_filter(y_orig, low, high, fs)
                y_recon = butter_bandpass_filter(y_recon, low, high, fs)
            except ValueError:
                st.error("Window size too small for this filter frequency.")

        # 3. Plotly Interactive Graph
        fig = go.Figure()

        # Original (Green)
        fig.add_trace(go.Scatter(
            x=x, y=y_orig,
            name='Original',
            line=dict(color='#2ecc71', width=1.5)
        ))

        # Reconstructed (Red)
        fig.add_trace(go.Scatter(
            x=x, y=y_recon,
            name='Reconstructed',
            line=dict(color='#e74c3c', width=1.5, dash='dash')
        ))

        fig.update_layout(
            title=f"Comparison: Channel {selected_ch} ({selected_band_name if apply_filter else 'Raw'})",
            xaxis_title="Samples",
            yaxis_title="Amplitude",
            hovermode="x unified",
            height=500,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # 4. Metrics
        st.divider()
        m1, m2, m3 = st.columns(3)
        
        mse = np.mean((y_orig - y_recon)**2)
        corr = np.corrcoef(y_orig, y_recon)[0, 1]
        snr = 10 * np.log10(np.sum(y_orig**2) / np.sum((y_orig - y_recon)**2))

        m1.metric("Mean Squared Error", f"{mse:.4f}")
        m2.metric("Correlation", f"{corr:.4f}")
        m3.metric("SNR (dB)", f"{snr:.2f}")

else:
    st.info("Please upload both H5 files to start the comparison.")