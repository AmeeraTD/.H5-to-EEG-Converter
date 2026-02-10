import streamlit as st
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, lfilter

# --- Configuration ---
st.set_page_config(page_title="EEG Reconstruction Pro", layout="wide")
st.title("🧠 EEG Original vs. Reconstructed Analysis")

# Hardcoded constant for SEED-IV
FS = 200 

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Safety guards for digital stability
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

# Standard EEG Bands + Your Full Spectrum Requirement
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
            
            # Auto-match logic
            default_idx = recon_keys.index(trial_orig) if trial_orig in recon_keys else 0
            trial_recon = st.sidebar.selectbox("Select Reconstructed Trial", recon_keys, index=default_idx)

            if trial_orig and trial_recon:
                ds_orig = f_orig[trial_orig]
                ds_recon = f_recon[trial_recon]
                num_channels, total_samples = ds_orig.shape

                # --- Scaling & View Controls ---
                st.sidebar.divider()
                st.sidebar.subheader("📏 Alignment")
                scaling_factor = st.sidebar.number_input("Reconstructed Scale Multiplier", value=1.0, format="%.4f")
                
                selected_ch = st.sidebar.selectbox("Select Channel", range(num_channels))
                window_size = st.sidebar.slider("View Window (Samples)", 100, 5000, 1000)
                start_idx = st.sidebar.slider("Time Scroll", 0, max(0, total_samples - window_size), 0)

                # 1. Extract Data
                y_orig_raw = ds_orig[selected_ch, start_idx : start_idx + window_size]
                y_recon_raw = ds_recon[selected_ch, start_idx : start_idx + window_size] * scaling_factor

                # 2. Apply Selected Band Filter (Always at least 0.5-50Hz)
                low, high = bands[selected_band_name]
                y_orig = butter_bandpass_filter(y_orig_raw, low, high, FS)
                y_recon = butter_bandpass_filter(y_recon_raw, low, high, FS)

                # 3. Plotting
                x = np.arange(start_idx, start_idx + window_size)
                fig = go.Figure()

                fig.add_trace(go.Scatter(x=x, y=y_orig, name='Original (Filtered)', line=dict(color='#2ecc71', width=2)))
                fig.add_trace(go.Scatter(x=x, y=y_recon, name='Reconstructed', line=dict(color='#e74c3c', width=1.5)))

                fig.update_layout(
                    title=f"Comparison: Ch {selected_ch} | Band: {selected_band_name} | FS: {FS}Hz",
                    xaxis_title="Samples", yaxis_title="Amplitude",
                    hovermode="x unified", height=600, template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

                # 4. Metrics
                st.divider()
                col1, col2, col3 = st.columns(3)
                corr = np.corrcoef(y_orig, y_recon)[0, 1]
                mse = np.mean((y_orig - y_recon)**2)
                
                col1.metric("Correlation", f"{corr:.4f}")
                col2.metric("MSE", f"{mse:.4f}")
                col3.metric("Selected Band", selected_band_name)

    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.info("👋 Please upload both H5 files to begin comparison.")