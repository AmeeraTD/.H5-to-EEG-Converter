import streamlit as st
import h5py
import numpy as np
import plotly.graph_objects as go
from scipy.signal import butter, lfilter

# --- Page Configuration ---
st.set_page_config(page_title="EEG Reconstruction Pro", layout="wide")
st.title("🧠 EEG Original vs. Reconstructed Analysis")

# --- 1. Signal Processing: The Butterworth Filter ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if high >= 1: 
        high = 0.99
    
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)

# --- 2. Sidebar: File Uploads ---
st.sidebar.header("📂 Data Upload")
orig_file = st.sidebar.file_uploader("Upload Original H5", type=["h5"], key="orig_up")
recon_file = st.sidebar.file_uploader("Upload Reconstructed H5", type=["h5"], key="recon_up")

# --- 3. Sidebar: Frequency Band Settings ---
st.sidebar.divider()
st.sidebar.header("⚡ Signal Processing")
fs = st.sidebar.number_input("Sampling Rate (Hz)", value=200)

bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 45),
    "Raw (Full)": (0.5, 45)
}

apply_filter = st.sidebar.checkbox("Apply Frequency Band Filter", value=False)
selected_band_name = st.sidebar.selectbox("Select EEG Band", list(bands.keys()), disabled=not apply_filter)

# --- 4. Main Logic ---
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

                # --- 5. Scaling & Visualization Controls ---
                st.sidebar.divider()
                st.sidebar.subheader("📏 Signal Alignment")
                scaling_factor = st.sidebar.number_input("Reconstructed Scale Multiplier", value=100.0, step=10.0)
                
                selected_ch = st.sidebar.selectbox("Select Channel", range(num_channels))
                window_size = st.sidebar.slider("View Window (Samples)", 100, 5000, 1000)
                start_idx = st.sidebar.slider("Time Scroll", 0, max(0, total_samples - window_size), 0)

                # Slice Data
                y_orig = ds_orig[selected_ch, start_idx : start_idx + window_size]
                y_recon = ds_recon[selected_ch, start_idx : start_idx + window_size] * scaling_factor
                x = np.arange(start_idx, start_idx + window_size)

                # Apply Filter
                if apply_filter:
                    low, high = bands[selected_band_name]
                    y_orig = butter_bandpass_filter(y_orig, low, high, fs)
                    y_recon = butter_bandpass_filter(y_recon, low, high, fs)

                # --- 6. Interactive Plotting (Solid Lines) ---
                fig = go.Figure()

                # Original Signal
                fig.add_trace(go.Scatter(
                    x=x, y=y_orig, 
                    name='Original Data', 
                    line=dict(color='#2ecc71', width=2) # Solid Green
                ))

                # Reconstructed Signal (Now a solid line)
                fig.add_trace(go.Scatter(
                    x=x, y=y_recon, 
                    name='Reconstructed Data', 
                    line=dict(color='#e74c3c', width=1.5) # Solid Red
                ))

                fig.update_layout(
                    title=f"Comparison: Channel {selected_ch} | Trial {trial_orig}",
                    xaxis_title="Samples",
                    yaxis_title="Amplitude",
                    hovermode="x unified",
                    height=600,
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)

                # --- 7. Similarity Metrics ---
                st.divider()
                st.subheader("📊 Statistical Comparison")
                col1, col2, col3 = st.columns(3)
                
                mse = np.mean((y_orig - y_recon)**2)
                corr = np.corrcoef(y_orig, y_recon)[0, 1]
                
                noise_power = np.sum((y_orig - y_recon)**2)
                snr = 10 * np.log10(np.sum(y_orig**2) / noise_power) if noise_power > 0 else 0

                col1.metric("Correlation", f"{corr:.4f}")
                col2.metric("MSE", f"{mse:.4f}")
                col3.metric("SNR (dB)", f"{snr:.2f}")

    except Exception as e:
        st.error(f"Error loading data: {e}")

else:
    st.info("👋 Upload both files in the sidebar to begin.")