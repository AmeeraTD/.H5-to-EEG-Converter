import streamlit as st
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Page Config
st.set_page_config(page_title="SEED-IV Multi-Channel Viewer", layout="wide")

st.title("🧠 SEED-IV Multi-Channel EEG Visualizer")

uploaded_file = st.file_uploader("Upload SEED-IV H5 File", type=["h5"])

if uploaded_file:
    # Use context manager to keep the file accessible
    with h5py.File(uploaded_file, 'r') as f:
        keys = list(f.keys())
        
        # Sidebar Controls
        st.sidebar.header("Navigation")
        trial = st.sidebar.selectbox("Select Trial", keys)
        
        # Accessing dataset shape without loading into memory yet
        dataset = f[trial]
        num_channels, total_samples = dataset.shape
        
        # 1. Channel Selection (Crucial for performance)
        default_channels = [0, 1, 2] if num_channels > 3 else [0]
        selected_channels = st.sidebar.multiselect(
            "Select Channels to View", 
            options=list(range(num_channels)), 
            default=default_channels
        )
        
        # 2. Time Window Controls
        window_size = st.sidebar.slider("Window Size (Samples)", 100, 5000, 1000)
        start_idx = st.sidebar.slider("Scroll Through Time", 0, total_samples - window_size, 0)
        
        st.sidebar.divider()
        st.sidebar.write(f"**Total Dataset Channels:** {num_channels}")
        st.sidebar.write(f"**Total Duration:** {total_samples} samples")

        if selected_channels:
            # Create subplots only for selected channels
            n_plots = len(selected_channels)
            fig, axes = plt.subplots(n_plots, 1, figsize=(15, n_plots * 2), sharex=True)
            
            # If only one channel is selected, axes is not an array
            if n_plots == 1:
                axes = [axes]

            plt.subplots_adjust(hspace=0.4)

            for idx, ch_idx in enumerate(selected_channels):
                # Efficient slicing: Load only the window and channel needed from H5
                channel_data = dataset[ch_idx, start_idx : start_idx + window_size]
                
                axes[idx].plot(channel_data, color='#1abc9c', linewidth=0.8)
                axes[idx].set_ylabel(f"Ch {ch_idx}", rotation=0, labelpad=25, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, linestyle='--')
                
                # Cleanup plot aesthetic
                axes[idx].spines['top'].set_visible(False)
                axes[idx].spines['right'].set_visible(False)

            axes[-1].set_xlabel("Samples (Time Offset)")
            
            # Display in Streamlit
            st.pyplot(fig)
        else:
            st.warning("Please select at least one channel in the sidebar to visualize data.")

else:
    st.info("Please upload a .h5 file to begin.")