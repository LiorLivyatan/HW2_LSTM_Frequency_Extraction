import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
from pathlib import Path
import yaml

# Page Config
st.set_page_config(
    page_title="LSTM Frequency Extraction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Feel
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444C;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .stPlotlyChart {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🌊 LSTM Frequency Extraction Dashboard")
st.markdown("Visualize the performance and outputs of the LSTM model designed to extract frequencies from noisy signals.")

# Sidebar - Configuration & Controls
st.sidebar.header("Configuration")

@st.cache_resource
def load_data():
    try:
        # Load Metrics
        with open('outputs/metrics.json', 'r') as f:
            metrics = json.load(f)
        
        # Load Config
        with open('outputs/run_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        # Load Predictions
        preds = np.load('outputs/predictions.npz')
        
        # Load Test Data (Inputs)
        test_data = np.load('data/test_data.npy')
        
        # Load Training History
        history_path = Path('models/training_history.json')
        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
        else:
            history = None
            
        return metrics, config, preds, test_data, history
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

metrics, config, preds, test_data, history = load_data()

if metrics:
    # --- Metrics Section ---
    st.header("📊 Model Performance")
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Train MSE", f"{metrics['overall']['mse_train']:.5f}")
    with c2:
        st.metric("Test MSE", f"{metrics['overall']['mse_test']:.5f}")
    with c3:
        st.metric("Generalization Gap", f"{metrics['generalization']['absolute_difference']:.2e}")
    with c4:
        is_good = metrics['generalization']['generalizes_well']
        st.metric("Generalizes Well?", "✅ Yes" if is_good else "❌ No")



    # --- Per Frequency Analysis ---
    st.subheader("Per-Frequency MSE Analysis")
    
    # Get frequencies from config for labels
    frequencies = config['data']['frequencies']
    freq_labels = [f"{f} Hz" for f in frequencies]
    
    # Get MSE data
    train_mse_dict = metrics['per_frequency']['train']
    test_mse_dict = metrics['per_frequency']['test']
    
    # Create DataFrame for plotting
    # Keys are strings "0", "1", etc.
    data = []
    for i, label in enumerate(freq_labels):
        idx_str = str(i)
        data.append({
            'Frequency': label,
            'Train MSE': train_mse_dict.get(idx_str, 0),
            'Test MSE': test_mse_dict.get(idx_str, 0)
        })
    
    df_freq = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig_freq = go.Figure()
    
    fig_freq.add_trace(go.Bar(
        x=df_freq['Frequency'],
        y=df_freq['Train MSE'],
        name='Train MSE',
        marker_color='#4A90E2'
    ))
    
    fig_freq.add_trace(go.Bar(
        x=df_freq['Frequency'],
        y=df_freq['Test MSE'],
        name='Test MSE',
        marker_color='#EF553B'
    ))
    
    fig_freq.update_layout(
        title="MSE per Frequency Component (Train vs Test)",
        xaxis_title="Frequency",
        yaxis_title="Mean Squared Error",
        template="plotly_dark",
        barmode='group',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    # --- Interactive Visualization ---
    st.header("📈 Signal Analysis")
    
    # Controls
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    
    with col_ctrl1:
        st.markdown("### Select Sample Range")
        # Total samples is 40000. Let's allow viewing a window.
        window_size = st.slider("Window Size", 100, 2000, 500)
        start_idx = st.slider("Start Index", 0, 40000 - window_size, 0)
        
        end_idx = start_idx + window_size
        
        st.markdown("### Legend")
        st.markdown("- **Noisy Input**: The mixed signal with random noise.")
        st.markdown("- **Clean Target**: The pure sinusoid we want to extract.")
        st.markdown("- **Model Prediction**: What the LSTM output.")
        
    with col_ctrl2:
        # Prepare data for plotting
        # test_data structure: [Signal, F1, F2, F3, F4, Target]
        # preds structure: 'test_predictions', 'test_targets'
        
        # Extract data
        input_signal = test_data[start_idx:end_idx, 0]
        clean_target = test_data[start_idx:end_idx, 5]
        model_pred = preds['test_predictions'][start_idx:end_idx]
        
        # Create Time Series Plot
        fig_ts = go.Figure()
        
        fig_ts.add_trace(go.Scatter(
            y=input_signal, 
            name="Noisy Input",
            line=dict(color='#808080', width=1),
            opacity=0.5
        ))
        
        fig_ts.add_trace(go.Scatter(
            y=clean_target, 
            name="Clean Target",
            line=dict(color='#00CC96', width=2)
        ))
        
        fig_ts.add_trace(go.Scatter(
            y=model_pred, 
            name="Model Prediction",
            line=dict(color='#EF553B', width=2, dash='dot')
        ))
        
        fig_ts.update_layout(
            title=f"Signal Comparison (Samples {start_idx} - {end_idx})",
            xaxis_title="Time Step",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)

    # --- Config View ---
    with st.expander("View Run Configuration"):
        st.json(config)

    # --- Dynamic Analysis Reports ---
    st.markdown("---")
    st.header("📊 Dynamic Analysis Reports")
    st.markdown("Interactive insights generated from the current dataset.")

    tab1, tab2, tab3 = st.tabs(["Spectral Analysis", "Error Analysis", "Training Dynamics"])

    with tab1:
        st.subheader("FFT Spectral Analysis")
        st.markdown("Frequency domain representation of the noisy input signal.")
        
        # Compute FFT on a segment of the test data
        # Use the first 1000 samples (1 second at 1000Hz)
        n_fft = 1000
        if len(test_data) >= n_fft:
            signal_segment = test_data[:n_fft, 0]
            fs = config['data']['sampling_rate']
            
            # Compute FFT
            fft_vals = np.fft.fft(signal_segment)
            freqs = np.fft.fftfreq(n_fft, 1/fs)
            
            # Take positive half
            pos_mask = freqs >= 0
            freqs = freqs[pos_mask]
            mag = np.abs(fft_vals)[pos_mask]
            
            # Scale magnitude
            mag = mag / n_fft
            
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=freqs, 
                y=mag, 
                mode='lines', 
                name='Magnitude',
                line=dict(color='#00CC96')
            ))
            
            # Add markers for expected frequencies
            expected_freqs = config['data']['frequencies']
            for f in expected_freqs:
                fig_fft.add_vline(x=f, line_dash="dash", line_color="white", annotation_text=f"{f}Hz")
            
            fig_fft.update_layout(
                title="Frequency Spectrum (First 1s)",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                template="plotly_dark",
                xaxis_range=[0, 20], # Focus on low frequencies
                height=400
            )
            st.plotly_chart(fig_fft, use_container_width=True)
        else:
            st.warning("Not enough data for FFT analysis.")

    with tab2:
        st.subheader("Error Analysis")
        
        # Calculate errors
        targets = preds['test_targets']
        predictions = preds['test_predictions']
        errors = predictions - targets
        
        c1, c2 = st.columns(2)
        
        with c1:
            # Error Histogram
            fig_err_hist = go.Figure()
            fig_err_hist.add_trace(go.Histogram(
                x=errors,
                nbinsx=50,
                marker_color='#EF553B',
                name='Error Distribution'
            ))
            fig_err_hist.update_layout(
                title="Error Distribution",
                xaxis_title="Error (Pred - Target)",
                yaxis_title="Count",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_err_hist, use_container_width=True)
            
        with c2:
            # Error over time (subset)
            subset_size = 1000
            fig_err_time = go.Figure()
            fig_err_time.add_trace(go.Scatter(
                y=errors[:subset_size],
                mode='lines',
                name='Error',
                line=dict(color='#FFA15A', width=1)
            ))
            fig_err_time.update_layout(
                title=f"Error over Time (First {subset_size} samples)",
                xaxis_title="Time Step",
                yaxis_title="Error",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_err_time, use_container_width=True)
            
        st.metric("Mean Absolute Error", f"{np.mean(np.abs(errors)):.5f}")
        st.metric("Standard Deviation of Error", f"{np.std(errors):.5f}")

    with tab3:
        st.subheader("Training Dynamics")
        
        if history:
            use_log = st.checkbox("Log Scale", value=True)
            
            hist_df = pd.DataFrame({
                'Epoch': range(1, len(history['train_loss']) + 1),
                'Loss': history['train_loss']
            })
            
            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(
                x=hist_df['Epoch'], 
                y=hist_df['Loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='#00CC96', width=2)
            ))
            
            fig_train.update_layout(
                title="Training Loss over Epochs",
                xaxis_title="Epoch",
                yaxis_title="MSE Loss",
                template="plotly_dark",
                height=400,
                yaxis_type="log" if use_log else "linear"
            )
            st.plotly_chart(fig_train, use_container_width=True)
        else:
            st.info("No training history found.")

else:
    st.warning("Could not load metrics. Please ensure the pipeline has been run.")
    st.info("Run `python main.py --mode all` to generate outputs.")
