"""
ARMA Exploration Page

Interactive exploration of AutoRegressive Moving Average (ARMA) models.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from exploration.univariate.arma import (
    simulate_arma, check_stationarity, check_invertibility,
    get_all_presets
)
from utils.plotting import (
    plot_time_series, plot_correlogram_comparison,
    plot_distribution, plot_summary_statistics
)
from utils.code_display import (
    format_arma_code, display_code_block,
    show_arma_theory, show_parameter_interpretation
)

# Load configuration
config_path = Path(__file__).parent.parent.parent / "config" / "app_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="ARMA Exploration", layout="wide")

st.title("ARMA Model Exploration")

st.markdown("""
Explore AutoRegressive Moving Average (ARMA) models by adjusting parameters and observing
how the time series evolves.
""")

# Sidebar for controls
st.sidebar.header("Model Configuration")

# Preset selector
st.sidebar.subheader("Quick Start")
presets = get_all_presets()
preset_names = ["Custom"] + list(presets.keys())
selected_preset = st.sidebar.selectbox(
    "Load Preset Configuration",
    preset_names,
    help="Choose a preset to quickly explore common ARMA patterns"
)

# Initialize session state for parameters
if 'ar_order' not in st.session_state:
    st.session_state.ar_order = 1
if 'ma_order' not in st.session_state:
    st.session_state.ma_order = 1

# Load preset if selected
if selected_preset != "Custom":
    preset = presets[selected_preset]
    st.session_state.ar_order = preset['ar_order']
    st.session_state.ma_order = preset['ma_order']
    st.sidebar.info(f"**{selected_preset}**: {preset['description']}")

# Model order selection
st.sidebar.subheader("Model Orders")
ar_order = st.sidebar.slider(
    "AR Order (p)",
    min_value=0,
    max_value=config['arma']['max_ar_order'],
    value=st.session_state.ar_order,
    help="Number of lagged observations in the AR component"
)
st.session_state.ar_order = ar_order

ma_order = st.sidebar.slider(
    "MA Order (q)",
    min_value=0,
    max_value=config['arma']['max_ma_order'],
    value=st.session_state.ma_order,
    help="Number of lagged forecast errors in the MA component"
)
st.session_state.ma_order = ma_order

# AR coefficients
st.sidebar.subheader("AR Coefficients")
ar_params = []
if ar_order > 0:
    for i in range(ar_order):
        # Load preset values if available
        if selected_preset != "Custom" and i < len(presets[selected_preset]['ar_params']):
            default_val = presets[selected_preset]['ar_params'][i]
        else:
            default_val = 0.5 if i == 0 else 0.0

        param = st.sidebar.slider(
            f"φ_{i+1} (AR lag {i+1})",
            min_value=-1.0,
            max_value=1.0,
            value=float(default_val),
            step=0.05,
            help=f"Coefficient for y(t-{i+1})"
        )
        ar_params.append(param)
else:
    st.sidebar.info("No AR component (p=0)")

# MA coefficients
st.sidebar.subheader("MA Coefficients")
ma_params = []
if ma_order > 0:
    for i in range(ma_order):
        # Load preset values if available
        if selected_preset != "Custom" and i < len(presets[selected_preset]['ma_params']):
            default_val = presets[selected_preset]['ma_params'][i]
        else:
            default_val = 0.5 if i == 0 else 0.0

        param = st.sidebar.slider(
            f"θ_{i+1} (MA lag {i+1})",
            min_value=-1.0,
            max_value=1.0,
            value=float(default_val),
            step=0.05,
            help=f"Coefficient for ε(t-{i+1})"
        )
        ma_params.append(param)
else:
    st.sidebar.info("No MA component (q=0)")

# Simulation parameters
st.sidebar.subheader("Simulation Settings")
n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=config['arma']['min_samples'],
    max_value=config['arma']['max_samples'],
    value=config['arma']['default_samples'],
    step=50,
    help="Number of time points to simulate"
)

sigma = st.sidebar.slider(
    "Noise Std Dev (σ)",
    min_value=0.1,
    max_value=5.0,
    value=config['arma']['default_sigma'],
    step=0.1,
    help="Standard deviation of the white noise process"
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42,
    help="Set seed for reproducible results"
)

# Generate button
generate_button = st.sidebar.button("Generate New Series", type="primary")

# Main content area
# Display model specification
st.subheader(f"Model: ARMA({ar_order}, {ma_order})")

# Check stationarity and invertibility
if ar_order > 0:
    is_stationary, ar_roots = check_stationarity(np.array(ar_params))
    if is_stationary:
        st.success("✓ AR process is stationary (all roots outside unit circle)")
    else:
        st.warning("⚠ AR process is NOT stationary. Results may be unreliable.")
        st.write(f"Root magnitudes: {[f'{abs(r):.3f}' for r in ar_roots]}")

if ma_order > 0:
    is_invertible, ma_roots = check_invertibility(np.array(ma_params))
    if is_invertible:
        st.success("✓ MA process is invertible (all roots outside unit circle)")
    else:
        st.warning("⚠ MA process is NOT invertible.")
        st.write(f"Root magnitudes: {[f'{abs(r):.3f}' for r in ma_roots]}")

# Simulate ARMA process
try:
    series = simulate_arma(
        ar_params=np.array(ar_params),
        ma_params=np.array(ma_params),
        n_samples=n_samples,
        sigma=sigma,
        seed=seed
    )

    # Summary statistics
    st.subheader("Summary Statistics")
    plot_summary_statistics(series)

    # Time series plot
    st.subheader("Time Series Plot")
    fig_ts = plot_time_series(series, title=f"ARMA({ar_order}, {ma_order}) Process")
    st.plotly_chart(fig_ts, use_container_width=True)

    # ACF and PACF plots
    st.subheader("Autocorrelation Analysis")
    plot_correlogram_comparison(series, lags=config['plotting']['acf_lags'])

    # Distribution plot
    st.subheader("Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_dist = plot_distribution(series)
        st.plotly_chart(fig_dist, use_container_width=True)
    with col2:
        st.markdown("### Distribution Info")
        st.write(f"**Skewness**: {series.skew():.4f}")
        st.write(f"**Kurtosis**: {series.kurtosis():.4f}")
        st.write(f"**Range**: [{series.min():.3f}, {series.max():.3f}]")

    # Code display section
    st.subheader("Python Code")
    show_code = st.checkbox("Show Python Code", value=False)

    if show_code:
        code = format_arma_code(
            ar_order=ar_order,
            ma_order=ma_order,
            ar_params=ar_params,
            ma_params=ma_params,
            n_samples=n_samples,
            sigma=sigma,
            seed=seed
        )
        display_code_block(code)

except Exception as e:
    st.error(f"Error simulating ARMA process: {str(e)}")
    st.info("Try adjusting the parameters or checking stationarity conditions.")

# Expandable theory section
with st.expander("📚 ARMA Theory & Concepts"):
    tab1, tab2 = st.tabs(["Theory", "Parameter Interpretation"])

    with tab1:
        show_arma_theory()

    with tab2:
        show_parameter_interpretation()

# Tips
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.markdown("""
- Start with **presets** to understand common patterns
- Observe how **ACF** decays for AR processes
- **PACF** cuts off at lag p for AR(p)
- **ACF** cuts off at lag q for MA(q)
- For ARMA, both decay gradually
- Adjust **noise variance** to see volatility effects
""")
