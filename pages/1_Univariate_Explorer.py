"""
Comprehensive Univariate Time Series Explorer

Explore complex univariate time series by combining:
- Trend (linear, quadratic, exponential)
- Cycle (periodic oscillations)
- Seasonality (repeating patterns)
- ARMA processes (stochastic component)
- Shocks (impulse responses)
"""

import streamlit as st
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from exploration.univariate.components import simulate_univariate_series
from exploration.univariate.arma import check_stationarity, check_invertibility
from utils.plotting import (
    plot_time_series, plot_correlogram_comparison,
    plot_distribution, plot_summary_statistics
)
from utils.data_manager import create_save_widget

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "app_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="Univariate Explorer", layout="wide")

st.title("Univariate Time Series Explorer")

st.markdown("""
Build complex time series by combining multiple components:
**Trend** + **Seasonality** + **ARMA** + **Shocks**
""")

# Sidebar configuration
st.sidebar.header("Time Series Components")

# Sample size
n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=2000,
    value=500,
    step=50
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42
)

st.sidebar.markdown("---")

# TREND COMPONENT
st.sidebar.subheader("📈 Trend Component")
enable_trend = st.sidebar.checkbox("Enable Trend", value=False)

trend_type = 'none'
trend_slope = 0.0
trend_intercept = 0.0
trend_quad = 0.0

if enable_trend:
    trend_type = st.sidebar.selectbox(
        "Trend Type",
        ['linear', 'quadratic', 'exponential'],
        help="Type of deterministic trend"
    )

    trend_intercept = st.sidebar.slider(
        "Intercept",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
        help="Starting value"
    )

    trend_slope = st.sidebar.slider(
        "Slope",
        min_value=-0.5,
        max_value=0.5,
        value=0.05,
        step=0.01,
        help="Linear trend coefficient"
    )

    if trend_type == 'quadratic':
        trend_quad = st.sidebar.slider(
            "Quadratic Coefficient",
            min_value=-0.01,
            max_value=0.01,
            value=0.0001,
            step=0.0001,
            format="%.4f",
            help="Curvature of quadratic trend"
        )

st.sidebar.markdown("---")

# SEASONAL COMPONENT
st.sidebar.subheader("📅 Seasonal Component")
enable_seasonality = st.sidebar.checkbox("Enable Seasonality", value=False)

seasonal_amplitude = 0.0
seasonal_period = 12
seasonal_type = 'additive'

if enable_seasonality:
    seasonal_amplitude = st.sidebar.slider(
        "Seasonal Amplitude",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Strength of seasonal pattern"
    )

    seasonal_period = st.sidebar.number_input(
        "Seasonal Period",
        min_value=2,
        max_value=52,
        value=12,
        step=1,
        help="Number of observations per season (e.g., 4=Quarterly, 12=Monthly, 52=Weekly)"
    )

    seasonal_type = st.sidebar.radio(
        "Seasonal Type",
        ['additive', 'multiplicative'],
        help="Additive: constant amplitude. Multiplicative: scales with level"
    )

st.sidebar.markdown("---")

# ARMA COMPONENT
st.sidebar.subheader("📊 ARMA Component")
enable_arma = st.sidebar.checkbox("Enable ARMA", value=True)

ar_order = 0
ma_order = 0
ar_params = []
ma_params = []
sigma = 1.0

if enable_arma:
    ar_order = st.sidebar.slider("AR Order (p)", 0, 3, 1)
    ma_order = st.sidebar.slider("MA Order (q)", 0, 3, 0)

    if ar_order > 0:
        st.sidebar.write("**AR Coefficients:**")
        for i in range(ar_order):
            param = st.sidebar.slider(
                f"φ_{i+1}",
                -1.0, 1.0, 0.5 if i == 0 else 0.0, 0.05,
                key=f"ar_{i}"
            )
            ar_params.append(param)

    if ma_order > 0:
        st.sidebar.write("**MA Coefficients:**")
        for i in range(ma_order):
            param = st.sidebar.slider(
                f"θ_{i+1}",
                -1.0, 1.0, 0.3 if i == 0 else 0.0, 0.05,
                key=f"ma_{i}"
            )
            ma_params.append(param)

    sigma = st.sidebar.slider(
        "Noise Std Dev (σ)",
        0.1, 5.0, 1.0, 0.1,
        help="Standard deviation of random shocks"
    )

st.sidebar.markdown("---")

# SHOCK COMPONENT
st.sidebar.subheader("⚡ Shock (Impulse Response)")
enable_shock = st.sidebar.checkbox("Add Shock", value=False)

shock_time = None
shock_magnitude = 0.0

if enable_shock:
    shock_time = st.sidebar.slider(
        "Shock Time",
        10, n_samples - 10,
        min(100, n_samples // 2)
    )

    shock_magnitude = st.sidebar.slider(
        "Shock Magnitude (σ)",
        -50.0, 50.0, 3.0, 0.5
    )

# Generate button
generate_button = st.sidebar.button("Generate Series", type="primary")

# Main content area
st.subheader("Active Components")

# Show which components are active
active_components = []
if enable_trend:
    active_components.append(f"Trend ({trend_type})")
if enable_seasonality:
    active_components.append(f"Seasonality ({seasonal_type}, period={seasonal_period})")
if enable_arma:
    active_components.append(f"ARMA({ar_order},{ma_order})")
if enable_shock:
    active_components.append(f"Shock at t={shock_time}")

if active_components:
    st.info("📊 " + " + ".join(active_components))
else:
    st.warning("⚠️ No components enabled. Enable at least one component to generate a series.")

# Check stationarity if ARMA is enabled
if enable_arma and ar_order > 0:
    is_stationary, ar_roots = check_stationarity(np.array(ar_params))
    if is_stationary:
        st.success("✓ AR component is stationary")
    else:
        st.warning(f"⚠️ AR component is non-stationary. Root magnitudes: {[f'{abs(r):.3f}' for r in ar_roots]}")

if enable_arma and ma_order > 0:
    is_invertible, ma_roots = check_invertibility(np.array(ma_params))
    if is_invertible:
        st.success("✓ MA component is invertible")
    else:
        st.warning(f"⚠️ MA component is non-invertible. Root magnitudes: {[f'{abs(r):.3f}' for r in ma_roots]}")

# Simulate the series
try:
    series, components, shock_info = simulate_univariate_series(
        n_samples=n_samples,
        # Trend
        trend_type=trend_type if enable_trend else 'none',
        trend_slope=trend_slope,
        trend_intercept=trend_intercept,
        trend_quad=trend_quad,
        # Seasonality
        seasonal_amplitude=seasonal_amplitude if enable_seasonality else 0.0,
        seasonal_period=seasonal_period,
        seasonal_type=seasonal_type,
        # ARMA
        ar_params=np.array(ar_params) if enable_arma else np.array([]),
        ma_params=np.array(ma_params) if enable_arma else np.array([]),
        sigma=sigma if enable_arma else 1.0,
        # Shock
        shock_time=shock_time if enable_shock else None,
        shock_magnitude=shock_magnitude if enable_shock else 0.0,
        # General
        seed=seed
    )

    # Summary statistics
    st.subheader("Summary Statistics")
    plot_summary_statistics(series)

    # Main time series plot
    st.subheader("Combined Time Series")
    fig_ts = plot_time_series(series, title="Univariate Time Series", shock_info=None)
    st.plotly_chart(fig_ts, width='stretch')

    # Equation/Formula view for presentations
    st.subheader("📊 Presentation View (with Formula)")

    # Build formula string
    formula_parts = []
    if enable_trend:
        if trend_type == 'linear':
            formula_parts.append(f"{trend_intercept:.2f} + {trend_slope:.4f}t")
        elif trend_type == 'quadratic':
            formula_parts.append(f"{trend_intercept:.2f} + {trend_slope:.4f}t + {trend_quad:.6f}t²")
        elif trend_type == 'exponential':
            formula_parts.append(f"{trend_intercept:.2f} exp({trend_slope:.4f}t)")

    if enable_seasonality:
        if seasonal_type == 'additive':
            formula_parts.append(f"{seasonal_amplitude:.2f}·sin(2πt/{seasonal_period})")
        else:
            formula_parts.append(f"[1 + {seasonal_amplitude:.2f}·sin(2πt/{seasonal_period})]")

    if enable_arma:
        arma_str = f"ARMA({ar_order},{ma_order})"
        formula_parts.append(arma_str)

    if len(formula_parts) == 0:
        formula_str = "y(t) = ε(t)"
    elif seasonal_type == 'multiplicative' and enable_seasonality:
        # For multiplicative seasonality
        non_seasonal = [p for i, p in enumerate(formula_parts) if not ('sin' in p and '·' in p)]
        seasonal_part = [p for p in formula_parts if 'sin' in p and '·' in p]
        if non_seasonal and seasonal_part:
            formula_str = f"y(t) = ({' + '.join(non_seasonal)}) × {seasonal_part[0]}"
        else:
            formula_str = f"y(t) = {' + '.join(formula_parts)}"
    else:
        formula_str = f"y(t) = {' + '.join(formula_parts)}"

    if enable_shock and shock_time:
        formula_str += f" + Shock({shock_magnitude:.1f}σ at t={shock_time})"

    # Create presentation figure
    fig_presentation = go.Figure()

    fig_presentation.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name='Series',
        line=dict(color='#1f77b4', width=2)
    ))

    fig_presentation.update_layout(
        title={
            'text': f"<b>{formula_str}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="Time",
        yaxis_title="Value",
        template='plotly_white',
        height=500,
        showlegend=False,
        font=dict(size=12)
    )

    st.plotly_chart(fig_presentation, width='stretch')
    st.caption("💡 Right-click chart → 'Save image as...' to export for presentations")

    # Component decomposition
    if enable_trend or enable_seasonality:
        st.subheader("Component Breakdown")

        # Show individual components
        show_components = st.checkbox("Show Individual Components", value=False)

        if show_components:
            cols = st.columns(2)

            if enable_trend:
                with cols[0]:
                    st.write("**Trend**")
                    fig_trend = plot_time_series(components['trend'], title="Trend Component", shock_info=None)
                    st.plotly_chart(fig_trend, width='stretch')

            if enable_seasonality:
                with cols[1]:
                    st.write("**Seasonality**")
                    fig_season = plot_time_series(components['seasonality'], title="Seasonal Component", shock_info=None)
                    st.plotly_chart(fig_season, width='stretch')

    # ACF/PACF analysis
    if enable_arma or len(active_components) > 0:
        st.subheader("Autocorrelation Analysis")
        plot_correlogram_comparison(series, lags=min(50, n_samples // 4))

    # Distribution
    st.subheader("Distribution")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_dist = plot_distribution(series)
        st.plotly_chart(fig_dist, width='stretch')
    with col2:
        st.markdown("### Distribution Info")
        st.write(f"**Mean**: {series.mean():.4f}")
        st.write(f"**Std Dev**: {series.std():.4f}")
        st.write(f"**Skewness**: {series.skew():.4f}")
        st.write(f"**Kurtosis**: {series.kurtosis():.4f}")
        st.write(f"**Range**: [{series.min():.3f}, {series.max():.3f}]")

    # Save dataset section
    st.markdown("---")
    components_list = []
    if enable_trend:
        components_list.append(f"Trend_{trend_type}")
    if enable_seasonality:
        components_list.append(f"Seasonal_{seasonal_type}_P{seasonal_period}")
    if enable_arma:
        components_list.append(f"ARMA_{ar_order}_{ma_order}")

    metadata = {
        'components': ', '.join(components_list) if components_list else 'None',
        'trend_type': trend_type if enable_trend else None,
        'seasonal_period': seasonal_period if enable_seasonality else None,
        'ar_order': ar_order if enable_arma else 0,
        'ma_order': ma_order if enable_arma else 0,
        'n_samples': n_samples
    }
    if enable_shock and shock_time:
        metadata['shock_time'] = shock_time
        metadata['shock_magnitude'] = shock_magnitude

    # Convert series to DataFrame for consistency
    series_df = series.to_frame(name='Univariate_series')
    create_save_widget(
        data=series_df,
        dataset_type='univariate',
        default_name="Univariate_" + "_".join(components_list[:2]) if components_list else "Univariate",
        metadata=metadata
    )

except Exception as e:
    st.error(f"Error generating series: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# Help section
with st.expander("ℹ️ About This Explorer"):
    st.markdown("""
    ### Univariate Time Series Components

    This tool allows you to build complex time series by combining different components:

    **📈 Trend**: Long-term movement in the data
    - Linear: Constant rate of change
    - Quadratic: Accelerating/decelerating change
    - Exponential: Growth/decay at increasing rate

    **📅 Seasonality**: Repeating patterns at fixed intervals
    - Additive: Constant seasonal effect
    - Multiplicative: Seasonal effect proportional to level
    - Common periods: 4 (quarterly), 12 (monthly), 52 (weekly)

    **📊 ARMA**: Random fluctuations with memory
    - AR (AutoRegressive): Depends on past values
    - MA (Moving Average): Depends on past errors

    **⚡ Shock**: One-time disturbance
    - **With ARMA**: Shock propagates through AR/MA dynamics
    - **Without ARMA**: Shock is a permanent level shift

    ### Model Equation

    **Additive**: `y(t) = Trend + Seasonality + ARMA`

    **Multiplicative**: `y(t) = Trend × Seasonality + ARMA`

    ### Shock Behavior

    - **Pure Trend/Seasonality** (no ARMA): Shock creates a permanent level shift from shock time onward
    - **With ARMA**: Shock propagates through the AR/MA process and gradually decays (or persists depending on parameters)
    """)

# Tips
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.markdown("""
- Start with one component at a time
- Combine gradually to understand interactions
- Use shocks to study impulse responses
- Compare additive vs multiplicative seasonality
- Observe how trend affects variance
""")
