"""
State Space / Unobserved Components Explorer

Explore canonical state space models with unobserved components.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from exploration.multivariate.state_space import (
    simulate_local_level,
    simulate_local_linear_trend,
    simulate_structural_model
)
from utils.data_manager import create_save_widget

st.set_page_config(page_title="State Space Explorer", layout="wide")

st.title("State Space / Unobserved Components Explorer")

st.markdown("""
Explore canonical state space models where observations are composed of unobserved components.
""")

# Sidebar configuration
st.sidebar.header("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ['Local Level', 'Local Linear Trend', 'Structural (Level + Trend + Seasonal)'],
    help="Select the state space model structure"
)

st.sidebar.markdown("---")

# Model-specific parameters
st.sidebar.subheader("Component Variances")

level_var = st.sidebar.slider(
    "Level Variance (σ²_η)",
    min_value=0.01,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Variance of level innovations"
)

trend_var = 0.1
seasonal_var = 0.5
seasonal_period = 12

if model_type in ['Local Linear Trend', 'Structural (Level + Trend + Seasonal)']:
    trend_var = st.sidebar.slider(
        "Trend Variance (σ²_ζ)",
        min_value=0.01,
        max_value=2.0,
        value=0.1,
        step=0.05,
        help="Variance of trend innovations"
    )

if model_type == 'Structural (Level + Trend + Seasonal)':
    seasonal_var = st.sidebar.slider(
        "Seasonal Variance (σ²_ω)",
        min_value=0.01,
        max_value=2.0,
        value=0.5,
        step=0.05,
        help="Variance of seasonal innovations"
    )

    seasonal_period = st.sidebar.number_input(
        "Seasonal Period",
        min_value=4,
        max_value=52,
        value=12,
        help="Number of periods per season"
    )

obs_var = st.sidebar.slider(
    "Observation Variance (σ²_ε)",
    min_value=0.01,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Variance of observation noise"
)

st.sidebar.markdown("---")

# Simulation settings
st.sidebar.subheader("Simulation Settings")

n_samples = st.sidebar.slider(
    "Number of Samples",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=10000,
    value=42
)

# Main content
st.subheader(f"Model: {model_type}")

# Display model equations
with st.expander("Model Equations"):
    if model_type == 'Local Level':
        st.latex(r'''
        \begin{align*}
        y_t &= \mu_t + \varepsilon_t \\
        \mu_t &= \mu_{t-1} + \eta_t
        \end{align*}
        ''')
        st.write("**Observation equation**: y = level + noise")
        st.write("**State equation**: level follows a random walk")

    elif model_type == 'Local Linear Trend':
        st.latex(r'''
        \begin{align*}
        y_t &= \mu_t + \varepsilon_t \\
        \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t \\
        \beta_t &= \beta_{t-1} + \zeta_t
        \end{align*}
        ''')
        st.write("**Observation equation**: y = level + noise")
        st.write("**State equations**: level evolves with time-varying trend")

    else:  # Structural
        st.latex(r'''
        \begin{align*}
        y_t &= \mu_t + \gamma_t + \varepsilon_t \\
        \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t \\
        \beta_t &= \beta_{t-1} + \zeta_t \\
        \sum_{j=0}^{s-1} \gamma_{t-j} &= \omega_t
        \end{align*}
        ''')
        st.write("**Observation equation**: y = level + seasonal + noise")
        st.write("**State equations**: level with trend + seasonal component")

# Simulate model
try:
    if model_type == 'Local Level':
        obs_series, level_series = simulate_local_level(
            n_samples=n_samples,
            level_var=level_var,
            obs_var=obs_var,
            seed=seed
        )
        components = {'level': level_series}

    elif model_type == 'Local Linear Trend':
        obs_series, level_series, trend_series = simulate_local_linear_trend(
            n_samples=n_samples,
            level_var=level_var,
            trend_var=trend_var,
            obs_var=obs_var,
            seed=seed
        )
        components = {
            'level': level_series,
            'trend': trend_series
        }

    else:  # Structural
        obs_series, components = simulate_structural_model(
            n_samples=n_samples,
            level_var=level_var,
            trend_var=trend_var,
            seasonal_var=seasonal_var,
            obs_var=obs_var,
            seasonal_period=seasonal_period,
            seed=seed
        )

    # Summary statistics
    st.subheader("Summary Statistics")
    stats_df = pd.DataFrame({
        'Observed': obs_series.describe()
    })
    st.dataframe(stats_df.T, width='stretch')

    # Plot observed series
    st.subheader("Observed Time Series")

    fig_obs = go.Figure()
    fig_obs.add_trace(go.Scatter(
        x=obs_series.index,
        y=obs_series.values,
        mode='lines',
        name='Observed',
        line=dict(color='blue', width=1.5)
    ))

    fig_obs.update_layout(
        title="Observed Series",
        xaxis_title="Time",
        yaxis_title="Value",
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig_obs, width='stretch')

    # Equation/Formula view for presentations
    st.subheader("📊 Presentation View (with Equations)")

    # Build equations based on model type (already in LaTeX format from earlier)
    if model_type == 'Local Level':
        st.latex(r'''
        \begin{align*}
        y_t &= \mu_t + \varepsilon_t \\
        \mu_t &= \mu_{t-1} + \eta_t
        \end{align*}
        ''')
        st.latex(f"\\sigma^2_\\eta = {level_var:.2f}, \\quad \\sigma^2_\\varepsilon = {obs_var:.2f}")

    elif model_type == 'Local Linear Trend':
        st.latex(r'''
        \begin{align*}
        y_t &= \mu_t + \varepsilon_t \\
        \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t \\
        \beta_t &= \beta_{t-1} + \zeta_t
        \end{align*}
        ''')
        st.latex(f"\\sigma^2_\\eta = {level_var:.2f}, \\quad \\sigma^2_\\zeta = {trend_var:.2f}, \\quad \\sigma^2_\\varepsilon = {obs_var:.2f}")

    else:  # Structural
        st.latex(r'''
        \begin{align*}
        y_t &= \mu_t + \gamma_t + \varepsilon_t \\
        \mu_t &= \mu_{t-1} + \beta_{t-1} + \eta_t \\
        \beta_t &= \beta_{t-1} + \zeta_t \\
        \sum_{j=0}^{s-1} \gamma_{t-j} &= \omega_t
        \end{align*}
        ''')
        st.latex(f"s = {seasonal_period}, \\quad \\sigma^2_\\eta = {level_var:.2f}, \\quad \\sigma^2_\\zeta = {trend_var:.2f}, \\quad \\sigma^2_\\omega = {seasonal_var:.2f}, \\quad \\sigma^2_\\varepsilon = {obs_var:.2f}")

    # Create presentation figure
    fig_presentation = go.Figure()

    fig_presentation.add_trace(go.Scatter(
        x=obs_series.index,
        y=obs_series.values,
        mode='lines',
        name='Observed',
        line=dict(color='#1f77b4', width=2)
    ))

    fig_presentation.update_layout(
        xaxis_title="Time",
        yaxis_title="Value",
        template='plotly_white',
        height=450,
        showlegend=False,
        font=dict(size=12)
    )

    st.plotly_chart(fig_presentation, width='stretch')
    st.caption("💡 Right-click chart → 'Save image as...' to export for presentations")

    # Plot components
    st.subheader("Unobserved Components")

    n_components = len(components)
    fig_components = make_subplots(
        rows=n_components + 1,
        cols=1,
        subplot_titles=['Observed'] + list(components.keys()),
        vertical_spacing=0.08
    )

    # Add observed series
    fig_components.add_trace(
        go.Scatter(
            x=obs_series.index,
            y=obs_series.values,
            mode='lines',
            name='Observed',
            line=dict(color='blue')
        ),
        row=1,
        col=1
    )

    # Add component plots
    colors = ['red', 'green', 'orange', 'purple']
    for idx, (comp_name, comp_series) in enumerate(components.items()):
        fig_components.add_trace(
            go.Scatter(
                x=comp_series.index,
                y=comp_series.values,
                mode='lines',
                name=comp_name.capitalize(),
                line=dict(color=colors[idx % len(colors)])
            ),
            row=idx + 2,
            col=1
        )

    fig_components.update_layout(
        height=300 * (n_components + 1),
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig_components, use_container_width=True)

    # Component contributions
    st.subheader("Component Breakdown")

    # Stack plot showing composition
    fig_stack = go.Figure()

    # Add components as stacked area
    if 'seasonal' in components:
        fig_stack.add_trace(go.Scatter(
            x=obs_series.index,
            y=components['seasonal'].values,
            mode='lines',
            name='Seasonal',
            stackgroup='one',
            fillcolor='rgba(255, 165, 0, 0.5)'
        ))

    if 'level' in components:
        fig_stack.add_trace(go.Scatter(
            x=obs_series.index,
            y=components['level'].values,
            mode='lines',
            name='Level',
            line=dict(color='red', width=2)
        ))

    if 'trend' in components:
        fig_stack.add_trace(go.Scatter(
            x=obs_series.index,
            y=components['trend'].values,
            mode='lines',
            name='Trend',
            line=dict(color='green', width=2)
        ))

    fig_stack.add_trace(go.Scatter(
        x=obs_series.index,
        y=obs_series.values,
        mode='lines',
        name='Observed',
        line=dict(color='blue', width=1, dash='dot')
    ))

    fig_stack.update_layout(
        title="Components and Observed Series",
        xaxis_title="Time",
        yaxis_title="Value",
        template='plotly_white',
        height=500
    )

    st.plotly_chart(fig_stack, use_container_width=True)

    # Variance decomposition
    st.subheader("Variance Decomposition")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Component Variances**")
        var_data = {
            'Component': [],
            'Variance': [],
            '% of Total': []
        }

        total_var = sum(comp.var() for comp in components.values()) + obs_var

        for name, comp in components.items():
            comp_var = comp.var()
            var_data['Component'].append(name.capitalize())
            var_data['Variance'].append(f"{comp_var:.4f}")
            var_data['% of Total'].append(f"{100 * comp_var / total_var:.2f}%")

        var_data['Component'].append('Observation Noise')
        var_data['Variance'].append(f"{obs_var:.4f}")
        var_data['% of Total'].append(f"{100 * obs_var / total_var:.2f}%")

        st.dataframe(pd.DataFrame(var_data), use_container_width=True)

    with col2:
        st.write("**Signal-to-Noise Ratio**")
        signal_var = sum(comp.var() for comp in components.values())
        snr = signal_var / obs_var
        st.metric("SNR", f"{snr:.4f}")
        st.caption("Ratio of signal variance to noise variance")

        st.write("**Observed Series Variance**")
        st.metric("Total Variance", f"{obs_series.var():.4f}")

    # Save dataset section
    st.markdown("---")
    metadata = {
        'model': model_type,
        'level_var': level_var,
        'obs_var': obs_var,
        'n_samples': n_samples
    }
    if model_type in ['Local Linear Trend', 'Structural (Level + Trend + Seasonal)']:
        metadata['trend_var'] = trend_var
    if model_type == 'Structural (Level + Trend + Seasonal)':
        metadata['seasonal_var'] = seasonal_var
        metadata['seasonal_period'] = seasonal_period

    # Convert series to DataFrame for consistency
    obs_df = obs_series.to_frame(name='StateSpace_series')
    create_save_widget(
        data=obs_df,
        dataset_type='univariate',
        default_name=f"StateSpace_{model_type.replace(' ', '_')}",
        metadata=metadata
    )

except Exception as e:
    st.error(f"Error simulating state space model: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# Help section
with st.expander("ℹ️ About State Space Models"):
    st.markdown("""
    ### State Space / Unobserved Components Models

    These models decompose observations into unobserved (latent) components.

    **General form**:
    - **Observation equation**: Links observations to unobserved states
    - **State equation**: Describes evolution of unobserved states

    ### Model Types

    **1. Local Level**
    - Observation = Random walk level + Noise
    - Simplest state space model
    - Level wanders over time

    **2. Local Linear Trend**
    - Observation = Level + Noise
    - Level = Previous level + Time-varying trend
    - Trend evolves as random walk
    - Captures smooth trends

    **3. Structural Model**
    - Observation = Level + Seasonal + Noise
    - Includes all components: trend, level, seasonality
    - Most comprehensive model

    ### Parameters

    **Variance parameters** control component smoothness:
    - **High variance**: Component changes rapidly
    - **Low variance**: Component is smooth
    - **Zero variance**: Component is deterministic

    ### Signal-to-Noise Ratio

    - **High SNR**: Clean signal, little noise
    - **Low SNR**: Noisy observations, hard to extract signal

    ### Applications

    - Trend extraction
    - Seasonal adjustment
    - Signal extraction from noisy data
    - Nowcasting and forecasting
    """)

# Sidebar tips
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.markdown("""
- Higher component variance = more variation
- Lower obs variance = cleaner signal
- Compare SNR across models
- Seasonal models need enough data
""")
