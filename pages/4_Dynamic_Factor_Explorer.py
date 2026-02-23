"""
Dynamic Factor Model Explorer

Explore dynamic factor models with latent factors and observed series.
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

from exploration.multivariate.dfm import (
    simulate_dynamic_factor,
    compute_factor_correlation,
    compute_loadings_contribution,
    generate_block_loading_matrix,
    check_factor_identification
)
from utils.data_manager import create_save_widget

st.set_page_config(page_title="Dynamic Factor Explorer", layout="wide")

st.title("Dynamic Factor Model Explorer")

st.markdown("""
Explore Dynamic Factor Models (DFM) where multiple observed series are driven by a few latent factors.
""")

# Sidebar configuration
st.sidebar.header("Model Configuration")

# Model dimensions
n_factors = st.sidebar.selectbox(
    "Number of Factors",
    [1, 2, 3],
    index=1,
    help="Number of latent factors driving the observed series"
)

n_series = st.sidebar.slider(
    "Number of Observed Series",
    min_value=2,
    max_value=10,
    value=5,
    step=1,
    help="Number of observable time series"
)

st.sidebar.markdown("---")

# Loading matrix configuration
st.sidebar.subheader("Loading Matrix")

loading_type = st.sidebar.radio(
    "Loading Structure",
    ['Random', 'Block'],
    help="Random: each series loads on all factors. Block: each factor dominates a subset of series"
)

# Manual loading adjustment
show_loading_controls = st.sidebar.checkbox("Custom Loadings", value=False)

loading_matrix = None
if show_loading_controls:
    st.sidebar.write("**Loading Matrix (Series × Factors)**")
    st.sidebar.caption("Each cell shows how much a series loads on a factor")

    # Create loading matrix input
    loading_matrix = np.zeros((n_series, n_factors))

    for i in range(n_series):
        cols = st.sidebar.columns(n_factors)
        for j in range(n_factors):
            with cols[j]:
                loading_matrix[i, j] = st.number_input(
                    f"S{i+1},F{j+1}",
                    min_value=-2.0,
                    max_value=2.0,
                    value=1.0 if j == (i % n_factors) else 0.0,
                    step=0.1,
                    format="%.2f",
                    key=f"loading_{i}_{j}",
                    label_visibility="collapsed"
                )
else:
    if loading_type == 'Block':
        loading_matrix = generate_block_loading_matrix(n_series, n_factors)
    # else: None = will generate random loadings

st.sidebar.markdown("---")

# Factor dynamics
st.sidebar.subheader("Factor Dynamics")

factor_ar_params = []
for k in range(n_factors):
    ar_param = st.sidebar.slider(
        f"Factor {k+1} AR(1) Coefficient",
        min_value=-0.95,
        max_value=0.95,
        value=0.7,
        step=0.05,
        key=f"factor_ar_{k}",
        help=f"Persistence of Factor {k+1}"
    )
    factor_ar_params.append(ar_param)

st.sidebar.markdown("---")

# Variance parameters
st.sidebar.subheader("Variance Components")

factor_var = st.sidebar.slider(
    "Factor Variance",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Variance of factor innovations"
)

idiosyncratic_var = st.sidebar.slider(
    "Idiosyncratic Variance",
    min_value=0.1,
    max_value=5.0,
    value=0.5,
    step=0.1,
    help="Variance of series-specific errors"
)

st.sidebar.markdown("---")

# Shock configuration
st.sidebar.subheader("⚡ Shock Configuration")

enable_shock = st.sidebar.checkbox("Add Shock to Factor", value=False)

shock_factor = None
shock_time = None
shock_magnitude = 0.0

if enable_shock:
    shock_factor = st.sidebar.selectbox(
        "Shock Factor",
        list(range(n_factors)),
        format_func=lambda x: f"Factor {x+1}",
        help="Which factor to shock"
    )

    shock_time = st.sidebar.slider(
        "Shock Time",
        min_value=10,
        max_value=max(100, n_samples - 10) if 'n_samples' in locals() else 200,
        value=100
    )

    shock_magnitude = st.sidebar.slider(
        "Shock Magnitude (σ)",
        min_value=-10.0,
        max_value=10.0,
        value=3.0,
        step=0.5
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
st.subheader(f"Model: DFM with {n_factors} factor(s) and {n_series} series")

# Model equations
with st.expander("Model Equations"):
    st.latex(r'''
    \begin{align*}
    y_t &= \Lambda f_t + \varepsilon_t \\
    f_t &= \Phi f_{t-1} + \eta_t
    \end{align*}
    ''')
    st.write("**Observation equation**: Observed series = Loadings × Factors + Idiosyncratic errors")
    st.write("**Factor equation**: Factors follow AR(1) processes")
    st.write(f"- **y_t**: Observed series ({n_series} × 1)")
    st.write(f"- **f_t**: Latent factors ({n_factors} × 1)")
    st.write(f"- **Λ**: Loading matrix ({n_series} × {n_factors})")
    st.write(f"- **Φ**: Factor AR coefficient matrix (diagonal)")

# Check identification
if loading_matrix is not None:
    is_identified, id_msg = check_factor_identification(loading_matrix)
    if is_identified:
        st.success(f"✓ {id_msg}")
    else:
        st.warning(f"⚠️ {id_msg}")

# Simulate model
try:
    observed_df, factors_df, loading_matrix, info = simulate_dynamic_factor(
        n_factors=n_factors,
        n_series=n_series,
        n_samples=n_samples,
        loading_matrix=loading_matrix,
        factor_ar_params=factor_ar_params,
        factor_var=factor_var,
        idiosyncratic_var=idiosyncratic_var,
        seed=seed,
        shock_factor=shock_factor if enable_shock else None,
        shock_time=shock_time if enable_shock else None,
        shock_magnitude=shock_magnitude if enable_shock else 0.0
    )

    # Display shock info if present
    if info['shock_info'] is not None:
        st.info(f"⚡ Shock of {shock_magnitude:.1f}σ applied to Factor {shock_factor+1} at t={shock_time}")

    # Summary statistics
    st.subheader("Summary Statistics")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Factors", n_factors)
    with col2:
        st.metric("Number of Series", n_series)
    with col3:
        st.metric("Avg. Variance Explained", f"{info['avg_variance_explained']:.2%}")

    # Display variance explained by series
    var_explained_df = pd.DataFrame({
        'Series': observed_df.columns,
        'Variance Explained': [f"{ve:.2%}" for ve in info['variance_explained']]
    })
    st.dataframe(var_explained_df.T, width='stretch')

    # Equation/Formula view for presentations
    st.subheader("📊 Presentation View (with Equations)")

    # Build equations
    equations = [
        f"y(t) = Λ·f(t) + ε(t)  [{n_series} series, {n_factors} factors]",
        f"f_k(t) = φ_k·f_k(t-1) + η_k(t)  [AR(1) factors]",
        f"φ = [{', '.join([f'{p:.2f}' for p in factor_ar_params])}]",
        f"σ²_f = {factor_var:.2f}, σ²_ε = {idiosyncratic_var:.2f}"
    ]

    if info['shock_info']:
        equations.append(f"Shock: {shock_magnitude:.1f}σ to Factor {shock_factor+1} at t={shock_time}")

    formula_str = " | ".join(equations[:2])

    # Create presentation figure - show first 3 series
    fig_presentation = make_subplots(
        rows=min(3, n_series),
        cols=1,
        subplot_titles=[f"{observed_df.columns[i]}" for i in range(min(3, n_series))],
        vertical_spacing=0.1
    )

    for i in range(min(3, n_series)):
        fig_presentation.add_trace(
            go.Scatter(
                x=observed_df.index,
                y=observed_df.iloc[:, i],
                mode='lines',
                name=observed_df.columns[i],
                line=dict(width=2)
            ),
            row=i+1,
            col=1
        )

    fig_presentation.update_layout(
        title={
            'text': f"<b>DFM: {formula_str}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        height=150 * min(3, n_series),
        showlegend=False,
        template='plotly_white',
        font=dict(size=11)
    )

    st.plotly_chart(fig_presentation, width='stretch')
    st.caption("💡 Right-click chart → 'Save image as...' to export for presentations")

    with st.expander("Full DFM Equations"):
        for eq in equations:
            st.code(eq, language=None)

    # Loading matrix display
    st.subheader("Loading Matrix")

    loadings_df = compute_loadings_contribution(
        loading_matrix,
        factors_df.columns.tolist(),
        observed_df.columns.tolist()
    )

    # Heatmap of loadings
    fig_loadings = go.Figure(data=go.Heatmap(
        z=loadings_df.values,
        x=loadings_df.columns,
        y=loadings_df.index,
        colorscale='RdBu',
        zmid=0,
        text=loadings_df.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 10},
        colorbar=dict(title="Loading")
    ))

    fig_loadings.update_layout(
        title="Factor Loadings (Series × Factors)",
        xaxis_title="Factors",
        yaxis_title="Observed Series",
        template='plotly_white',
        height=max(300, n_series * 40)
    )

    st.plotly_chart(fig_loadings, use_container_width=True)

    # Show numerical loadings table
    with st.expander("Numerical Loading Values"):
        st.dataframe(loadings_df.style.format("{:.3f}"), use_container_width=True)

    # Factor time series
    st.subheader("Latent Factors")

    fig_factors = make_subplots(
        rows=n_factors,
        cols=1,
        subplot_titles=[f"Factor {i+1} (AR={factor_ar_params[i]:.2f})" for i in range(n_factors)],
        vertical_spacing=0.1
    )

    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i in range(n_factors):
        fig_factors.add_trace(
            go.Scatter(
                x=factors_df.index,
                y=factors_df.iloc[:, i],
                mode='lines',
                name=f'Factor {i+1}',
                line=dict(color=colors[i % len(colors)], width=1.5)
            ),
            row=i+1,
            col=1
        )

        # Add shock marker if present
        if info['shock_info'] is not None and i == info['shock_info']['factor']:
            shock_idx = info['shock_info']['time']
            fig_factors.add_vline(
                x=factors_df.index[shock_idx],
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=i+1,
                col=1
            )

    fig_factors.update_layout(
        height=250 * n_factors,
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig_factors, use_container_width=True)

    # Z-score overlay plot
    st.subheader("Factor vs Series Overlay (Z-Score Standardized)")

    st.markdown("**Compare factor movement with observed series (both standardized to mean=0, std=1)**")

    col1, col2 = st.columns(2)
    with col1:
        selected_factor = st.selectbox(
            "Select Factor",
            list(range(n_factors)),
            format_func=lambda x: f"Factor {x+1}",
            key="overlay_factor"
        )
    with col2:
        selected_series = st.selectbox(
            "Select Series",
            list(range(n_series)),
            format_func=lambda x: f"Series {x+1}",
            key="overlay_series"
        )

    # Compute z-scores
    factor_data = factors_df.iloc[:, selected_factor]
    series_data = observed_df.iloc[:, selected_series]

    factor_zscore = (factor_data - factor_data.mean()) / factor_data.std()
    series_zscore = (series_data - series_data.mean()) / series_data.std()

    # Create overlay plot
    fig_overlay = go.Figure()

    fig_overlay.add_trace(go.Scatter(
        x=factors_df.index,
        y=factor_zscore,
        mode='lines',
        name=f'Factor {selected_factor+1}',
        line=dict(color='red', width=2)
    ))

    fig_overlay.add_trace(go.Scatter(
        x=observed_df.index,
        y=series_zscore,
        mode='lines',
        name=f'Series {selected_series+1}',
        line=dict(color='blue', width=1.5, dash='dot')
    ))

    # Add shock marker if present
    if info['shock_info'] is not None and selected_factor == info['shock_info']['factor']:
        shock_idx = info['shock_info']['time']
        fig_overlay.add_vline(
            x=factors_df.index[shock_idx],
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )
        fig_overlay.add_annotation(
            x=factors_df.index[shock_idx],
            y=max(factor_zscore.max(), series_zscore.max()),
            text="Shock",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            ax=20,
            ay=-30
        )

    fig_overlay.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3)

    fig_overlay.update_layout(
        title=f"Factor {selected_factor+1} vs Series {selected_series+1} (Z-Scores)",
        xaxis_title="Time",
        yaxis_title="Z-Score (standardized units)",
        template='plotly_white',
        height=400,
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig_overlay, use_container_width=True)

    # Show loading for selected series
    st.caption(f"**Loading of Series {selected_series+1} on Factor {selected_factor+1}**: {loading_matrix[selected_series, selected_factor]:.3f}")

    # Factor correlation
    if n_factors > 1:
        st.subheader("Factor Correlation")
        factor_corr = compute_factor_correlation(factors_df)

        fig_factor_corr = go.Figure(data=go.Heatmap(
            z=factor_corr.values,
            x=factor_corr.columns,
            y=factor_corr.index,
            colorscale='RdBu',
            zmid=0,
            text=factor_corr.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title="Correlation"),
            zmin=-1,
            zmax=1
        ))

        fig_factor_corr.update_layout(
            title="Factor Correlations",
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig_factor_corr, use_container_width=True)

    # Observed series
    st.subheader("Observed Time Series")

    # Plot all series in subplots
    n_cols = 2
    n_rows = (n_series + n_cols - 1) // n_cols

    fig_obs = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=observed_df.columns.tolist(),
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )

    for idx, col in enumerate(observed_df.columns):
        row = idx // n_cols + 1
        col_pos = idx % n_cols + 1

        fig_obs.add_trace(
            go.Scatter(
                x=observed_df.index,
                y=observed_df[col],
                mode='lines',
                name=col,
                line=dict(width=1)
            ),
            row=row,
            col=col_pos
        )

    fig_obs.update_layout(
        height=250 * n_rows,
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig_obs, use_container_width=True)

    # Cross-correlation of observed series
    st.subheader("Series Cross-Correlation")
    obs_corr = observed_df.corr()

    fig_obs_corr = go.Figure(data=go.Heatmap(
        z=obs_corr.values,
        x=obs_corr.columns,
        y=obs_corr.index,
        colorscale='RdBu',
        zmid=0,
        text=obs_corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 9},
        colorbar=dict(title="Correlation"),
        zmin=-1,
        zmax=1
    ))

    fig_obs_corr.update_layout(
        title="Observed Series Correlations",
        template='plotly_white',
        height=max(400, n_series * 40)
    )

    st.plotly_chart(fig_obs_corr, use_container_width=True)

    # Variance decomposition
    st.subheader("Variance Decomposition")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Variance Explained by Series**")
        var_decomp_df = pd.DataFrame({
            'Series': observed_df.columns,
            'Variance Explained': [f"{ve:.1%}" for ve in info['variance_explained']],
            'Common Component': [f"{ve:.1%}" for ve in info['variance_explained']],
            'Idiosyncratic': [f"{1-ve:.1%}" for ve in info['variance_explained']]
        })
        st.dataframe(var_decomp_df, use_container_width=True)

    with col2:
        st.write("**Overall Signal-to-Noise Ratio**")
        avg_signal = np.mean(info['variance_explained'])
        avg_noise = 1 - avg_signal
        snr = avg_signal / avg_noise if avg_noise > 0 else np.inf

        st.metric("Average Variance Explained", f"{avg_signal:.1%}")
        st.metric("Average Idiosyncratic", f"{avg_noise:.1%}")
        st.metric("Signal-to-Noise Ratio", f"{snr:.2f}")

    # Save dataset section
    st.markdown("---")
    metadata = {
        'model': f'DFM({n_factors} factors, {n_series} series)',
        'n_factors': n_factors,
        'n_series': n_series,
        'factor_ar_params': factor_ar_params,
        'loading_structure': loading_type,
        'avg_variance_explained': info['avg_variance_explained']
    }
    if info['shock_info']:
        metadata['shock_factor'] = info['shock_info']['factor'] + 1
        metadata['shock_time'] = info['shock_info']['time']
        metadata['shock_magnitude'] = info['shock_info']['magnitude']

    create_save_widget(
        data=observed_df,
        dataset_type='multivariate',
        default_name=f"DFM_{n_factors}F_{n_series}S",
        metadata=metadata
    )

except Exception as e:
    st.error(f"Error simulating dynamic factor model: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# Help section
with st.expander("ℹ️ About Dynamic Factor Models"):
    st.markdown("""
    ### Dynamic Factor Models (DFM)

    Dynamic Factor Models explain the co-movement of multiple time series through a small number of latent factors.

    **Model Structure:**
    ```
    y_t = Λ * f_t + ε_t    (Observation equation)
    f_t = Φ * f_{t-1} + η_t  (Factor equation)
    ```

    ### Components

    **Latent Factors (f_t)**:
    - Unobserved common drivers
    - Each factor follows AR(1) process
    - Capture shared dynamics across series

    **Loading Matrix (Λ)**:
    - Shows how each series responds to each factor
    - **High loading**: Series is strongly driven by that factor
    - **Low loading**: Series weakly responds to that factor
    - **Sign**: Positive = same direction, Negative = opposite direction

    **Idiosyncratic Errors (ε_t)**:
    - Series-specific random noise
    - Not explained by common factors

    ### Loading Structures

    **Random**:
    - All series load on all factors
    - No particular structure

    **Block**:
    - Each factor dominates a subset of series
    - Mimics industry/sector groupings
    - Easier to interpret

    ### Variance Decomposition

    For each series:
    - **Common component**: Variance from factors (Λ * f_t)
    - **Idiosyncratic component**: Series-specific variance (ε_t)

    **Variance Explained** = Common Variance / Total Variance

    ### Identification

    A factor model is identified when:
    - At least k+1 series for k factors
    - Loading matrix has full rank
    - Factors are uncorrelated (orthogonal)

    ### Applications

    - **Macroeconomics**: Business cycle analysis
    - **Finance**: Risk factor models, portfolio analysis
    - **Nowcasting**: Real-time economic monitoring
    - **Dimensionality reduction**: Summarize many series with few factors

    ### Interpretation Tips

    - **High variance explained**: Series is mostly common variation
    - **Low variance explained**: Series is mostly idiosyncratic
    - **Factor correlation**: Ideally factors should be uncorrelated
    - **Loading patterns**: Look for which series load strongly on which factors
    """)

# Sidebar tips
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.markdown("""
- Start with 2 factors and 5 series
- Block structure easier to interpret
- Higher factor AR = more persistent
- Use z-score plot to see co-movement
- Add shocks to study propagation
- Compare variance explained across series
- Look for loading patterns
""")
