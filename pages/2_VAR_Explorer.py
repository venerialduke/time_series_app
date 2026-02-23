"""
VAR (Vector Autoregression) Explorer

Explore multivariate VAR models with 2-3 variables.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from exploration.multivariate.var import (
    simulate_var, simulate_var_with_shock,
    compute_irf, check_var_stability
)
from utils.data_manager import create_save_widget

st.set_page_config(page_title="VAR Explorer", layout="wide")

st.title("VAR Model Explorer")

st.markdown("""
Explore Vector Autoregression (VAR) models with multiple time series.
VAR models capture the dynamic relationships between variables.
""")

# Sidebar configuration
st.sidebar.header("VAR Configuration")

# Number of variables
n_vars = st.sidebar.selectbox(
    "Number of Variables",
    [2, 3],
    index=0,
    help="Number of time series in the VAR system"
)

# VAR order
var_order = st.sidebar.slider(
    "VAR Order (p)",
    min_value=1,
    max_value=3,
    value=1,
    help="Number of lags in the VAR model"
)

st.sidebar.markdown("---")

# Variable names
st.sidebar.subheader("Variable Names")
var_names = []
for i in range(n_vars):
    name = st.sidebar.text_input(
        f"Variable {i+1} Name",
        value=f"Var{i+1}",
        key=f"varname_{i}"
    )
    var_names.append(name)

st.sidebar.markdown("---")

# Coefficient matrices (simplified input)
st.sidebar.subheader(f"VAR({var_order}) Coefficients")

st.sidebar.markdown("**Note**: Enter small values (e.g., 0.1-0.5) for stability")

coef_matrices = []
for lag in range(var_order):
    st.sidebar.write(f"**Lag {lag+1} Coefficients**")

    # Create coefficient matrix for this lag
    A = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                default = 0.3 if lag == 0 else 0.1
            else:
                default = 0.1 if lag == 0 else 0.0

            val = st.sidebar.number_input(
                f"{var_names[i]} ← {var_names[j]}(t-{lag+1})",
                min_value=-1.0,
                max_value=1.0,
                value=float(default),
                step=0.05,
                format="%.2f",
                key=f"coef_{lag}_{i}_{j}",
                help=f"Effect of {var_names[j]} lag {lag+1} on {var_names[i]}"
            )
            A[i, j] = val

    coef_matrices.append(A)

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

# Shock configuration
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Shock Configuration")

enable_shock = st.sidebar.checkbox("Add Shock", value=False)

shock_var = 0
shock_time = None
shock_magnitude = 0.0

if enable_shock:
    shock_var = st.sidebar.selectbox(
        "Shock Variable",
        list(range(n_vars)),
        format_func=lambda x: var_names[x],
        help="Which variable to shock"
    )

    shock_time = st.sidebar.slider(
        "Shock Time",
        min_value=10,
        max_value=n_samples - 10,
        value=min(100, n_samples // 2)
    )

    shock_magnitude = st.sidebar.slider(
        "Shock Magnitude (σ)",
        min_value=-10.0,
        max_value=10.0,
        value=3.0,
        step=0.5
    )

# Main content
st.subheader("Model Specification")

# Check stability
is_stable, eigenvalues = check_var_stability(coef_matrices)
if is_stable:
    st.success(f"✓ VAR({var_order}) is stable")
else:
    st.warning(f"⚠️ VAR({var_order}) is NOT stable. Series may explode.")

st.write(f"**Companion matrix eigenvalues (max magnitude)**: {np.abs(eigenvalues).max():.4f}")
st.caption("All eigenvalues must be < 1.0 for stability")

# Simulate VAR
sigma = np.eye(n_vars)  # Identity covariance for simplicity

try:
    if enable_shock and shock_time is not None:
        df, shock_info = simulate_var_with_shock(
            coef_matrices=coef_matrices,
            shock_var=shock_var,
            shock_time=shock_time,
            shock_magnitude=shock_magnitude,
            n_samples=n_samples,
            sigma=sigma,
            seed=seed
        )
        st.info(f"⚡ Shock of {shock_magnitude:.1f}σ applied to {var_names[shock_var]} at t={shock_time}")
    else:
        df = simulate_var(
            coef_matrices=coef_matrices,
            n_samples=n_samples,
            sigma=sigma,
            seed=seed
        )
        shock_info = None

    # Rename columns
    df.columns = var_names

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(df.describe().T, width='stretch')

    # Time series plots
    st.subheader("Time Series")

    fig = make_subplots(
        rows=n_vars,
        cols=1,
        subplot_titles=var_names,
        vertical_spacing=0.1
    )

    for i, var in enumerate(var_names):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[var],
                mode='lines',
                name=var,
                line=dict(width=1.5)
            ),
            row=i+1,
            col=1
        )

        # Add shock marker if present
        if shock_info and i == shock_var:
            shock_idx = shock_time
            fig.add_vline(
                x=df.index[shock_idx],
                line_dash="dash",
                line_color="red",
                opacity=0.5,
                row=i+1,
                col=1
            )

    fig.update_layout(
        height=250 * n_vars,
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig, width='stretch')

    # Equation/Formula view for presentations
    st.subheader("📊 Presentation View (with Equations)")

    # Build VAR equations
    equations = []
    for i, var in enumerate(var_names):
        eq_terms = []
        for lag in range(var_order):
            for j, lag_var in enumerate(var_names):
                coef = coef_matrices[lag][i, j]
                if abs(coef) > 0.001:  # Only show non-zero coefficients
                    sign = '+' if coef >= 0 else ''
                    eq_terms.append(f"{sign}{coef:.3f}·{lag_var}(t-{lag+1})")

        eq_str = f"{var}(t) = {' '.join(eq_terms)}" if eq_terms else f"{var}(t) = ε(t)"
        equations.append(eq_str)

    formula_str = " | ".join(equations)

    if shock_info:
        formula_str += f" | Shock: {shock_magnitude:.1f}σ to {var_names[shock_var]} at t={shock_time}"

    # Create presentation figure
    fig_presentation = make_subplots(
        rows=n_vars,
        cols=1,
        subplot_titles=var_names,
        vertical_spacing=0.08
    )

    for i, var in enumerate(var_names):
        fig_presentation.add_trace(
            go.Scatter(
                x=df.index,
                y=df[var],
                mode='lines',
                name=var,
                line=dict(width=2)
            ),
            row=i+1,
            col=1
        )

    fig_presentation.update_layout(
        title={
            'text': f"<b>VAR({var_order}): {' | '.join(equations[:2])}</b>",  # Show first 2 equations
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14}
        },
        height=200 * n_vars,
        showlegend=False,
        template='plotly_white',
        font=dict(size=11)
    )

    st.plotly_chart(fig_presentation, width='stretch')
    st.caption("💡 Right-click chart → 'Save image as...' to export for presentations")

    with st.expander("Full VAR System Equations"):
        for eq in equations:
            st.code(eq, language=None)

    # Cross-correlation
    st.subheader("Cross-Correlation Matrix")
    corr_matrix = df.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=var_names,
        y=var_names,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))

    fig_corr.update_layout(
        title="Variable Correlations",
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig_corr, use_container_width=True)

    # Impulse Response Functions
    st.subheader("Impulse Response Functions (IRFs)")

    st.markdown("**IRFs show how each variable responds to a shock in another variable**")

    shock_to = st.selectbox(
        "Shock to variable:",
        list(range(n_vars)),
        format_func=lambda x: var_names[x],
        key="irf_shock_var"
    )

    irf_periods = st.slider(
        "IRF Periods",
        min_value=10,
        max_value=50,
        value=20,
        help="Number of periods to plot for IRF"
    )

    # Compute IRF
    irf = compute_irf(
        coef_matrices=coef_matrices,
        sigma=sigma,
        periods=irf_periods,
        shock_var=shock_to
    )

    # Plot IRFs
    fig_irf = make_subplots(
        rows=1,
        cols=n_vars,
        subplot_titles=[f"Response of {name}" for name in var_names],
        horizontal_spacing=0.1
    )

    for i in range(n_vars):
        fig_irf.add_trace(
            go.Scatter(
                x=list(range(irf_periods)),
                y=irf[:, i],
                mode='lines+markers',
                name=f"{var_names[i]} response",
                line=dict(width=2)
            ),
            row=1,
            col=i+1
        )

        fig_irf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=i+1)

    fig_irf.update_xaxes(title_text="Periods")
    fig_irf.update_yaxes(title_text="Response")
    fig_irf.update_layout(
        title=f"Response to 1 SD shock in {var_names[shock_to]}",
        height=400,
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig_irf, use_container_width=True)

    # Save dataset section
    st.markdown("---")
    metadata = {
        'model': f'VAR({var_order})',
        'n_vars': n_vars,
        'var_names': var_names,
        'var_order': var_order,
        'coefficient_matrices': [mat.tolist() for mat in coef_matrices],
        'is_stable': is_stable
    }
    if shock_info:
        metadata['shock_var'] = shock_info['variable_name']
        metadata['shock_time'] = shock_info['time']
        metadata['shock_magnitude'] = shock_info['magnitude']

    create_save_widget(
        data=df,
        dataset_type='multivariate',
        default_name=f"VAR{var_order}_{n_vars}vars",
        metadata=metadata
    )

except Exception as e:
    st.error(f"Error simulating VAR: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

# Help section
with st.expander("ℹ️ About VAR Models"):
    st.markdown("""
    ### Vector Autoregression (VAR)

    A VAR(p) model specifies that each variable depends on:
    - Its own past p values
    - Past p values of all other variables in the system

    **Model equation for VAR(1)**:
    ```
    Y₁(t) = a₁₁·Y₁(t-1) + a₁₂·Y₂(t-1) + ... + ε₁(t)
    Y₂(t) = a₂₁·Y₁(t-1) + a₂₂·Y₂(t-1) + ... + ε₂(t)
    ...
    ```

    ### Coefficient Interpretation

    - **Diagonal elements** (aᵢᵢ): Own-variable persistence
    - **Off-diagonal elements** (aᵢⱼ, i≠j): Cross-variable effects

    ### Stability

    A VAR is **stable** if all eigenvalues of the companion matrix are < 1.0.
    Unstable VARs can produce explosive series.

    ### Impulse Response Functions (IRFs)

    IRFs show how shocks propagate through the system:
    - **Direct effect**: Immediate response of shocked variable
    - **Spillover effects**: How shocks transmit to other variables
    - **Persistence**: How long shocks last

    ### Tips

    - Keep coefficients small (0.1-0.5) for stability
    - Diagonal elements control persistence
    - Off-diagonal elements create spillovers
    - Use IRFs to understand shock transmission
    """)

# Sidebar tips
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Tips")
st.sidebar.markdown("""
- Keep coefficients < 0.5 for stability
- Diagonal = persistence
- Off-diagonal = spillovers
- Use IRFs to see shock propagation
- Add shocks to study dynamics
""")
