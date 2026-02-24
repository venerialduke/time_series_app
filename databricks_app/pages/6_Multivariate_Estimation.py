"""
Multivariate Time Series Estimation

Estimate VAR models on saved or uploaded multivariate time series data.
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

from utils.data_manager import (
    initialize_data_storage,
    list_datasets,
    get_dataset,
    display_dataset_info
)
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

st.set_page_config(page_title="Multivariate Estimation", layout="wide")

st.title("Multivariate Time Series Estimation")

st.markdown("""
Estimate VAR (Vector Autoregression) models on your saved multivariate time series data.
statsmodels VAR handles missing values automatically.
""")

# Initialize data storage
initialize_data_storage()

# Initialize session state for workflow
if 'mv_estimation_data' not in st.session_state:
    st.session_state.mv_estimation_data = None
    print("🔧 [MV-INIT] Initialized mv_estimation_data to None")
if 'mv_transformed_data' not in st.session_state:
    st.session_state.mv_transformed_data = None
    print("🔧 [MV-INIT] Initialized mv_transformed_data to None")

# Custom load widget
st.subheader("📂 Load Dataset")

available_datasets = list_datasets('multivariate')

if not available_datasets:
    st.info("No saved multivariate datasets found. Generate data from an exploration page first.")
    print("⚠️  [MV-LOAD] No datasets available")
else:
    selected_dataset = st.selectbox(
        "Select Dataset",
        available_datasets,
        help="Choose a saved dataset to load",
        key='mv_dataset_selector'
    )

    print(f"📊 [MV-LOAD] Selected dataset: {selected_dataset}")

    if selected_dataset:
        dataset_info = get_dataset(selected_dataset)

        with st.expander("Dataset Info"):
            display_dataset_info(dataset_info)

        if st.button("Load Dataset", type="primary", key='mv_load_dataset_button'):
            print(f"🔄 [MV-LOAD] Load button clicked for: {selected_dataset}")
            st.session_state.mv_estimation_data = dataset_info['data'].copy()
            st.session_state.mv_transformed_data = dataset_info['data'].copy()
            print(f"✅ [MV-LOAD] Data loaded into session state. Shape: {dataset_info['data'].shape}")
            st.success(f"✓ Loaded dataset '{selected_dataset}'")
            st.rerun()

if st.session_state.mv_estimation_data is not None:
    print(f"✅ [MV-WORKFLOW] Data present in session. Shape: {st.session_state.mv_estimation_data.shape}")

    # Add reset button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Reset All", help="Clear all data and start over"):
            print("🔄 [MV-RESET] Clearing all session state")
            st.session_state.mv_estimation_data = None
            st.session_state.mv_transformed_data = None
            st.rerun()
    st.markdown("---")

    # Current working data
    current_data = st.session_state.mv_transformed_data.copy()
    print(f"📈 [MV-WORKFLOW] Working with data, shape: {current_data.shape}")

    # Add missing values if requested
    st.subheader("Add Missing Values (Optional)")

    add_missing = st.checkbox("Add random blocks of missing values", value=False, key='mv_add_missing_check')

    if add_missing:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            missing_pct = st.slider(
                "Missing %",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key='mv_missing_pct'
            ) / 100

        with col_b:
            block_size = st.number_input(
                "Block Size",
                min_value=1,
                max_value=20,
                value=5,
                key='mv_block_size'
            )

        with col_c:
            seed_missing = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=10000,
                value=42,
                key='mv_seed_missing'
            )

        if st.button("Apply Missing Values", key='mv_apply_missing'):
            from utils.data_manager import add_missing_values
            current_data = add_missing_values(current_data, missing_pct, block_size, seed_missing)
            st.session_state.mv_transformed_data = current_data
            st.success(f"✓ Added missing values")
            st.rerun()

    data_to_estimate = current_data

    st.markdown("---")

    # Display the data
    st.subheader("Data Preview")

    n_vars = data_to_estimate.shape[1]
    n_obs = len(data_to_estimate)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Variables", n_vars)
    with col2:
        st.metric("Observations", n_obs)
    with col3:
        n_missing = data_to_estimate.isna().sum().sum()
        st.metric("Missing Values", n_missing)
    with col4:
        missing_pct = (n_missing / (n_obs * n_vars)) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")

    # Plot the data
    fig_data = make_subplots(
        rows=n_vars,
        cols=1,
        subplot_titles=data_to_estimate.columns.tolist(),
        vertical_spacing=0.08
    )

    for i, col in enumerate(data_to_estimate.columns):
        fig_data.add_trace(
            go.Scatter(
                x=data_to_estimate.index,
                y=data_to_estimate[col],
                mode='lines',
                name=col,
                line=dict(width=1.5)
            ),
            row=i+1,
            col=1
        )

    fig_data.update_layout(
        height=250 * n_vars,
        showlegend=False,
        template='plotly_white'
    )

    st.plotly_chart(fig_data, width='stretch')

    st.markdown("---")

    # Model specification
    st.subheader("VAR Model Specification")

    col1, col2 = st.columns([1, 2])

    with col1:
        order_selection = st.radio(
            "Order Selection",
            ["Manual", "Auto (AIC)", "Auto (BIC)"],
            help="Manual: specify lag order. Auto: select by information criterion",
            key='mv_order_selection'
        )

    with col2:
        if order_selection == "Manual":
            var_order = st.number_input(
                "VAR Lag Order (p)",
                min_value=1,
                max_value=10,
                value=1,
                help="Number of lags in the VAR model",
                key='mv_var_order'
            )
        else:
            max_lags = st.slider(
                "Maximum Lags to Consider",
                min_value=1,
                max_value=15,
                value=8,
                help="Upper bound on lag order for selection",
                key='mv_max_lags'
            )

    # Fit button
    if st.button("Estimate VAR Model", type="primary", key='mv_estimate_button'):
        print(f"🎯 [MV-ESTIMATE] Starting VAR estimation. Order selection: {order_selection}")
        with st.spinner("Fitting VAR model..."):
            try:
                # Prepare data - drop rows with any missing values for VAR
                # Note: statsmodels VAR requires complete data
                y_complete = data_to_estimate.dropna()
                print(f"🎯 [MV-ESTIMATE] Original data: {len(data_to_estimate)}, Complete: {len(y_complete)}")

                if len(y_complete) < 50:
                    st.warning(f"Only {len(y_complete)} complete observations after dropping missing values. Results may be unreliable.")

                # Create VAR model
                model = VAR(y_complete)

                if order_selection == "Manual":
                    # Fit specified model
                    result = model.fit(var_order)
                    st.success(f"✓ Fitted VAR({var_order})")

                elif order_selection == "Auto (AIC)":
                    # Select by AIC
                    result = model.fit(maxlags=max_lags, ic='aic')
                    selected_order = result.k_ar
                    st.success(f"✓ Selected VAR({selected_order}) by AIC")
                    var_order = selected_order

                else:  # Auto (BIC)
                    # Select by BIC
                    result = model.fit(maxlags=max_lags, ic='bic')
                    selected_order = result.k_ar
                    st.success(f"✓ Selected VAR({selected_order}) by BIC")
                    var_order = selected_order

                # Display results
                st.markdown("---")
                st.subheader("Estimation Results")

                # Model summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Lag Order", var_order)
                with col2:
                    st.metric("AIC", f"{result.aic:.2f}")
                with col3:
                    st.metric("BIC", f"{result.bic:.2f}")
                with col4:
                    n_params = result.params.size
                    st.metric("Total Parameters", n_params)

                # Show order selection results if auto
                if order_selection != "Manual":
                    with st.expander("Information Criteria by Lag Order"):
                        ic_df = pd.DataFrame({
                            'Lag': list(range(1, max_lags + 1)),
                            'AIC': [model.fit(p).aic for p in range(1, max_lags + 1)],
                            'BIC': [model.fit(p).bic for p in range(1, max_lags + 1)]
                        })
                        st.dataframe(ic_df.style.format({
                            'AIC': '{:.2f}',
                            'BIC': '{:.2f}'
                        }), width='stretch')

                # Parameter estimates
                st.subheader("Parameter Estimates")

                # Display equation-by-equation
                for eq_idx, eq_name in enumerate(data_to_estimate.columns):
                    with st.expander(f"Equation for {eq_name}"):
                        eq_results = result.params.iloc[:, eq_idx]
                        eq_stderr = result.stderr.iloc[:, eq_idx]
                        eq_tvalues = result.tvalues.iloc[:, eq_idx]

                        eq_df = pd.DataFrame({
                            'Coefficient': eq_results,
                            'Std Error': eq_stderr,
                            't-value': eq_tvalues
                        })

                        st.dataframe(eq_df.style.format({
                            'Coefficient': '{:.4f}',
                            'Std Error': '{:.4f}',
                            't-value': '{:.3f}'
                        }), width='stretch')

                # Fitted values
                st.subheader("Fitted Values vs Actual")

                fitted_values = result.fittedvalues

                fig_fit = make_subplots(
                    rows=n_vars,
                    cols=1,
                    subplot_titles=[f"{col} - Actual vs Fitted" for col in data_to_estimate.columns],
                    vertical_spacing=0.08
                )

                for i, col in enumerate(data_to_estimate.columns):
                    # Actual
                    fig_fit.add_trace(
                        go.Scatter(
                            x=y_complete.index,
                            y=y_complete[col],
                            mode='lines',
                            name=f'{col} Actual',
                            line=dict(color='blue', width=1.5)
                        ),
                        row=i+1,
                        col=1
                    )

                    # Fitted
                    fig_fit.add_trace(
                        go.Scatter(
                            x=fitted_values.index,
                            y=fitted_values[col],
                            mode='lines',
                            name=f'{col} Fitted',
                            line=dict(color='red', width=1.5, dash='dot')
                        ),
                        row=i+1,
                        col=1
                    )

                fig_fit.update_layout(
                    height=250 * n_vars,
                    showlegend=False,
                    template='plotly_white'
                )

                st.plotly_chart(fig_fit, width='stretch')

                # Residuals
                st.subheader("Residuals")

                residuals = result.resid

                fig_resid = make_subplots(
                    rows=n_vars,
                    cols=1,
                    subplot_titles=[f"{col} Residuals" for col in data_to_estimate.columns],
                    vertical_spacing=0.08
                )

                for i, col in enumerate(data_to_estimate.columns):
                    fig_resid.add_trace(
                        go.Scatter(
                            x=residuals.index,
                            y=residuals[col],
                            mode='lines',
                            name=f'{col} Residuals',
                            line=dict(color='black', width=1)
                        ),
                        row=i+1,
                        col=1
                    )
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=i+1, col=1)

                fig_resid.update_layout(
                    height=250 * n_vars,
                    showlegend=False,
                    template='plotly_white'
                )

                st.plotly_chart(fig_resid, width='stretch')

                # Residual correlation matrix
                st.subheader("Residual Cross-Correlation Matrix")

                resid_corr = residuals.corr()

                fig_corr = go.Figure(data=go.Heatmap(
                    z=resid_corr.values,
                    x=resid_corr.columns,
                    y=resid_corr.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=resid_corr.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 11},
                    colorbar=dict(title="Correlation"),
                    zmin=-1,
                    zmax=1
                ))

                fig_corr.update_layout(
                    title="Residual Correlations",
                    template='plotly_white',
                    height=400
                )

                st.plotly_chart(fig_corr, width='stretch')
                st.caption("Ideally residuals should be uncorrelated across variables")

                # Granger causality
                st.subheader("Granger Causality Tests")

                st.markdown("Test if one variable Granger-causes another")

                test_var1 = st.selectbox(
                    "Causing variable",
                    data_to_estimate.columns.tolist(),
                    key='granger_cause'
                )

                test_var2 = st.selectbox(
                    "Caused variable",
                    data_to_estimate.columns.tolist(),
                    key='granger_caused'
                )

                if test_var1 != test_var2:
                    gc_test = result.test_causality(test_var2, [test_var1], kind='f')

                    st.write(f"**Null Hypothesis**: {test_var1} does NOT Granger-cause {test_var2}")
                    st.write(f"**Test Statistic**: {gc_test.test_statistic:.4f}")
                    st.write(f"**p-value**: {gc_test.pvalue:.4f}")

                    if gc_test.pvalue < 0.05:
                        st.success(f"✓ Reject null: {test_var1} Granger-causes {test_var2} (p < 0.05)")
                    else:
                        st.info(f"✗ Cannot reject null: {test_var1} does not Granger-cause {test_var2} (p ≥ 0.05)")
                else:
                    st.warning("Select different variables for Granger causality test")

                # Impulse Response Functions
                st.subheader("Impulse Response Functions (IRFs)")

                irf_periods = st.slider(
                    "IRF Periods",
                    min_value=5,
                    max_value=30,
                    value=10,
                    help="Number of periods for IRF",
                    key='mv_irf_periods'
                )

                shock_var = st.selectbox(
                    "Shock Variable",
                    data_to_estimate.columns.tolist(),
                    key='irf_shock'
                )

                irf = result.irf(irf_periods)

                # Plot IRF for all variables responding to the shock
                fig_irf = make_subplots(
                    rows=1,
                    cols=n_vars,
                    subplot_titles=[f"Response of {col}" for col in data_to_estimate.columns],
                    horizontal_spacing=0.1
                )

                shock_idx = data_to_estimate.columns.tolist().index(shock_var)

                for i, col in enumerate(data_to_estimate.columns):
                    irf_values = irf.irfs[:, i, shock_idx]

                    fig_irf.add_trace(
                        go.Scatter(
                            x=list(range(irf_periods)),
                            y=irf_values,
                            mode='lines+markers',
                            name=f'{col} response',
                            line=dict(width=2)
                        ),
                        row=1,
                        col=i+1
                    )

                    fig_irf.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=i+1)

                fig_irf.update_xaxes(title_text="Periods")
                fig_irf.update_yaxes(title_text="Response")
                fig_irf.update_layout(
                    title=f"IRF: Response to shock in {shock_var}",
                    height=400,
                    showlegend=False,
                    template='plotly_white'
                )

                st.plotly_chart(fig_irf, width='stretch')

                # Forecast
                st.subheader("Forecast")

                forecast_steps = st.slider(
                    "Forecast Horizon",
                    min_value=1,
                    max_value=20,
                    value=5,
                    help="Number of steps ahead to forecast",
                    key='mv_forecast_steps'
                )

                forecast = result.forecast(y_complete.values[-var_order:], steps=forecast_steps)
                forecast_df = pd.DataFrame(
                    forecast,
                    columns=data_to_estimate.columns,
                    index=pd.date_range(
                        start=y_complete.index[-1],
                        periods=forecast_steps + 1,
                        freq=y_complete.index.freq
                    )[1:]
                )

                fig_forecast = make_subplots(
                    rows=n_vars,
                    cols=1,
                    subplot_titles=[f"{col} - Forecast" for col in data_to_estimate.columns],
                    vertical_spacing=0.08
                )

                for i, col in enumerate(data_to_estimate.columns):
                    # Historical
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=y_complete.index,
                            y=y_complete[col],
                            mode='lines',
                            name=f'{col} Historical',
                            line=dict(color='blue', width=1.5)
                        ),
                        row=i+1,
                        col=1
                    )

                    # Forecast
                    fig_forecast.add_trace(
                        go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df[col],
                            mode='lines+markers',
                            name=f'{col} Forecast',
                            line=dict(color='red', width=2, dash='dash'),
                            marker=dict(size=6)
                        ),
                        row=i+1,
                        col=1
                    )

                fig_forecast.update_layout(
                    height=250 * n_vars,
                    showlegend=False,
                    template='plotly_white'
                )

                st.plotly_chart(fig_forecast, width='stretch')

            except Exception as e:
                st.error(f"Error fitting VAR model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("👈 Load a saved multivariate dataset from the sidebar to begin estimation.")
    st.markdown("""
    ### How to use:

    1. **Generate data** from any multivariate exploration page (VAR, DFM)
    2. **Save the dataset** using the save widget at the bottom of the page
    3. **Come back here** and load the saved dataset
    4. **Optionally add missing values** to test model robustness
    5. **Specify model** manually or use automatic lag selection
    6. **Estimate** and review diagnostics, IRFs, and forecasts
    """)

# Help section
with st.expander("ℹ️ About VAR Estimation"):
    st.markdown("""
    ### Vector Autoregression (VAR)

    This page estimates VAR models using statsmodels.

    **VAR Model:**
    - Each variable is regressed on lagged values of itself and all other variables
    - VAR(p) includes p lags of each variable

    **Order Selection:**
    - **Manual**: Specify lag order directly
    - **Auto (AIC/BIC)**: Select lag that minimizes information criterion

    **Missing Values:**
    - statsmodels VAR requires complete data
    - Rows with missing values are automatically dropped
    - Be aware that this can significantly reduce sample size

    **Diagnostics:**
    - **AIC/BIC**: Lower is better
    - **Parameter significance**: Check t-values
    - **Residual correlations**: Should be near zero
    - **Granger causality**: Test if one variable helps predict another

    **Impulse Response Functions (IRFs):**
    - Show how a shock to one variable affects all variables over time
    - Used to understand dynamic relationships and shock transmission

    **Interpretation:**
    - **Good model**: Low AIC/BIC, significant parameters, uncorrelated residuals
    - **Poor model**: High AIC/BIC, many insignificant parameters, correlated residuals
    - **Granger causality**: Helps identify lead-lag relationships

    ### Tips
    - Start with automatic lag selection
    - Check if residuals are white noise (no autocorrelation)
    - Use IRFs to understand shock propagation
    - Test Granger causality for economic relationships
    - Be careful with missing values - they reduce effective sample size
    """)
