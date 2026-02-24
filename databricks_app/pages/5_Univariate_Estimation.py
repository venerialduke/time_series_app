"""
Univariate Time Series Estimation

Estimate ARMA/ARIMA models on saved or uploaded univariate time series data.
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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf, adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import signal

st.set_page_config(page_title="Univariate Estimation", layout="wide")

st.title("Univariate Time Series Estimation")

st.markdown("""
Estimate ARMA/ARIMA models on your saved time series data.
statsmodels handles missing values automatically using the Kalman filter.
""")

# Initialize data storage
initialize_data_storage()

# Initialize session state for workflow
if 'estimation_data' not in st.session_state:
    st.session_state.estimation_data = None
    print("🔧 [INIT] Initialized estimation_data to None")
if 'transformed_data' not in st.session_state:
    st.session_state.transformed_data = None
    print("🔧 [INIT] Initialized transformed_data to None")
if 'transformation_steps' not in st.session_state:
    st.session_state.transformation_steps = []
    print("🔧 [INIT] Initialized transformation_steps to []")

# Custom load widget that works with session state
st.subheader("📂 Load Dataset")

available_datasets = list_datasets('univariate')

if not available_datasets:
    st.info("No saved univariate datasets found. Generate data from an exploration page first.")
    print("⚠️  [LOAD] No datasets available")
else:
    selected_dataset = st.selectbox(
        "Select Dataset",
        available_datasets,
        help="Choose a saved dataset to load",
        key='dataset_selector'
    )

    print(f"📊 [LOAD] Selected dataset: {selected_dataset}")

    if selected_dataset:
        dataset_info = get_dataset(selected_dataset)

        with st.expander("Dataset Info"):
            display_dataset_info(dataset_info)

        if st.button("Load Dataset", type="primary", key='load_dataset_button'):
            print(f"🔄 [LOAD] Load button clicked for: {selected_dataset}")
            st.session_state.estimation_data = dataset_info['data'].copy()
            st.session_state.transformed_data = dataset_info['data'].copy()
            st.session_state.transformation_steps = []
            # Clear test results
            if 'adf_result' in st.session_state:
                del st.session_state.adf_result
            if 'kpss_result' in st.session_state:
                del st.session_state.kpss_result
            print(f"✅ [LOAD] Data loaded into session state. Shape: {dataset_info['data'].shape}")
            st.success(f"✓ Loaded dataset '{selected_dataset}'")
            st.rerun()

if st.session_state.estimation_data is not None:
    print(f"✅ [WORKFLOW] Data present in session. Shape: {st.session_state.estimation_data.shape}")

    # Add reset button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🔄 Reset All", help="Clear all data and start over"):
            print("🔄 [RESET] Clearing all session state")
            st.session_state.estimation_data = None
            st.session_state.transformed_data = None
            st.session_state.transformation_steps = []
            if 'adf_result' in st.session_state:
                del st.session_state.adf_result
            if 'kpss_result' in st.session_state:
                del st.session_state.kpss_result
            if 'model_result' in st.session_state:
                del st.session_state.model_result
            if 'model_order' in st.session_state:
                del st.session_state.model_order
            if 'model_data' in st.session_state:
                del st.session_state.model_data
            st.rerun()

    st.markdown("---")

    # Current working data
    current_data = st.session_state.transformed_data.copy()
    series_col = current_data.columns[0]
    print(f"📈 [WORKFLOW] Working with series: {series_col}, shape: {current_data.shape}")

    # Step 1: Add missing values if requested
    st.header("Step 1: Add Missing Values (Optional)")

    col1, col2 = st.columns([2, 1])

    with col1:
        add_missing = st.checkbox("Add random blocks of missing values", value=False, key='add_missing_check')

    if add_missing:
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            missing_pct = st.slider(
                "Missing %",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key='missing_pct'
            ) / 100

        with col_b:
            block_size = st.number_input(
                "Block Size",
                min_value=1,
                max_value=20,
                value=5,
                key='block_size'
            )

        with col_c:
            seed_missing = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=10000,
                value=42,
                key='seed_missing'
            )

        if st.button("Apply Missing Values"):
            print(f"🔲 [MISSING] Adding {missing_pct*100:.0f}% missing values")
            from utils.data_manager import add_missing_values
            current_data = add_missing_values(current_data, missing_pct, block_size, seed_missing)
            st.session_state.transformed_data = current_data
            st.session_state.transformation_steps.append(f"Added {missing_pct*100:.0f}% missing values")
            print(f"✅ [MISSING] Missing values added. New shape: {current_data.shape}")
            st.success(f"✓ Added missing values")
            st.rerun()

    data_to_estimate = current_data

    # Show transformation history
    if st.session_state.transformation_steps:
        st.info("**Transformations Applied**: " + " → ".join(st.session_state.transformation_steps))

    # Display the data
    st.markdown("---")
    st.header("Data Preview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Observations", len(data_to_estimate))
    with col2:
        n_missing = data_to_estimate.isna().sum().sum()
        st.metric("Missing Values", n_missing)
    with col3:
        missing_pct = (n_missing / len(data_to_estimate)) * 100
        st.metric("Missing %", f"{missing_pct:.1f}%")

    # Plot the data
    fig_data = go.Figure()
    series_col = data_to_estimate.columns[0]

    fig_data.add_trace(go.Scatter(
        x=data_to_estimate.index,
        y=data_to_estimate[series_col],
        mode='lines+markers',
        name='Data',
        line=dict(color='blue'),
        marker=dict(size=3)
    ))

    fig_data.update_layout(
        title="Time Series Data",
        xaxis_title="Time",
        yaxis_title="Value",
        template='plotly_white',
        height=400
    )

    st.plotly_chart(fig_data, width='stretch')

    st.markdown("---")

    # Step 2: Stationarity Tests
    st.header("Step 2: Check Stationarity")

    st.markdown("""
    Before estimating ARMA models, the series should be stationary.
    Run tests to check stationarity, then apply transformations if needed.
    """)

    if st.button("Run Stationarity Tests", type="primary"):
        print("🧪 [TEST] Running stationarity tests...")
        # Store test results in session state
        series_clean = data_to_estimate[series_col].dropna()
        print(f"🧪 [TEST] Clean series length: {len(series_clean)}")

        # ADF Test
        adf_result = adfuller(series_clean, autolag='AIC')
        print(f"🧪 [TEST] ADF: statistic={adf_result[0]:.4f}, p-value={adf_result[1]:.4f}")

        # KPSS Test
        kpss_result = kpss(series_clean, regression='ct', nlags='auto')
        print(f"🧪 [TEST] KPSS: statistic={kpss_result[0]:.4f}, p-value={kpss_result[1]:.4f}")

        st.session_state.adf_result = {
            'statistic': adf_result[0],
            'pvalue': adf_result[1],
            'critical_values': adf_result[4]
        }

        st.session_state.kpss_result = {
            'statistic': kpss_result[0],
            'pvalue': kpss_result[1],
            'critical_values': kpss_result[3]
        }
        print("✅ [TEST] Test results stored in session state")

    # Display test results if available
    if 'adf_result' in st.session_state and 'kpss_result' in st.session_state:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ADF Test (Augmented Dickey-Fuller)")
            adf = st.session_state.adf_result

            st.write(f"**Test Statistic**: {adf['statistic']:.4f}")
            st.write(f"**p-value**: {adf['pvalue']:.4f}")

            if adf['pvalue'] < 0.05:
                st.success("✓ **Stationary** (reject unit root, p < 0.05)")
            else:
                st.warning("⚠️ **Non-stationary** (cannot reject unit root, p ≥ 0.05)")

            st.write("**Critical Values**:")
            for key, value in adf['critical_values'].items():
                st.write(f"  - {key}: {value:.3f}")

        with col2:
            st.subheader("KPSS Test (Kwiatkowski-Phillips-Schmidt-Shin)")
            kpss_res = st.session_state.kpss_result

            st.write(f"**Test Statistic**: {kpss_res['statistic']:.4f}")
            st.write(f"**p-value**: {kpss_res['pvalue']:.4f}")

            if kpss_res['pvalue'] >= 0.05:
                st.success("✓ **Stationary** (cannot reject stationarity, p ≥ 0.05)")
            else:
                st.warning("⚠️ **Non-stationary** (reject stationarity, p < 0.05)")

            st.write("**Critical Values**:")
            for key, value in kpss_res['critical_values'].items():
                st.write(f"  - {key}: {value:.3f}")

        st.info("""
        **Interpretation**:
        - **Both tests say stationary**: Series is likely stationary ✓
        - **Both tests say non-stationary**: Series needs transformation (difference or detrend)
        - **Mixed results**: Check plots and use judgment
        """)

    st.markdown("---")

    # Step 3: Transformations
    st.header("Step 3: Apply Transformations (If Needed)")

    st.markdown("""
    If the series is non-stationary, apply transformations:
    - **Differencing**: Removes stochastic trends (random walks)
    - **Detrending**: Removes deterministic trends (linear/quadratic)
    """)

    transformation_type = st.radio(
        "Transformation Type",
        ["None", "First Difference", "Second Difference", "Linear Detrend", "Reset to Original"],
        key='transformation_type'
    )

    if st.button("Apply Transformation"):
        print(f"🔄 [TRANSFORM] Applying transformation: {transformation_type}")
        if transformation_type == "None":
            print("⚠️  [TRANSFORM] No transformation selected")
            st.info("No transformation selected")
        elif transformation_type == "Reset to Original":
            print("🔄 [TRANSFORM] Resetting to original data")
            st.session_state.transformed_data = st.session_state.estimation_data.copy()
            st.session_state.transformation_steps = []
            st.success("✓ Reset to original data")
            st.rerun()
        elif transformation_type == "First Difference":
            print("📉 [TRANSFORM] Applying first difference")
            transformed = data_to_estimate[series_col].diff().to_frame(series_col)
            st.session_state.transformed_data = transformed
            st.session_state.transformation_steps.append("First Difference")
            # Clear test results to force re-testing
            if 'adf_result' in st.session_state:
                del st.session_state.adf_result
            if 'kpss_result' in st.session_state:
                del st.session_state.kpss_result
            print(f"✅ [TRANSFORM] First difference applied. Shape: {transformed.shape}")
            st.success("✓ Applied first difference - re-run stationarity tests!")
            st.rerun()
        elif transformation_type == "Second Difference":
            print("📉 [TRANSFORM] Applying second difference")
            transformed = data_to_estimate[series_col].diff().diff().to_frame(series_col)
            st.session_state.transformed_data = transformed
            st.session_state.transformation_steps.append("Second Difference")
            if 'adf_result' in st.session_state:
                del st.session_state.adf_result
            if 'kpss_result' in st.session_state:
                del st.session_state.kpss_result
            print(f"✅ [TRANSFORM] Second difference applied. Shape: {transformed.shape}")
            st.success("✓ Applied second difference - re-run stationarity tests!")
            st.rerun()
        elif transformation_type == "Linear Detrend":
            print("📊 [TRANSFORM] Applying linear detrend")
            series_vals = data_to_estimate[series_col].dropna()
            detrended = signal.detrend(series_vals.values, type='linear')
            transformed = pd.DataFrame(
                detrended,
                index=series_vals.index,
                columns=[series_col]
            )
            # Reindex to original to preserve NaNs
            transformed = transformed.reindex(data_to_estimate.index)
            st.session_state.transformed_data = transformed
            st.session_state.transformation_steps.append("Linear Detrend")
            if 'adf_result' in st.session_state:
                del st.session_state.adf_result
            if 'kpss_result' in st.session_state:
                del st.session_state.kpss_result
            print(f"✅ [TRANSFORM] Linear detrend applied. Shape: {transformed.shape}")
            st.success("✓ Applied linear detrend - re-run stationarity tests!")
            st.rerun()

    st.markdown("---")

    # Step 4: Model specification
    st.header("Step 4: Model Specification")

    col1, col2 = st.columns([1, 2])

    with col1:
        model_type = st.selectbox(
            "Model Type",
            ["ARMA", "ARIMA"],
            help="ARMA for stationary series, ARIMA for non-stationary"
        )

        order_selection = st.radio(
            "Order Selection",
            ["Manual", "Auto (AIC)"],
            help="Manual: specify orders. Auto: search for best orders by AIC"
        )

    with col2:
        if order_selection == "Manual":
            col_p, col_d, col_q = st.columns(3)

            with col_p:
                p = st.number_input(
                    "AR Order (p)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Number of autoregressive lags"
                )

            with col_d:
                if model_type == "ARIMA":
                    d = st.number_input(
                        "Difference Order (d)",
                        min_value=0,
                        max_value=2,
                        value=0,
                        help="Number of differences for stationarity"
                    )
                else:
                    d = 0
                    st.info("d=0 for ARMA")

            with col_q:
                q = st.number_input(
                    "MA Order (q)",
                    min_value=0,
                    max_value=5,
                    value=1,
                    help="Number of moving average lags"
                )

        else:  # Auto selection
            st.info("Searching over p, q ∈ {0, 1, 2, 3} and d ∈ {0, 1, 2} (if ARIMA)")

            max_p = 3
            max_q = 3
            max_d = 2 if model_type == "ARIMA" else 0

    # Fit button
    st.markdown("---")
    st.header("Step 5: Estimate Model")

    if st.button("Estimate Model", type="primary", key='estimate_button'):
        print(f"🎯 [ESTIMATE] Starting model estimation. Order selection: {order_selection}")
        with st.spinner("Fitting model..."):
            try:
                # Prepare data
                y = data_to_estimate[series_col]
                print(f"🎯 [ESTIMATE] Series length: {len(y)}, missing: {y.isna().sum()}")

                if order_selection == "Manual":
                    print(f"🎯 [ESTIMATE] Manual order: ARIMA({p},{d},{q})")
                    # Fit specified model
                    model = ARIMA(y, order=(p, d, q))
                    result = model.fit()

                    st.success(f"✓ Fitted ARIMA({p},{d},{q})")

                else:  # Auto selection
                    # Grid search over orders
                    best_aic = np.inf
                    best_order = None
                    best_result = None

                    progress_bar = st.progress(0)
                    n_models = (max_p + 1) * (max_d + 1) * (max_q + 1)
                    model_count = 0

                    for p_try in range(max_p + 1):
                        for d_try in range(max_d + 1):
                            for q_try in range(max_q + 1):
                                if p_try == 0 and q_try == 0:
                                    continue  # Skip (0,d,0)

                                try:
                                    model = ARIMA(y, order=(p_try, d_try, q_try))
                                    result_try = model.fit()

                                    if result_try.aic < best_aic:
                                        best_aic = result_try.aic
                                        best_order = (p_try, d_try, q_try)
                                        best_result = result_try

                                except:
                                    pass  # Skip failed fits

                                model_count += 1
                                progress_bar.progress(model_count / n_models)

                    progress_bar.empty()

                    if best_result is None:
                        st.error("Could not fit any models. Try manual specification.")
                    else:
                        result = best_result
                        p, d, q = best_order
                        st.success(f"✓ Best model: ARIMA({p},{d},{q}) with AIC={best_aic:.2f}")

                # Store results in session state for persistence
                st.session_state.model_result = result
                st.session_state.model_order = (p, d, q)
                st.session_state.model_data = y
                print(f"✅ [ESTIMATE] Model fitted and stored in session state")

                # Display results
                st.markdown("---")
                st.subheader("Estimation Results")

                # Model summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("AIC", f"{result.aic:.2f}")
                with col2:
                    st.metric("BIC", f"{result.bic:.2f}")
                with col3:
                    st.metric("Log-Likelihood", f"{result.llf:.2f}")
                with col4:
                    n_params = len(result.params)
                    st.metric("Parameters", n_params)

                # Parameter estimates
                st.subheader("Parameter Estimates")

                params_df = pd.DataFrame({
                    'Coefficient': result.params,
                    'Std Error': result.bse,
                    'z-value': result.tvalues,
                    'p-value': result.pvalues
                })

                st.dataframe(params_df.style.format({
                    'Coefficient': '{:.4f}',
                    'Std Error': '{:.4f}',
                    'z-value': '{:.3f}',
                    'p-value': '{:.4f}'
                }), width='stretch')

                # Fitted values vs actual
                st.subheader("Fitted Values vs Actual")

                fig_fit = go.Figure()

                fig_fit.add_trace(go.Scatter(
                    x=y.index,
                    y=y.values,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=1.5)
                ))

                fig_fit.add_trace(go.Scatter(
                    x=result.fittedvalues.index,
                    y=result.fittedvalues.values,
                    mode='lines',
                    name='Fitted',
                    line=dict(color='red', width=1.5, dash='dot')
                ))

                fig_fit.update_layout(
                    title=f"ARIMA({p},{d},{q}) Fit",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    template='plotly_white',
                    height=400,
                    legend=dict(x=0.01, y=0.99)
                )

                st.plotly_chart(fig_fit, width='stretch')

                # Residual diagnostics
                st.subheader("Residual Diagnostics")

                residuals = result.resid

                # Create diagnostic plots
                fig_diagnostics = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Residuals Over Time',
                        'Residual Distribution',
                        'Residual ACF',
                        'Residual PACF'
                    ),
                    vertical_spacing=0.12,
                    horizontal_spacing=0.1
                )

                # Residuals over time
                fig_diagnostics.add_trace(
                    go.Scatter(
                        x=residuals.index,
                        y=residuals.values,
                        mode='lines',
                        name='Residuals',
                        line=dict(color='black', width=1)
                    ),
                    row=1, col=1
                )
                fig_diagnostics.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

                # Residual histogram
                fig_diagnostics.add_trace(
                    go.Histogram(
                        x=residuals.dropna().values,
                        name='Distribution',
                        marker_color='steelblue',
                        nbinsx=30
                    ),
                    row=1, col=2
                )

                # Residual ACF
                resid_acf = acf(residuals.dropna(), nlags=20)
                lags = list(range(len(resid_acf)))
                fig_diagnostics.add_trace(
                    go.Bar(x=lags, y=resid_acf, name='ACF', marker_color='steelblue'),
                    row=2, col=1
                )
                # Confidence bands
                conf = 1.96 / np.sqrt(len(residuals))
                fig_diagnostics.add_hline(y=conf, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
                fig_diagnostics.add_hline(y=-conf, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)

                # Residual PACF
                resid_pacf = pacf(residuals.dropna(), nlags=20)
                fig_diagnostics.add_trace(
                    go.Bar(x=lags, y=resid_pacf, name='PACF', marker_color='darkorange'),
                    row=2, col=2
                )
                fig_diagnostics.add_hline(y=conf, line_dash="dash", line_color="red", opacity=0.5, row=2, col=2)
                fig_diagnostics.add_hline(y=-conf, line_dash="dash", line_color="red", opacity=0.5, row=2, col=2)

                fig_diagnostics.update_layout(
                    height=800,
                    showlegend=False,
                    template='plotly_white'
                )

                st.plotly_chart(fig_diagnostics, width='stretch')

                # Ljung-Box test
                st.subheader("Ljung-Box Test (Residual Autocorrelation)")
                lb_test = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
                st.dataframe(lb_test.style.format({
                    'lb_stat': '{:.3f}',
                    'lb_pvalue': '{:.4f}'
                }), width='stretch')
                st.caption("**p-value > 0.05**: Residuals appear to be white noise (good)")
                st.caption("**p-value < 0.05**: Residuals show autocorrelation (may need better model)")

            except Exception as e:
                st.error(f"Error fitting model: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Forecast section (outside button block, persists across reruns)
    if 'model_result' in st.session_state and st.session_state.model_result is not None:
        st.markdown("---")
        st.subheader("Forecast")

        p, d, q = st.session_state.model_order
        result = st.session_state.model_result
        y = st.session_state.model_data

        forecast_steps = st.slider(
            "Forecast Horizon",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of steps ahead to forecast",
            key='forecast_horizon_slider'
        )

        print(f"📊 [FORECAST] Generating {forecast_steps}-step forecast")

        forecast_result = result.forecast(steps=forecast_steps)
        forecast_index = pd.date_range(
            start=y.index[-1],
            periods=forecast_steps + 1,
            freq=y.index.freq
        )[1:]

        fig_forecast = go.Figure()

        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=y.index,
            y=y.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=1.5)
        ))

        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_result,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))

        fig_forecast.update_layout(
            title=f"ARIMA({p},{d},{q}) - {forecast_steps}-Step Ahead Forecast",
            xaxis_title="Time",
            yaxis_title="Value",
            template='plotly_white',
            height=400,
            legend=dict(x=0.01, y=0.99)
        )

        st.plotly_chart(fig_forecast, width='stretch')

else:
    st.info("👈 Load a saved univariate dataset from the sidebar to begin estimation.")
    st.markdown("""
    ### How to use:

    1. **Generate data** from any univariate exploration page (ARMA, Univariate, State Space)
    2. **Save the dataset** using the save widget at the bottom of the page
    3. **Come back here** and load the saved dataset
    4. **Optionally add missing values** to test model robustness
    5. **Specify model** manually or use automatic order selection
    6. **Estimate** and review diagnostics
    """)

# Help section
with st.expander("ℹ️ About Univariate Estimation"):
    st.markdown("""
    ### ARMA/ARIMA Estimation

    This page estimates ARMA/ARIMA models using statsmodels.

    **Model Types:**
    - **ARMA(p,q)**: For stationary time series
    - **ARIMA(p,d,q)**: For non-stationary time series (d = differencing order)

    **Order Selection:**
    - **Manual**: Specify p, d, q based on ACF/PACF analysis
    - **Automatic**: Grid search over orders, select model with minimum AIC

    **Missing Values:**
    - statsmodels ARIMA uses Kalman filter to handle missing values automatically
    - No need for imputation or special handling

    **Diagnostics:**
    - **AIC/BIC**: Lower is better (balance fit and complexity)
    - **Parameter p-values**: Check if coefficients are significant
    - **Residual ACF/PACF**: Should show no significant autocorrelation
    - **Ljung-Box test**: Tests for residual autocorrelation (want p > 0.05)
    - **Residual distribution**: Should be approximately normal

    **Interpretation:**
    - **Good model**: Low AIC/BIC, significant parameters, white noise residuals
    - **Poor model**: High AIC/BIC, insignificant parameters, autocorrelated residuals

    ### Tips
    - Start with automatic selection to get initial orders
    - Check residual diagnostics carefully
    - If residuals show patterns, try different orders
    - Use missing values to test model robustness
    """)
