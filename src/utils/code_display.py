"""
Utilities for displaying Python code snippets in the Streamlit app.
"""

import streamlit as st


def format_arma_code(ar_order: int, ma_order: int, ar_params: list, ma_params: list,
                     n_samples: int, sigma: float, seed: int = None) -> str:
    """
    Generate Python code for ARMA simulation.

    Parameters
    ----------
    ar_order : int
        AR order (p)
    ma_order : int
        MA order (q)
    ar_params : list
        AR coefficients
    ma_params : list
        MA coefficients
    n_samples : int
        Number of samples to generate
    sigma : float
        Standard deviation of noise
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    str
        Formatted Python code
    """
    ar_str = ", ".join([f"{p:.3f}" for p in ar_params]) if ar_params else ""
    ma_str = ", ".join([f"{q:.3f}" for q in ma_params]) if ma_params else ""

    code = f"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set random seed for reproducibility
np.random.seed({seed if seed is not None else 42})

# Define ARMA parameters
ar_params = np.array([{ar_str}])  # AR coefficients (excluding lag 0)
ma_params = np.array([{ma_str}])  # MA coefficients (excluding lag 0)

# Generate ARMA process
# Note: statsmodels uses the convention where the AR polynomial is
# (1 - ar[0]*L - ar[1]*L^2 - ...) and MA is (1 + ma[0]*L + ma[1]*L^2 + ...)
y = arma_generate_sample(
    ar=ar_params,
    ma=ma_params,
    nsample={n_samples},
    sigma={sigma},
    burnin=500  # Burn-in period to reach stationarity
)

# Create time series
ts = pd.Series(y, index=pd.date_range(start='2020-01-01', periods={n_samples}, freq='D'))

# Plot the time series
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Time series plot
axes[0].plot(ts)
axes[0].set_title(f'ARMA({ar_order}, {ma_order}) Process')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3)

# ACF plot
plot_acf(ts, lags=40, ax=axes[1])
axes[1].set_title('Autocorrelation Function (ACF)')

# PACF plot
plot_pacf(ts, lags=40, ax=axes[2])
axes[2].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.show()

# Summary statistics
print(f"Mean: {{ts.mean():.4f}}")
print(f"Std Dev: {{ts.std():.4f}}")
print(f"Min: {{ts.min():.4f}}")
print(f"Max: {{ts.max():.4f}}")
"""

    return code


def display_code_block(code: str, language: str = "python"):
    """
    Display a code block with copy button.

    Parameters
    ----------
    code : str
        Code to display
    language : str
        Programming language for syntax highlighting
    """
    st.code(code, language=language)

    # Note: Streamlit automatically provides a copy button in code blocks
    st.caption("Click the copy icon in the top-right corner of the code block to copy.")


def show_arma_theory():
    """
    Display theoretical explanation of ARMA models.
    """
    st.markdown("""
    ### ARMA Model Theory

    An **ARMA(p, q)** model combines autoregressive (AR) and moving average (MA) components:

    $$y_t = c + \\phi_1 y_{t-1} + \\phi_2 y_{t-2} + ... + \\phi_p y_{t-p} + \\varepsilon_t + \\theta_1 \\varepsilon_{t-1} + ... + \\theta_q \\varepsilon_{t-q}$$

    Where:
    - $y_t$ is the value at time $t$
    - $\\phi_i$ are the AR coefficients (p lags)
    - $\\theta_j$ are the MA coefficients (q lags)
    - $\\varepsilon_t$ is white noise with variance $\\sigma^2$
    - $c$ is a constant term

    #### AR Component (AutoRegressive)
    The AR part models the current value as a linear combination of past values.
    - Higher AR coefficients mean stronger dependence on past values
    - AR processes create smooth, persistent patterns

    #### MA Component (Moving Average)
    The MA part models the current value as a linear combination of past error terms.
    - MA coefficients determine how past shocks affect current values
    - MA processes create short-term dependencies

    #### Stationarity
    For an ARMA process to be stationary, the AR polynomial roots must lie outside the unit circle.

    ---

    ### Autocorrelation Function (ACF)

    The **ACF** measures the correlation between the time series and its lagged values.

    $$ACF(k) = \\frac{Cov(y_t, y_{t-k})}{Var(y_t)}$$

    **What it tells you:**
    - Measures linear dependence between observations separated by k time periods
    - Values range from -1 to 1
    - ACF at lag 0 is always 1 (perfect correlation with itself)

    **Interpretation patterns:**
    - **AR processes**: ACF decays gradually (exponentially or sinusoidally)
    - **MA(q) processes**: ACF cuts off sharply after lag q (zero after lag q)
    - **White noise**: ACF is approximately zero at all lags except 0

    **Confidence bands** (dashed red lines): Values outside these bands are statistically significant at the 95% level.

    ---

    ### Partial Autocorrelation Function (PACF)

    The **PACF** measures the correlation between the time series and its lagged values **after removing** the effect of intermediate lags.

    **What it tells you:**
    - Shows the "direct" correlation at each lag, controlling for shorter lags
    - Answers: "What's the correlation at lag k after accounting for lags 1, 2, ..., k-1?"
    - Helps identify the order of AR processes

    **Interpretation patterns:**
    - **AR(p) processes**: PACF cuts off sharply after lag p (zero after lag p)
    - **MA processes**: PACF decays gradually
    - **White noise**: PACF is approximately zero at all lags

    ---

    ### Using ACF and PACF Together

    | Process | ACF Pattern | PACF Pattern |
    |---------|-------------|--------------|
    | **AR(p)** | Gradual decay | Cuts off after lag p |
    | **MA(q)** | Cuts off after lag q | Gradual decay |
    | **ARMA(p,q)** | Gradual decay | Gradual decay |
    | **White Noise** | ~0 for all lags | ~0 for all lags |

    **Model identification:**
    - If PACF cuts off at lag p and ACF decays: try AR(p)
    - If ACF cuts off at lag q and PACF decays: try MA(q)
    - If both decay gradually: try ARMA(p,q)
    """)


def show_parameter_interpretation():
    """
    Display interpretation guide for ARMA parameters.
    """
    st.markdown("""
    ### Parameter Interpretation

    **AR Order (p)**: Number of lagged observations
    - p=0: No AR component (pure MA or white noise)
    - p=1: Current value depends on previous value
    - p>1: Current value depends on multiple past values

    **MA Order (q)**: Number of lagged forecast errors
    - q=0: No MA component (pure AR)
    - q=1: Current value affected by previous error
    - q>1: Current value affected by multiple past errors

    **AR Coefficients ($\\phi$)**:
    - Positive: Previous values in same direction influence current value
    - Negative: Previous values in opposite direction influence current value
    - Magnitude close to 1: Strong persistence
    - Magnitude close to 0: Weak dependence

    **MA Coefficients ($\\theta$)**:
    - Positive: Previous shocks positively affect current value
    - Negative: Previous shocks negatively affect current value
    - Larger magnitude: Stronger shock influence

    **Noise Variance ($\\sigma^2$)**:
    - Controls the volatility of the series
    - Larger values create more erratic series
    """)
