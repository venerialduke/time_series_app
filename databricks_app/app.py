"""
Time Series Analysis Explorer

A Streamlit application for interactive exploration and estimation of time series models.
"""

import streamlit as st
import yaml
from pathlib import Path

# Load configuration
config_path = Path(__file__).parent / "config" / "app_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Page configuration
st.set_page_config(
    page_title=config['app']['title'],
    layout=config['app']['layout'],
    initial_sidebar_state=config['app']['initial_sidebar_state']
)

# Main page
st.title("Time Series Analysis Explorer")

st.markdown("""
## Welcome to the Time Series Analysis Explorer

This interactive application helps you understand time series analysis through exploration and estimation.

### 📊 Exploration Pages

Interactively explore time series models by adjusting parameters and observing dynamics:

**Univariate Models:**
- **Univariate Explorer**: Build series with Trend + Seasonality + ARMA + Shocks

**Multivariate Models:**
- **VAR Explorer**: 2-3 variable Vector Autoregression with Impulse Response Functions
- **State Space Explorer**: Unobserved components (Local Level, Local Linear Trend, Structural)
- **Dynamic Factor Explorer**: 1-3 latent factors driving up to 10 observed series

### 🎯 Estimation Pages

Fit time series models to data with comprehensive diagnostics:

- **Univariate Estimation**: ARMA/ARIMA with stationarity testing, transformations, and forecasting
- **Multivariate Estimation**: VAR models with lag selection and diagnostics

---

### Getting Started

1. **Explore**: Start with exploration pages to understand model behavior
2. **Generate**: Create synthetic datasets with known parameters
3. **Save**: Save generated datasets for later analysis
4. **Estimate**: Load saved datasets and fit models
5. **Diagnose**: Evaluate model fit with residual plots, ACF/PACF, and forecasts

### Features

- Interactive parameter controls with real-time updates
- Presentation-ready charts with LaTeX equations
- Save/load datasets across sessions
- Stationarity testing (ADF, KPSS)
- Data transformations (differencing, detrending)
- Comprehensive diagnostics and forecasting

---

### Built With

- **Streamlit** - Interactive web interface
- **statsmodels** - Time series modeling and estimation
- **plotly** - Interactive visualizations
- **scipy** - Signal processing
- **scikit-learn** - Factor analysis

---

**👈 Select a page from the sidebar to get started!**
""")

st.sidebar.success("Select a page above to get started.")
