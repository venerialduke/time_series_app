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

This interactive application helps you understand time series analysis through:

### 1. Exploration
Interactively explore different time series models by adjusting parameters and observing how the series evolves.

**Available Explorers:**
- **ARMA Exploration**: Focus on pure ARMA processes with AR and MA parameters
- **Univariate Explorer**: Build complex series by combining Trend + Cycle + Seasonality + ARMA + Shocks

### 2. Estimation (Coming Soon)
Fit time series models to your data and evaluate their performance with comprehensive diagnostics.

---

### Getting Started

Use the sidebar to navigate between different sections of the app.

**Start with Exploration** to understand how different models behave with various parameter settings.

### Features

- Interactive parameter controls
- Real-time visualization
- Code generation (see the Python code behind each model)
- Educational explanations

---

### About

This app is built on:
- **Streamlit** for the interactive interface
- **statsmodels** for time series modeling
- **plotly** for interactive visualizations

The goal is to make time series analysis accessible and understandable through hands-on exploration.
""")

st.sidebar.success("Select a page above to get started.")
