# Time Series Analysis Explorer

An interactive Streamlit application for exploring and estimating time series models.

## Features

### Phase 1 & 2: Exploration Tools (Current)

**ARMA Exploration**
- Interactive parameter adjustment for ARMA models
- Real-time visualization of time series, ACF, PACF, and distributions
- Stationarity and invertibility checks
- Preset configurations for common patterns
- Shock/impulse response analysis (±50σ shocks)
- Code generation for reproducibility
- Educational content with theory and parameter interpretation

**Univariate Explorer** (NEW!)
- Build complex time series by combining components:
  - **Trend**: Linear, quadratic, exponential
  - **Seasonality**: Additive or multiplicative (quarterly, monthly, weekly, daily)
  - **ARMA**: Stochastic AR/MA processes
  - **Shocks**: Impulse response analysis
    - With ARMA: Shock propagates through dynamics
    - Without ARMA: Shock is a permanent level shift
- Toggle components on/off to understand interactions
- Decomposition visualization showing individual components
- Real-time ACF/PACF analysis

### Coming Soon
- Multivariate models (VAR, SVAR, DFM, State Space)
- Estimation and model fitting
- Comprehensive diagnostics and forecasting

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
cd time_series_app
```

2. Create and activate a virtual environment (recommended):

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

### Local Development

Run the Streamlit app locally:

```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Navigation

- **Home Page**: Overview and introduction
- **ARMA Exploration**: Interactive ARMA model exploration

## Usage Guide

### ARMA Exploration

1. **Quick Start**: Select a preset configuration from the dropdown to explore common ARMA patterns
2. **Custom Configuration**:
   - Adjust AR order (p) and MA order (q) using sliders
   - Set AR coefficients (φ) and MA coefficients (θ)
   - Configure simulation settings (sample size, noise variance, random seed)
3. **Generate**: Click "Generate New Series" to create a new realization with the same parameters
4. **Analyze**:
   - View the time series plot
   - Examine ACF and PACF for correlation structure
   - Check the distribution of values
5. **Learn**: Expand the theory section to understand ARMA concepts
6. **Code**: Toggle "Show Python Code" to see reproducible code

### Tips

- Start with presets to understand common patterns
- Watch how ACF decays gradually for AR processes
- PACF cuts off at lag p for AR(p) processes
- ACF cuts off at lag q for MA(q) processes
- For ARMA processes, both ACF and PACF decay gradually
- Adjust noise variance to observe volatility effects

## Project Structure

```
time_series_app/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── config/
│   └── app_config.yaml            # Application configuration
├── src/
│   ├── exploration/
│   │   └── univariate/
│   │       └── arma.py            # ARMA simulation functions
│   ├── utils/
│   │   ├── code_display.py        # Code generation utilities
│   │   └── plotting.py            # Visualization utilities
│   └── pages/
│       └── 1_ARMA_Exploration.py  # ARMA exploration page
└── tests/                          # Unit tests (coming soon)
```

## Development Principles

- **Leverage existing packages**: Uses statsmodels, scipy, and other established packages
- **Minimal custom code**: Wrapper functions only where needed
- **Educational focus**: Clear explanations and code transparency
- **Local-first**: Designed for local testing, easy Databricks migration

## Tech Stack

- **Framework**: Streamlit
- **Time Series**: statsmodels, sktime
- **Visualization**: plotly, matplotlib
- **Data**: pandas, numpy

## Roadmap

See `claude.md` for the detailed implementation plan.

**Phase 1**: ✅ Basic Skeleton + ARMA Exploration
**Phase 2**: Univariate Model Expansion (ARIMA, SARIMA, components)
**Phase 3**: Multivariate Models (VAR, SVAR, DFM, State Space)
**Phase 4**: Estimation Component
**Phase 5**: Testing & QA
**Phase 6**: Documentation
**Phase 7**: Databricks Deployment Prep

## Contributing

This project is under active development. See `claude.md` for the implementation plan.

## License

[To be determined]

## Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/)
- [statsmodels](https://www.statsmodels.org/)
- [Plotly](https://plotly.com/)