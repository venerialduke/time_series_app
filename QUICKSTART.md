# Quick Start Guide

## Setup and Run Instructions

### Step 1: Create Virtual Environment (Recommended)

It's best practice to use a virtual environment to isolate project dependencies:

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

You should see `(venv)` in your terminal prompt when the virtual environment is activated.

### Step 2: Install Dependencies

With the virtual environment activated, install all required Python packages:

```bash
pip install -r requirements.txt
```

This will install:
- streamlit
- pandas
- numpy
- matplotlib
- plotly
- statsmodels
- scikit-learn
- sktime
- pyyaml
- scipy

### Step 3: Run the Application

Start the Streamlit app (make sure your virtual environment is activated):

```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

### Step 4: Explore Time Series Models

The app has two exploration tools:

#### Option A: ARMA Exploration (Focused)
1. In the sidebar, click on **"ARMA Exploration"** page
2. Try the preset configurations:
   - Select "AR(1) - Positive" to see a smooth, persistent series
   - Select "AR(1) - Negative" to see an oscillating series
   - Select "ARMA(1,1)" to see combined effects

3. Customize parameters:
   - Adjust AR order (p) and MA order (q)
   - Set coefficient values using the sliders
   - Change sample size and noise variance
   - Click "Generate New Series" for a new realization

4. Analyze the results:
   - View the time series plot
   - Examine ACF (Autocorrelation Function)
   - Examine PACF (Partial Autocorrelation Function)
   - Check the distribution

5. Explore impulse responses with shocks:
   - Check "Add Shock to Series" in the sidebar
   - Set the shock time (observation number)
   - Set the shock magnitude (in standard deviations)
   - Observe how different models respond:
     - AR processes: Persistent, gradual decay
     - MA processes: Temporary, quick dissipation
     - ARMA processes: Combined behavior

6. Learn more:
   - Expand the "ARMA Theory & Concepts" section to read about ACF, PACF, and model theory
   - Toggle "Show Python Code" to see reproducible code

#### Option B: Univariate Explorer (Comprehensive)
1. In the sidebar, click on **"Univariate Explorer"** page
2. Build complex time series by combining components:
   - **Enable Trend**: Linear, quadratic, or exponential growth/decline
   - **Enable Cycle**: Regular oscillations (adjust period and amplitude)
   - **Enable Seasonality**: Repeating patterns (monthly, quarterly, etc.)
   - **Enable ARMA**: Add stochastic AR/MA processes
   - **Add Shock**: Apply impulse at specific time points

3. Customize each component:
   - Each component has its own controls
   - Toggle components on/off to isolate effects
   - Combine multiple components to see interactions

4. Analyze the complex series:
   - View combined time series
   - Show individual component breakdowns
   - Examine ACF/PACF of the combined series
   - Observe how components interact

5. Learn about decomposition:
   - See how trend, cycle, and seasonality combine
   - Understand additive vs multiplicative seasonality
   - Observe how ARMA adds randomness to deterministic patterns

### Step 5: Run Tests (Optional)

To verify the installation and basic functionality (with venv activated):

```bash
cd tests
python test_arma.py
```

You should see all tests pass with checkmarks.

## Troubleshooting

### Dependencies Not Found
If you get `ModuleNotFoundError`, make sure:
1. Your virtual environment is activated (you should see `(venv)` in your prompt)
2. You installed dependencies with:
```bash
pip install -r requirements.txt
```

### Virtual Environment Not Activated
If commands aren't working, activate your virtual environment:
- **Windows**: `venv\Scripts\activate`
- **macOS/Linux**: `source venv/bin/activate`

To deactivate when done:
```bash
deactivate
```

### Port Already in Use
If port 8501 is already in use, Streamlit will automatically try the next available port (8502, 8503, etc.)

### Browser Doesn't Open Automatically
If the browser doesn't open automatically, manually navigate to the URL shown in the terminal (usually `http://localhost:8501`)

## What's Included

### Phase 1 & 2 Complete: ARMA Exploration
- Interactive ARMA model simulation
- Parameter controls (AR order, MA order, coefficients)
- Visualization (time series, ACF, PACF, distribution)
- Stationarity and invertibility checks
- 7 preset configurations
- Python code generation
- Educational content

### Project Files Created
```
time_series_app/
├── app.py                              # Main app entry point
├── requirements.txt                    # Dependencies
├── config/app_config.yaml              # Configuration
├── src/
│   ├── exploration/univariate/arma.py  # ARMA simulation
│   ├── utils/code_display.py           # Code generation
│   ├── utils/plotting.py               # Visualization
│   └── pages/1_ARMA_Exploration.py     # ARMA page
└── tests/test_arma.py                  # Unit tests
```

## Next Steps

After exploring ARMA models, future phases will add:
- ARIMA and SARIMA models (with trends and seasonality)
- Multivariate models (VAR, SVAR, Dynamic Factor, State Space)
- Estimation component (fit models to your data)
- Comprehensive diagnostics and forecasting

See `claude.md` for the full implementation plan.

## Tips for Exploration

1. **Start Simple**: Begin with AR(1) or MA(1) presets
2. **Observe Patterns**:
   - AR processes: ACF decays, PACF cuts off at lag p
   - MA processes: ACF cuts off at lag q, PACF decays
   - ARMA processes: Both ACF and PACF decay gradually
3. **Experiment**: Change one parameter at a time to see its effect
4. **Check Stationarity**: The app will warn you if parameters lead to non-stationary processes
5. **Generate Code**: Use the code display to recreate results in your own scripts

## Getting Help

- Expand the "ARMA Theory & Concepts" section in the app
- Check the "Parameter Interpretation" tab
- Read tooltips (hover over ? icons)
- Review the example presets

Enjoy exploring time series models!
