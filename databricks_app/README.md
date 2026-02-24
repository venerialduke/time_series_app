# Databricks App - Deployment Ready

This folder contains the Time Series Analysis App structured specifically for Databricks deployment.

## Structure

```
databricks_app/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── config/
│   └── app_config.yaml            # App configuration
├── pages/
│   ├── 1_Univariate_Explorer.py
│   ├── 2_VAR_Explorer.py
│   ├── 3_State_Space_Explorer.py
│   ├── 4_Dynamic_Factor_Explorer.py
│   ├── 5_Univariate_Estimation.py
│   └── 6_Multivariate_Estimation.py
└── src/
    ├── exploration/
    │   ├── univariate/
    │   │   ├── arma.py
    │   │   └── components.py
    │   └── multivariate/
    │       ├── var.py
    │       ├── state_space.py
    │       └── dfm.py
    ├── estimation/
    └── utils/
        ├── data_manager.py
        ├── code_display.py
        └── plotting.py
```

## Deployment to Databricks

### Option 1: Git Integration (Recommended)

1. Push this folder's contents to your Git repository (GitHub, GitLab, etc.)
2. In Databricks workspace, navigate to **Repos**
3. Click **"Add Repo"**
4. Enter your Git repository URL
5. Select branch and create repo
6. This creates `/Repos/<username>/time_series_app` in your workspace

### Option 2: Manual Upload

1. In Databricks workspace, navigate to **Workspace**
2. Create folder: `/Workspace/Users/<your-email>/time_series_app`
3. Upload all files from this `databricks_app/` folder
4. Maintain the directory structure exactly as shown above

### Creating the Databricks App

1. Navigate to **Apps** in Databricks workspace
2. Click **"Create App"**
3. Configure:
   - **App Name**: `Time Series Analysis`
   - **Source Type**: `Python file`
   - **Source Path**: `/Repos/<username>/time_series_app/app.py` (or workspace path)
   - **Python File Type**: `Streamlit`
   - **Compute**: Select or create compute (see recommendations below)

### Compute Recommendations

**Development:**
- Cluster Mode: Single Node
- Runtime: 13.3 LTS or higher
- Node Type: Standard_DS3_v2 (4 cores, 14GB RAM)
- Autotermination: 60 minutes

**Production:**
- Cluster Mode: Single Node
- Runtime: 13.3 LTS or higher
- Node Type: Standard_DS4_v2 (8 cores, 28GB RAM)
- Autotermination: 60 minutes

## What's Included

### Exploration Pages
- **Univariate Explorer**: Trend + Seasonality + ARMA models with shocks
- **VAR Explorer**: 2-3 variable Vector Autoregression with IRFs
- **State Space Explorer**: Local Level, Local Linear Trend, Structural models
- **Dynamic Factor Explorer**: 1-3 factors driving up to 10 series

### Estimation Pages
- **Univariate Estimation**: ARMA/ARIMA with stationarity testing, transformations, diagnostics
- **Multivariate Estimation**: VAR models with lag selection and diagnostics

### Features
- Interactive parameter adjustment
- Presentation-ready charts with LaTeX equations
- Data persistence via session state
- Save/load datasets
- Comprehensive diagnostics and forecasting

## Dependencies

All dependencies are listed in `requirements.txt`:
- streamlit >= 1.30.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- plotly >= 5.17.0
- statsmodels >= 0.14.0
- scikit-learn >= 1.3.0
- scipy >= 1.11.0
- pyyaml >= 6.0

Databricks will automatically install these when the app starts.

## Testing Locally (Optional)

Before deploying to Databricks, you can test locally:

```bash
cd databricks_app
streamlit run app.py
```

## Full Deployment Guide

See `../DATABRICKS_DEPLOYMENT.md` in the parent directory for comprehensive deployment instructions, troubleshooting, and production considerations.

## Support

For issues or questions:
1. Check app logs in Databricks Apps UI
2. Review `../DATABRICKS_DEPLOYMENT.md` troubleshooting section
3. Contact your Databricks workspace administrator

---

**Ready for Databricks Deployment** ✅
