# Deploying Time Series App to Databricks

This guide walks you through deploying the Time Series Analysis App as a Databricks App.

## Prerequisites

- **Databricks Workspace** with Apps enabled (requires Databricks Runtime 13.0 or higher)
- **Workspace permissions** to create and manage apps
- **Git integration** (optional but recommended for version control)

## Overview

Databricks Apps allow you to deploy Streamlit applications directly within your Databricks workspace. The app will run on Databricks infrastructure and can access workspace resources.

---

## Step 1: Prepare Your Repository

### 1.1 Verify File Structure

Ensure your repository has this structure:

```
time_series_app/
├── app.py                          # Main Streamlit entry point
├── requirements.txt                # Python dependencies
├── config/
│   └── app_config.yaml
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
    │   └── multivariate/
    └── utils/
```

### 1.2 Review requirements.txt

Your `requirements.txt` should include all dependencies:

```txt
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.17.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
scipy>=1.11.0
pyyaml>=6.0
```

**Note**: Databricks pre-installs many packages. You only need to specify packages not in the default runtime.

---

## Step 2: Connect Git Repository to Databricks

### Option A: Using Git Integration (Recommended)

1. **Navigate to Repos** in your Databricks workspace
2. Click **"Add Repo"**
3. Enter your Git repository URL (GitHub, GitLab, Bitbucket, Azure DevOps)
4. Select branch (e.g., `main` or `master`)
5. Click **"Create Repo"**

This creates a `/Repos/<username>/time_series_app` directory in your workspace.

### Option B: Manual Upload

1. **Navigate to Workspace** in Databricks
2. Create a new folder: `/Workspace/Users/<your-email>/time_series_app`
3. Upload files manually:
   - Right-click folder → **Import**
   - Drag and drop all files
   - Maintain directory structure

---

## Step 3: Create Databricks App

### 3.1 Navigate to Apps

1. In Databricks workspace, click **"Apps"** in the left sidebar
2. Click **"Create App"**

### 3.2 Configure App

**App Configuration:**

| Field | Value | Notes |
|-------|-------|-------|
| **App Name** | `Time Series Analysis` | Display name in workspace |
| **Source Type** | `Python file` | We're using Streamlit |
| **Source Path** | `/Repos/<username>/time_series_app/app.py` | Path to main app file |
| **Python File Type** | `Streamlit` | Framework selection |
| **Compute** | Select or create compute | See compute configuration below |

### 3.3 Compute Configuration

**Recommended Settings:**

- **Cluster Mode**: `Single Node` (sufficient for this app)
- **Databricks Runtime**: `13.3 LTS` or higher
- **Node Type**:
  - Development: `Standard_DS3_v2` (4 cores, 14GB RAM)
  - Production: `Standard_DS4_v2` (8 cores, 28GB RAM)
- **Autotermination**: `60 minutes` (to save costs)

**Why Single Node?**
- This app doesn't require distributed computing
- Reduces costs
- Faster startup time

### 3.4 Environment Variables (Optional)

If you need to configure the app:

```yaml
STREAMLIT_SERVER_PORT: 8501
STREAMLIT_SERVER_HEADLESS: true
```

### 3.5 Advanced Settings

**App Permissions:**
- Set who can view/run the app
- Options: `All users`, `Specific users`, `Admins only`

**Resource Limits:**
- Memory limit: `8GB` (adjust based on data size)
- Timeout: `3600s` (1 hour)

---

## Step 4: Install Dependencies

Databricks will automatically install dependencies from `requirements.txt` when the app starts.

### Verify Installation

After app creation, check the **App Logs** for:

```
Successfully installed streamlit-1.30.0
Successfully installed plotly-5.17.0
Successfully installed statsmodels-0.14.0
...
```

### Common Issues

**Issue**: Package conflicts
- **Solution**: Pin specific versions in `requirements.txt`

**Issue**: Missing packages
- **Solution**: Add to `requirements.txt`, restart app

---

## Step 5: Launch and Access App

### 5.1 Start the App

1. In the Apps page, find your app
2. Click **"Start"**
3. Wait 1-3 minutes for initialization

**Startup Process:**
1. Provisions compute resources
2. Installs dependencies
3. Launches Streamlit server
4. Exposes via Databricks proxy

### 5.2 Access the App

Once started, you'll see an **"Open App"** button:

1. Click **"Open App"**
2. App opens in new browser tab
3. URL format: `https://<databricks-instance>/apps/<app-id>`

**Note**: App is secured by your Databricks workspace authentication.

---

## Step 6: Using the App in Databricks

### Data Persistence

**Session State**: Persists during app session
**Saved Datasets**: Stored in session state (lost on app restart)

**For Production**: Consider modifying `src/utils/data_manager.py` to:
- Save to Databricks DBFS: `/dbfs/time_series_app/datasets/`
- Use Delta Lake tables
- Connect to Unity Catalog

### Sharing with Team

1. **Set App Permissions**:
   - Navigate to App settings → Permissions
   - Add users or groups
   - Set permission level (Can View, Can Run)

2. **Share App URL**:
   - Copy app URL from browser
   - Share with team members
   - They need workspace access

---

## Step 7: Updating the App

### Git-based Updates (Recommended)

1. **Push changes** to your Git repository
2. In Databricks Repos, click **"Pull"** to sync
3. **Restart the app** to apply changes
4. Changes take effect immediately

### Manual Updates

1. Navigate to workspace folder
2. **Upload modified files**
3. **Restart app**

---

## Monitoring and Maintenance

### View Logs

**Access Logs**:
1. Navigate to Apps → Your App
2. Click **"Logs"** tab
3. View stdout/stderr

**Log Levels**:
- App startup logs
- Python print statements (from our logging)
- Error traces
- Streamlit server logs

**Search Logs**:
```
[LOAD] Selected dataset: ARMA_2_1_test
[ESTIMATE] Starting model estimation
[FORECAST] Generating 10-step forecast
```

### Performance Monitoring

**Metrics to Watch**:
- **Load time**: Should be < 5s
- **Memory usage**: Monitor for leaks
- **Active users**: Concurrent sessions

**Optimization Tips**:
- Cache expensive computations with `@st.cache_data`
- Minimize session state size
- Use lighter compute if possible

### App Maintenance

**Regular Tasks**:
- **Weekly**: Check app logs for errors
- **Monthly**: Update dependencies in `requirements.txt`
- **Quarterly**: Review compute costs and optimize

**Cost Optimization**:
- Set aggressive autotermination (30-60 min)
- Use smaller nodes for development
- Stop app when not in use

---

## Troubleshooting

### App Won't Start

**Symptom**: App status shows "Failed"

**Debugging Steps**:
1. Check **Logs** for Python errors
2. Verify `app.py` path is correct
3. Ensure `requirements.txt` has no conflicts
4. Try recreating app with fresh compute

**Common Causes**:
- Wrong file path in app configuration
- Dependency conflicts in `requirements.txt`
- Insufficient compute resources
- Python version incompatibility

### Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'exploration'`

**Solution**:
```python
# Ensure app.py has this path setup
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
```

### Slow Performance

**Symptom**: App takes >30s to load pages

**Solutions**:
- Add `@st.cache_data` decorators to expensive functions
- Increase compute node size
- Reduce default `n_samples` in explorers
- Profile code to find bottlenecks

### Data Not Persisting

**Symptom**: Saved datasets disappear on app restart

**Explanation**: Session state is ephemeral

**Solutions**:
1. **Short-term**: Document that datasets are temporary
2. **Long-term**: Modify `data_manager.py` to save to DBFS:

```python
# In save_dataset()
import pickle
dataset_path = f"/dbfs/time_series_app/datasets/{name}.pkl"
with open(dataset_path, 'wb') as f:
    pickle.dump({'data': data, 'metadata': metadata}, f)
```

### Permission Errors

**Symptom**: "You don't have permission to access this app"

**Solution**:
1. Go to App settings → Permissions
2. Add users or groups
3. Grant "Can Run" permission

---

## Production Considerations

### 1. Data Storage

**Current**: Session state (temporary)

**Production Options**:

**Option A: DBFS**
```python
# Save to Databricks File System
dataset_path = f"/dbfs/FileStore/timeseries/datasets/{name}.parquet"
data.to_parquet(dataset_path)
```

**Option B: Delta Lake**
```python
# Save to Delta table
from delta import DeltaTable
data.write.format("delta").mode("overwrite").saveAsTable(f"timeseries.{name}")
```

**Option C: Unity Catalog**
```python
# Save to Unity Catalog
data.write.mode("overwrite").saveAsTable(f"catalog.schema.{name}")
```

### 2. User Management

**Integrate with Databricks Auth**:
- Users automatically authenticated via workspace
- Leverage workspace groups for permissions
- Audit trail via Databricks logs

### 3. Scalability

**For Large Datasets**:
- Use Spark DataFrames instead of Pandas
- Leverage Databricks cluster compute
- Implement lazy loading

**For Many Users**:
- Use shared compute cluster
- Implement request queuing
- Monitor concurrent sessions

### 4. CI/CD Integration

**Automated Deployment**:

```yaml
# .github/workflows/deploy-databricks.yml
name: Deploy to Databricks

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy to Databricks
        uses: databricks/databricks-cli-action@v1
        with:
          databricks-token: ${{ secrets.DATABRICKS_TOKEN }}
          command: repos update --repo-id <your-repo-id> --branch main
```

---

## Security Best Practices

### 1. Access Control

- **Principle of least privilege**: Only grant necessary permissions
- **Regular audits**: Review who has access quarterly
- **Group-based access**: Use Databricks groups, not individual users

### 2. Data Protection

- **Sensitive data**: Don't hardcode credentials or API keys
- **Use Databricks Secrets**: Store sensitive configs in secret scopes
- **Audit logs**: Monitor data access patterns

### 3. Network Security

- **App runs within Databricks VPC**: Already isolated
- **Enable IP access lists**: Restrict workspace access by IP
- **Use SSO**: Enforce corporate SSO for authentication

---

## Cost Management

### Estimating Costs

**Formula**:
```
Cost = (DBU rate × DBUs per hour × Hours running) + (VM cost × Hours running)
```

**Example** (Standard_DS3_v2, 100 hours/month):
- DBU cost: ~$50-100/month
- VM cost: ~$100-150/month
- **Total**: ~$150-250/month

### Cost Optimization Tips

1. **Aggressive Autotermination**: Set to 30-60 minutes
2. **Right-size Compute**: Start small, scale only if needed
3. **Development vs Production**: Use smaller nodes for dev
4. **Monitoring**: Track costs in Databricks account console
5. **Scheduled Downtime**: Stop app outside business hours if interactive use only

---

## Next Steps

### Immediate Actions

1. ✅ Deploy app to Databricks
2. ✅ Test all exploration pages
3. ✅ Test estimation workflows
4. ✅ Share with 2-3 beta users

### Phase 2 Enhancements

1. **Persistent Storage**: Implement DBFS or Delta Lake for datasets
2. **Data Upload**: Allow users to upload CSV/Parquet files
3. **Export Reports**: Generate PDF reports with equations and charts
4. **Model Comparison**: Side-by-side model comparison tools

### Phase 3 Features

1. **Forecasting Dashboard**: Real-time forecasting with scheduled updates
2. **Alerts**: Automated anomaly detection and alerts
3. **Collaboration**: Shared workspaces for teams
4. **API Integration**: REST API for programmatic access

---

## Support and Resources

### Databricks Documentation

- [Databricks Apps Overview](https://docs.databricks.com/en/apps/index.html)
- [Streamlit on Databricks](https://docs.databricks.com/en/apps/streamlit.html)
- [App Deployment Best Practices](https://docs.databricks.com/en/apps/best-practices.html)

### Time Series App Resources

- **GitHub Repository**: (your repo URL)
- **Internal Documentation**: See `README.md` and `QUICKSTART.md`
- **Issue Tracking**: (your issue tracker)

### Getting Help

1. **Check Logs**: Always start with app logs
2. **Databricks Support**: Use support portal for platform issues
3. **Community**: Databricks Community forums
4. **Internal**: Contact your Databricks workspace admin

---

## Appendix: Alternative Deployment Options

### Option 1: Databricks Jobs (Scheduled Reports)

If you want scheduled execution instead of interactive app:

```python
# Convert to batch job
# Run estimation on schedule
# Output results to Delta table
# Email results to stakeholders
```

### Option 2: MLflow Integration

For model tracking and versioning:

```python
import mlflow
# Log models to MLflow
# Track experiments
# Serve models via MLflow
```

### Option 3: External Deployment

If Databricks Apps aren't available:

1. **Deploy to Cloud**: AWS EC2, Azure VM, GCP Compute
2. **Containerize**: Create Docker image, deploy to Kubernetes
3. **Serverless**: AWS Lambda + API Gateway (requires modification)

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-22 | Initial deployment guide |

---

**Questions?** Contact your Databricks administrator or data platform team.
