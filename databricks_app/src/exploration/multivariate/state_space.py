"""
State Space Models - Unobserved Components.

Canonical state space models using statsmodels.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from statsmodels.tsa.statespace.structural import UnobservedComponents


def simulate_local_level(
    n_samples: int = 500,
    level_var: float = 1.0,
    obs_var: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate a local level model:
    y_t = μ_t + ε_t
    μ_t = μ_{t-1} + η_t

    Parameters
    ----------
    n_samples : int
        Number of observations
    level_var : float
        Variance of level innovations (η_t)
    obs_var : float
        Variance of observation noise (ε_t)
    seed : int, optional
        Random seed

    Returns
    -------
    tuple
        (observed_series, level_series)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate innovations
    level_innov = np.random.normal(0, np.sqrt(level_var), n_samples)
    obs_noise = np.random.normal(0, np.sqrt(obs_var), n_samples)

    # Generate level (random walk)
    level = np.cumsum(level_innov)

    # Generate observations
    observed = level + obs_noise

    # Create series
    index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    obs_series = pd.Series(observed, index=index, name='Observed')
    level_series = pd.Series(level, index=index, name='Level')

    return obs_series, level_series


def simulate_local_linear_trend(
    n_samples: int = 500,
    level_var: float = 1.0,
    trend_var: float = 0.1,
    obs_var: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Simulate a local linear trend model:
    y_t = μ_t + ε_t
    μ_t = μ_{t-1} + β_{t-1} + η_t
    β_t = β_{t-1} + ζ_t

    Parameters
    ----------
    n_samples : int
        Number of observations
    level_var : float
        Variance of level innovations
    trend_var : float
        Variance of trend innovations
    obs_var : float
        Variance of observation noise
    seed : int, optional
        Random seed

    Returns
    -------
    tuple
        (observed_series, level_series, trend_series)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate innovations
    level_innov = np.random.normal(0, np.sqrt(level_var), n_samples)
    trend_innov = np.random.normal(0, np.sqrt(trend_var), n_samples)
    obs_noise = np.random.normal(0, np.sqrt(obs_var), n_samples)

    # Initialize
    level = np.zeros(n_samples)
    trend = np.zeros(n_samples)

    # Generate process
    for t in range(1, n_samples):
        trend[t] = trend[t-1] + trend_innov[t]
        level[t] = level[t-1] + trend[t-1] + level_innov[t]

    # Generate observations
    observed = level + obs_noise

    # Create series
    index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    obs_series = pd.Series(observed, index=index, name='Observed')
    level_series = pd.Series(level, index=index, name='Level')
    trend_series = pd.Series(trend, index=index, name='Trend')

    return obs_series, level_series, trend_series


def simulate_structural_model(
    n_samples: int = 500,
    level_var: float = 1.0,
    trend_var: float = 0.1,
    seasonal_var: float = 0.5,
    obs_var: float = 1.0,
    seasonal_period: int = 12,
    seed: Optional[int] = None
) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """
    Simulate a basic structural model with level, trend, and seasonal components.

    y_t = μ_t + γ_t + ε_t
    μ_t = μ_{t-1} + β_{t-1} + η_t
    β_t = β_{t-1} + ζ_t
    γ_t + γ_{t-1} + ... + γ_{t-s+1} = ω_t

    Parameters
    ----------
    n_samples : int
        Number of observations
    level_var : float
        Variance of level innovations
    trend_var : float
        Variance of trend innovations
    seasonal_var : float
        Variance of seasonal innovations
    obs_var : float
        Variance of observation noise
    seasonal_period : int
        Seasonal period
    seed : int, optional
        Random seed

    Returns
    -------
    tuple
        (observed_series, components_dict)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate innovations
    level_innov = np.random.normal(0, np.sqrt(level_var), n_samples)
    trend_innov = np.random.normal(0, np.sqrt(trend_var), n_samples)
    seasonal_innov = np.random.normal(0, np.sqrt(seasonal_var), n_samples)
    obs_noise = np.random.normal(0, np.sqrt(obs_var), n_samples)

    # Initialize
    level = np.zeros(n_samples)
    trend = np.zeros(n_samples)
    seasonal = np.zeros(n_samples)

    # Generate level and trend
    for t in range(1, n_samples):
        trend[t] = trend[t-1] + trend_innov[t]
        level[t] = level[t-1] + trend[t-1] + level_innov[t]

    # Generate seasonal component (sum to zero constraint)
    for t in range(seasonal_period, n_samples):
        seasonal[t] = -sum(seasonal[t-seasonal_period+1:t]) + seasonal_innov[t]

    # Generate observations
    observed = level + seasonal + obs_noise

    # Create series
    index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    obs_series = pd.Series(observed, index=index, name='Observed')

    components = {
        'level': pd.Series(level, index=index, name='Level'),
        'trend': pd.Series(trend, index=index, name='Trend'),
        'seasonal': pd.Series(seasonal, index=index, name='Seasonal')
    }

    return obs_series, components


def fit_unobserved_components(
    data: pd.Series,
    model_type: str = 'local_level',
    seasonal_period: Optional[int] = None
) -> Tuple[object, pd.DataFrame]:
    """
    Fit an unobserved components model using statsmodels.

    Parameters
    ----------
    data : pd.Series
        Observed time series
    model_type : str
        'local_level', 'local_linear_trend', or 'structural'
    seasonal_period : int, optional
        Seasonal period (required for 'structural')

    Returns
    -------
    tuple
        (fitted_model, smoothed_states DataFrame)
    """
    # Build model specification
    if model_type == 'local_level':
        model = UnobservedComponents(
            data,
            level='local level'
        )
    elif model_type == 'local_linear_trend':
        model = UnobservedComponents(
            data,
            level='local linear trend'
        )
    elif model_type == 'structural':
        if seasonal_period is None:
            raise ValueError("seasonal_period required for structural model")
        model = UnobservedComponents(
            data,
            level='local linear trend',
            seasonal=seasonal_period,
            stochastic_seasonal=True
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Fit model
    result = model.fit(disp=False)

    # Extract smoothed states
    smoothed_states = pd.DataFrame({
        'level': result.level.smoothed,
    })

    if model_type in ['local_linear_trend', 'structural']:
        smoothed_states['trend'] = result.trend.smoothed

    if model_type == 'structural' and hasattr(result, 'seasonal'):
        smoothed_states['seasonal'] = result.seasonal.smoothed

    return result, smoothed_states
