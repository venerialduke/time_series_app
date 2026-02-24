"""
Univariate time series components: Trend, Cycle, Seasonality.

This module provides functions to generate and combine various components
of univariate time series models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict
from scipy import signal


def generate_trend(
    n_samples: int,
    trend_type: str = 'none',
    slope: float = 0.0,
    intercept: float = 0.0,
    quad_coef: float = 0.0
) -> np.ndarray:
    """
    Generate a deterministic trend component.

    Parameters
    ----------
    n_samples : int
        Number of samples
    trend_type : str
        Type of trend: 'none', 'linear', 'quadratic', 'exponential'
    slope : float
        Linear trend coefficient
    intercept : float
        Starting value (intercept)
    quad_coef : float
        Quadratic coefficient (for quadratic trend)

    Returns
    -------
    np.ndarray
        Trend component

    Examples
    --------
    >>> trend = generate_trend(100, trend_type='linear', slope=0.1, intercept=5.0)
    """
    t = np.arange(n_samples)

    if trend_type == 'none':
        return np.zeros(n_samples)
    elif trend_type == 'linear':
        return intercept + slope * t
    elif trend_type == 'quadratic':
        return intercept + slope * t + quad_coef * (t ** 2)
    elif trend_type == 'exponential':
        # Exponential trend: exp(slope * t)
        return intercept * np.exp(slope * t / 100)  # Scaled for stability
    else:
        raise ValueError(f"Unknown trend type: {trend_type}")


def generate_cycle(
    n_samples: int,
    period: float = 50.0,
    amplitude: float = 1.0,
    phase: float = 0.0
) -> np.ndarray:
    """
    Generate a cyclical component using sine wave.

    Parameters
    ----------
    n_samples : int
        Number of samples
    period : float
        Period of the cycle (in time units)
    amplitude : float
        Amplitude of the cycle
    phase : float
        Phase shift in radians

    Returns
    -------
    np.ndarray
        Cyclical component

    Examples
    --------
    >>> cycle = generate_cycle(100, period=20, amplitude=2.0)
    """
    if period <= 0:
        return np.zeros(n_samples)

    t = np.arange(n_samples)
    frequency = 2 * np.pi / period
    return amplitude * np.sin(frequency * t + phase)


def generate_seasonality(
    n_samples: int,
    seasonal_period: int = 12,
    seasonal_amplitude: float = 1.0,
    seasonal_type: str = 'additive'
) -> np.ndarray:
    """
    Generate a seasonal component.

    Parameters
    ----------
    n_samples : int
        Number of samples
    seasonal_period : int
        Number of observations per season (e.g., 12 for monthly, 4 for quarterly)
    seasonal_amplitude : float
        Amplitude of seasonal pattern
    seasonal_type : str
        'additive' or 'multiplicative' (returns as multiplier around 1)

    Returns
    -------
    np.ndarray
        Seasonal component

    Examples
    --------
    >>> season = generate_seasonality(100, seasonal_period=12, seasonal_amplitude=2.0)
    """
    if seasonal_period <= 0:
        return np.zeros(n_samples)

    # Create a simple seasonal pattern using multiple harmonics
    t = np.arange(n_samples)
    season = np.zeros(n_samples)

    # Use first 3 harmonics for richer seasonal pattern
    for k in range(1, 4):
        freq = 2 * np.pi * k / seasonal_period
        # Random phase and weight for each harmonic
        phase = np.random.uniform(0, 2 * np.pi)
        weight = 1.0 / k  # Diminishing weight for higher harmonics
        season += weight * np.sin(freq * t + phase)

    # Normalize and scale
    if np.std(season) > 0:
        season = season / np.std(season)
    season = seasonal_amplitude * season

    if seasonal_type == 'multiplicative':
        # Return as multiplier (centered around 1)
        return 1.0 + season / 10.0  # Scale down to avoid extreme multipliers
    else:
        return season


def generate_arma_component(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    n_samples: int,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    burnin: int = 500
) -> np.ndarray:
    """
    Generate ARMA component (wrapper around statsmodels).

    Parameters
    ----------
    ar_params : np.ndarray
        AR coefficients
    ma_params : np.ndarray
        MA coefficients
    n_samples : int
        Number of samples
    sigma : float
        Standard deviation of noise
    seed : int, optional
        Random seed
    burnin : int
        Burn-in period

    Returns
    -------
    np.ndarray
        ARMA component values
    """
    from statsmodels.tsa.arima_process import arma_generate_sample

    if seed is not None:
        np.random.seed(seed)

    # Prepare parameters
    if len(ar_params) > 0:
        ar_full = np.concatenate(([1], -ar_params))
    else:
        ar_full = np.array([1])

    if len(ma_params) > 0:
        ma_full = np.concatenate(([1], ma_params))
    else:
        ma_full = np.array([1])

    # Generate ARMA
    y = arma_generate_sample(
        ar=ar_full,
        ma=ma_full,
        nsample=n_samples,
        scale=sigma,
        burnin=burnin
    )

    return y


def combine_components(
    trend: np.ndarray,
    seasonality: np.ndarray,
    arma: np.ndarray,
    seasonal_type: str = 'additive'
) -> np.ndarray:
    """
    Combine time series components.

    Parameters
    ----------
    trend : np.ndarray
        Trend component
    seasonality : np.ndarray
        Seasonal component
    arma : np.ndarray
        ARMA (stochastic) component
    seasonal_type : str
        'additive' or 'multiplicative'

    Returns
    -------
    np.ndarray
        Combined time series

    Notes
    -----
    For additive: y = trend + seasonality + arma
    For multiplicative: y = trend * seasonality + arma
    """
    if seasonal_type == 'multiplicative':
        # Multiplicative seasonality
        return trend * seasonality + arma
    else:
        # Additive (default)
        return trend + seasonality + arma


def apply_shock_to_components(
    trend: np.ndarray,
    seasonality: np.ndarray,
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    shock_time: int,
    shock_magnitude: float,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    seasonal_type: str = 'additive'
) -> tuple:
    """
    Generate components with a shock applied.

    Shock behavior depends on whether ARMA is present:
    - With ARMA: Shock propagates through AR/MA dynamics
    - Without ARMA: Shock is a permanent level shift from shock_time onward

    Parameters
    ----------
    trend : np.ndarray
        Pre-generated trend
    seasonality : np.ndarray
        Pre-generated seasonality
    ar_params : np.ndarray
        AR coefficients (empty array if no AR)
    ma_params : np.ndarray
        MA coefficients (empty array if no MA)
    shock_time : int
        Time index for shock
    shock_magnitude : float
        Magnitude of shock in standard deviations
    sigma : float
        Standard deviation of noise (only used if ARMA present)
    seed : int, optional
        Random seed
    seasonal_type : str
        'additive' or 'multiplicative'

    Returns
    -------
    tuple
        (combined_series, shock_info)
    """
    n_samples = len(trend)
    has_arma = len(ar_params) > 0 or len(ma_params) > 0

    if seed is not None:
        np.random.seed(seed)

    if has_arma:
        # With ARMA: shock propagates through dynamics
        burnin = 500

        # Prepare AR/MA parameters
        if len(ar_params) > 0:
            ar_full = np.concatenate(([1], -ar_params))
        else:
            ar_full = np.array([1])

        if len(ma_params) > 0:
            ma_full = np.concatenate(([1], ma_params))
        else:
            ma_full = np.array([1])

        # Generate noise with shock
        noise = np.random.normal(0, sigma, n_samples + burnin)
        noise[burnin + shock_time] += shock_magnitude * sigma

        # Apply ARMA filter
        arma = signal.lfilter(ma_full, ar_full, noise)
        arma = arma[burnin:]
    else:
        # Without ARMA: shock is a permanent level shift
        arma = np.zeros(n_samples)
        # Apply shock as level shift from shock_time onward
        arma[shock_time:] = shock_magnitude * sigma

    # Combine components
    combined = combine_components(trend, seasonality, arma, seasonal_type)

    shock_info = {
        'time': shock_time,
        'magnitude': shock_magnitude,
        'actual_value': shock_magnitude * sigma
    }

    return combined, shock_info


def simulate_univariate_series(
    n_samples: int = 500,
    # Trend parameters
    trend_type: str = 'none',
    trend_slope: float = 0.0,
    trend_intercept: float = 0.0,
    trend_quad: float = 0.0,
    # Seasonal parameters
    seasonal_amplitude: float = 0.0,
    seasonal_period: int = 12,
    seasonal_type: str = 'additive',
    # ARMA parameters
    ar_params: np.ndarray = None,
    ma_params: np.ndarray = None,
    sigma: float = 1.0,
    # Shock parameters
    shock_time: Optional[int] = None,
    shock_magnitude: float = 0.0,
    # General
    seed: Optional[int] = None
) -> tuple:
    """
    Simulate a univariate time series: Trend + Seasonality + ARMA.

    Shock behavior:
    - With ARMA: Shock propagates through AR/MA dynamics
    - Without ARMA: Shock is a permanent level shift

    Returns
    -------
    tuple
        (series, components_dict, shock_info)
        where components_dict contains individual components
    """
    if ar_params is None:
        ar_params = np.array([])
    if ma_params is None:
        ma_params = np.array([])

    # Generate individual components
    trend = generate_trend(n_samples, trend_type, trend_slope, trend_intercept, trend_quad)
    seasonality = generate_seasonality(n_samples, seasonal_period, seasonal_amplitude, seasonal_type)

    shock_info = None

    # Generate ARMA component (with or without shock)
    if shock_time is not None and shock_magnitude != 0:
        combined, shock_info = apply_shock_to_components(
            trend, seasonality,
            ar_params, ma_params,
            shock_time, shock_magnitude,
            sigma, seed, seasonal_type
        )
    else:
        arma = generate_arma_component(ar_params, ma_params, n_samples, sigma, seed)
        combined = combine_components(trend, seasonality, arma, seasonal_type)

    # Create pandas Series
    date_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    series = pd.Series(combined, index=date_index, name='Univariate')

    # Store components for visualization
    components = {
        'trend': pd.Series(trend, index=date_index, name='Trend'),
        'seasonality': pd.Series(seasonality, index=date_index, name='Seasonality'),
    }

    return series, components, shock_info
