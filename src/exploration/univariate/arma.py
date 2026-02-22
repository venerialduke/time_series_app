"""
ARMA (AutoRegressive Moving Average) model exploration.

This module provides functions for simulating and exploring ARMA processes.
Relies on statsmodels for the core ARMA simulation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from typing import Tuple, Optional, List, Dict


def simulate_arma(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    n_samples: int = 500,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    burnin: int = 500
) -> pd.Series:
    """
    Simulate an ARMA process using statsmodels.

    This is a thin wrapper around statsmodels.tsa.arima_process.arma_generate_sample
    that handles parameter validation and returns a pandas Series.

    Parameters
    ----------
    ar_params : np.ndarray
        AR coefficients (excluding lag 0). Empty array for no AR component.
        Convention: y_t = ar[0]*y_{t-1} + ar[1]*y_{t-2} + ... + noise
    ma_params : np.ndarray
        MA coefficients (excluding lag 0). Empty array for no MA component.
        Convention: y_t = noise + ma[0]*noise_{t-1} + ma[1]*noise_{t-2} + ...
    n_samples : int, default=500
        Number of samples to generate
    sigma : float, default=1.0
        Standard deviation of the white noise process
    seed : int, optional
        Random seed for reproducibility
    burnin : int, default=500
        Number of burn-in samples to discard for stationarity

    Returns
    -------
    pd.Series
        Simulated ARMA time series with DatetimeIndex

    Raises
    ------
    ValueError
        If parameters are invalid

    Examples
    --------
    >>> # AR(1) process with coefficient 0.7
    >>> ar_params = np.array([0.7])
    >>> ma_params = np.array([])
    >>> series = simulate_arma(ar_params, ma_params, n_samples=100, seed=42)

    >>> # ARMA(2, 1) process
    >>> ar_params = np.array([0.5, -0.3])
    >>> ma_params = np.array([0.4])
    >>> series = simulate_arma(ar_params, ma_params, n_samples=200, seed=42)
    """
    # Validate inputs
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    # Ensure arrays are numpy arrays
    ar_params = np.asarray(ar_params)
    ma_params = np.asarray(ma_params)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Prepare parameters for statsmodels
    # statsmodels requires AR/MA arrays to include the zero-lag coefficient (always 1)
    # AR polynomial: (1 - ar[0]*L - ar[1]*L^2 - ...)
    # MA polynomial: (1 + ma[0]*L + ma[1]*L^2 + ...)
    if len(ar_params) > 0:
        ar_full = np.concatenate(([1], -ar_params))
    else:
        ar_full = np.array([1])

    if len(ma_params) > 0:
        ma_full = np.concatenate(([1], ma_params))
    else:
        ma_full = np.array([1])

    # Use statsmodels to generate ARMA samples
    y = arma_generate_sample(
        ar=ar_full,
        ma=ma_full,
        nsample=n_samples,
        scale=sigma,
        burnin=burnin
    )

    # Create pandas Series with date index
    date_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    series = pd.Series(y, index=date_index, name='ARMA')

    return series


def apply_shock_to_arma(
    ar_params: np.ndarray,
    ma_params: np.ndarray,
    shock_time: int,
    shock_magnitude: float,
    n_samples: int = 500,
    sigma: float = 1.0,
    seed: Optional[int] = None,
    burnin: int = 500
) -> Tuple[pd.Series, Dict]:
    """
    Simulate an ARMA process with a shock at a specific time point.

    This function generates an ARMA process and applies a shock at the specified
    time, then propagates it through the AR and MA dynamics.

    Parameters
    ----------
    ar_params : np.ndarray
        AR coefficients (excluding lag 0)
    ma_params : np.ndarray
        MA coefficients (excluding lag 0)
    shock_time : int
        Time index (0-based) when shock occurs
    shock_magnitude : float
        Magnitude of shock in standard deviations
    n_samples : int, default=500
        Number of samples to generate
    sigma : float, default=1.0
        Standard deviation of the white noise process
    seed : int, optional
        Random seed for reproducibility
    burnin : int, default=500
        Number of burn-in samples

    Returns
    -------
    tuple
        (series, shock_info) where series is the time series with shock and
        shock_info contains details about the shock

    Examples
    --------
    >>> ar_params = np.array([0.7])
    >>> ma_params = np.array([])
    >>> series, info = apply_shock_to_arma(ar_params, ma_params, shock_time=100, shock_magnitude=3.0)
    """
    # Validate shock timing
    if shock_time < 0 or shock_time >= n_samples:
        raise ValueError(f"shock_time must be between 0 and {n_samples-1}")

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Ensure arrays are numpy arrays
    ar_params = np.asarray(ar_params)
    ma_params = np.asarray(ma_params)

    # Prepare parameters for statsmodels
    if len(ar_params) > 0:
        ar_full = np.concatenate(([1], -ar_params))
    else:
        ar_full = np.array([1])

    if len(ma_params) > 0:
        ma_full = np.concatenate(([1], ma_params))
    else:
        ma_full = np.array([1])

    # Generate noise series
    np.random.seed(seed if seed is not None else 42)
    noise = np.random.normal(0, sigma, n_samples + burnin)

    # Apply the shock to the noise at the specified time
    # The shock is in standard deviations, so multiply by sigma
    noise[burnin + shock_time] += shock_magnitude * sigma

    # Use scipy's lfilter to apply ARMA dynamics (this is what statsmodels uses internally)
    from scipy import signal

    # Apply the ARMA filter
    # MA part: multiply noise by MA polynomial
    # AR part: recursive filter with AR polynomial
    y = signal.lfilter(ma_full, ar_full, noise)

    # Remove burnin period
    y = y[burnin:]

    # Create pandas Series with date index
    date_index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    series = pd.Series(y, index=date_index, name='ARMA')

    # Store shock information
    shock_info = {
        'time': shock_time,
        'magnitude': shock_magnitude,
        'actual_value': shock_magnitude * sigma,
        'date': date_index[shock_time]
    }

    return series, shock_info


def check_stationarity(ar_params: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Check if AR parameters satisfy stationarity conditions.

    An AR process is stationary if all roots of the AR polynomial lie outside
    the unit circle.

    Parameters
    ----------
    ar_params : np.ndarray
        AR coefficients (excluding lag 0)

    Returns
    -------
    tuple
        (is_stationary, roots) where is_stationary is bool and roots are complex array

    Examples
    --------
    >>> ar_params = np.array([0.7])
    >>> is_stationary, roots = check_stationarity(ar_params)
    >>> print(is_stationary)
    True
    """
    if len(ar_params) == 0:
        return True, np.array([])

    # Create AR polynomial: 1 - ar[0]*z - ar[1]*z^2 - ...
    ar_poly = np.concatenate(([1], -ar_params))
    roots = np.roots(ar_poly)

    # For stationarity, all roots must be outside unit circle
    # i.e., all |root| > 1
    is_stationary = np.all(np.abs(roots) > 1.0)

    return is_stationary, roots


def check_invertibility(ma_params: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Check if MA parameters satisfy invertibility conditions.

    An MA process is invertible if all roots of the MA polynomial lie outside
    the unit circle.

    Parameters
    ----------
    ma_params : np.ndarray
        MA coefficients (excluding lag 0)

    Returns
    -------
    tuple
        (is_invertible, roots) where is_invertible is bool and roots are complex array

    Examples
    --------
    >>> ma_params = np.array([0.5])
    >>> is_invertible, roots = check_invertibility(ma_params)
    >>> print(is_invertible)
    True
    """
    if len(ma_params) == 0:
        return True, np.array([])

    # Create MA polynomial: 1 + ma[0]*z + ma[1]*z^2 + ...
    ma_poly = np.concatenate(([1], ma_params))
    roots = np.roots(ma_poly)

    # For invertibility, all roots must be outside unit circle
    is_invertible = np.all(np.abs(roots) > 1.0)

    return is_invertible, roots


def get_arma_preset(preset_name: str) -> dict:
    """
    Get preset ARMA configurations for common patterns.

    Parameters
    ----------
    preset_name : str
        Name of the preset configuration

    Returns
    -------
    dict
        Dictionary with 'ar_order', 'ma_order', 'ar_params', 'ma_params', 'description'

    Examples
    --------
    >>> preset = get_arma_preset('AR1_positive')
    >>> print(preset['description'])
    'Positive AR(1): Smooth, persistent series'
    """
    presets = {
        'white_noise': {
            'ar_order': 0,
            'ma_order': 0,
            'ar_params': [],
            'ma_params': [],
            'description': 'White Noise: Pure random series with no correlation'
        },
        'AR1_positive': {
            'ar_order': 1,
            'ma_order': 0,
            'ar_params': [0.7],
            'ma_params': [],
            'description': 'Positive AR(1): Smooth, persistent series'
        },
        'AR1_negative': {
            'ar_order': 1,
            'ma_order': 0,
            'ar_params': [-0.7],
            'ma_params': [],
            'description': 'Negative AR(1): Oscillating series'
        },
        'AR2': {
            'ar_order': 2,
            'ma_order': 0,
            'ar_params': [0.5, 0.3],
            'ma_params': [],
            'description': 'AR(2): Depends on two previous values'
        },
        'MA1': {
            'ar_order': 0,
            'ma_order': 1,
            'ar_params': [],
            'ma_params': [0.6],
            'description': 'MA(1): Short-term memory of shocks'
        },
        'ARMA11': {
            'ar_order': 1,
            'ma_order': 1,
            'ar_params': [0.6],
            'ma_params': [0.4],
            'description': 'ARMA(1,1): Combines AR and MA effects'
        },
        'ARMA22': {
            'ar_order': 2,
            'ma_order': 2,
            'ar_params': [0.5, -0.3],
            'ma_params': [0.4, 0.2],
            'description': 'ARMA(2,2): Complex pattern with multiple lags'
        }
    }

    return presets.get(preset_name, presets['white_noise'])


def get_all_presets() -> dict:
    """
    Get all available ARMA presets.

    Returns
    -------
    dict
        Dictionary of all preset configurations
    """
    return {
        'White Noise': get_arma_preset('white_noise'),
        'AR(1) - Positive': get_arma_preset('AR1_positive'),
        'AR(1) - Negative': get_arma_preset('AR1_negative'),
        'AR(2)': get_arma_preset('AR2'),
        'MA(1)': get_arma_preset('MA1'),
        'ARMA(1,1)': get_arma_preset('ARMA11'),
        'ARMA(2,2)': get_arma_preset('ARMA22')
    }
