"""
Dynamic Factor Models (DFM).

This module provides functions for simulating and exploring dynamic factor models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy import signal


def simulate_dynamic_factor(
    n_factors: int = 2,
    n_series: int = 5,
    n_samples: int = 500,
    loading_matrix: Optional[np.ndarray] = None,
    factor_ar_params: Optional[list] = None,
    factor_var: float = 1.0,
    idiosyncratic_var: float = 0.5,
    seed: Optional[int] = None,
    burnin: int = 500,
    shock_factor: Optional[int] = None,
    shock_time: Optional[int] = None,
    shock_magnitude: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, Dict]:
    """
    Simulate a Dynamic Factor Model.

    Model:
    y_t = Λ * f_t + ε_t
    f_t = Φ * f_{t-1} + η_t

    where:
    - y_t: observed series (n_series x 1)
    - f_t: latent factors (n_factors x 1)
    - Λ: loading matrix (n_series x n_factors)
    - ε_t: idiosyncratic errors (n_series x 1)
    - Φ: factor AR coefficient matrix (diagonal)

    Parameters
    ----------
    n_factors : int
        Number of latent factors
    n_series : int
        Number of observed series
    n_samples : int
        Number of observations
    loading_matrix : np.ndarray, optional
        Custom loading matrix (n_series x n_factors)
        If None, generates random loadings
    factor_ar_params : list, optional
        AR(1) parameters for each factor
        If None, uses moderate persistence (0.7)
    factor_var : float
        Variance of factor innovations
    idiosyncratic_var : float
        Variance of idiosyncratic errors
    seed : int, optional
        Random seed
    burnin : int
        Burn-in period
    shock_factor : int, optional
        Index of factor to shock (0-based)
    shock_time : int, optional
        Time index for shock
    shock_magnitude : float
        Size of shock in standard deviations

    Returns
    -------
    tuple
        (observed_df, factors_df, loading_matrix, info_dict)
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate or validate loading matrix
    if loading_matrix is None:
        # Generate random loadings with some structure
        loading_matrix = np.random.uniform(-1, 1, size=(n_series, n_factors))
        # Normalize columns to unit variance
        for j in range(n_factors):
            loading_matrix[:, j] /= np.linalg.norm(loading_matrix[:, j])
    else:
        if loading_matrix.shape != (n_series, n_factors):
            raise ValueError(f"Loading matrix shape {loading_matrix.shape} doesn't match ({n_series}, {n_factors})")

    # Set factor AR parameters
    if factor_ar_params is None:
        factor_ar_params = [0.7] * n_factors  # Default moderate persistence
    elif len(factor_ar_params) != n_factors:
        raise ValueError(f"Need {n_factors} AR parameters, got {len(factor_ar_params)}")

    # Generate factors (each follows AR(1) process)
    factors = np.zeros((n_samples + burnin, n_factors))

    for k in range(n_factors):
        # Factor innovations
        innovations = np.random.normal(0, np.sqrt(factor_var), n_samples + burnin)

        # Apply shock if specified
        if shock_factor is not None and shock_time is not None and k == shock_factor:
            shock_std = np.sqrt(factor_var)
            innovations[burnin + shock_time] += shock_magnitude * shock_std

        # Apply AR(1) dynamics
        ar_coef = np.array([1, -factor_ar_params[k]])
        ma_coef = np.array([1])
        factors[:, k] = signal.lfilter(ma_coef, ar_coef, innovations)

    # Remove burnin from factors
    factors = factors[burnin:]

    # Generate idiosyncratic errors
    idio_errors = np.random.normal(
        0,
        np.sqrt(idiosyncratic_var),
        size=(n_samples, n_series)
    )

    # Generate observed series: y = Λ * f + ε
    observed = factors @ loading_matrix.T + idio_errors

    # Create DataFrames
    index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    observed_df = pd.DataFrame(
        observed,
        index=index,
        columns=[f'Series{i+1}' for i in range(n_series)]
    )

    factors_df = pd.DataFrame(
        factors,
        index=index,
        columns=[f'Factor{i+1}' for i in range(n_factors)]
    )

    # Compute variance explained
    signal_var = np.var(factors @ loading_matrix.T, axis=0)
    total_var = np.var(observed, axis=0)
    var_explained = signal_var / total_var

    info = {
        'n_factors': n_factors,
        'n_series': n_series,
        'factor_ar_params': factor_ar_params,
        'variance_explained': var_explained,
        'avg_variance_explained': np.mean(var_explained),
        'shock_info': None
    }

    # Add shock info if shock was applied
    if shock_factor is not None and shock_time is not None:
        info['shock_info'] = {
            'factor': shock_factor,
            'time': shock_time,
            'magnitude': shock_magnitude
        }

    return observed_df, factors_df, loading_matrix, info


def compute_factor_correlation(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of factors.

    Parameters
    ----------
    factors_df : pd.DataFrame
        DataFrame of factor time series

    Returns
    -------
    pd.DataFrame
        Factor correlation matrix
    """
    return factors_df.corr()


def compute_loadings_contribution(
    loading_matrix: np.ndarray,
    factor_names: list,
    series_names: list
) -> pd.DataFrame:
    """
    Create a DataFrame showing factor loadings.

    Parameters
    ----------
    loading_matrix : np.ndarray
        Loading matrix (n_series x n_factors)
    factor_names : list
        Names of factors
    series_names : list
        Names of observed series

    Returns
    -------
    pd.DataFrame
        Loadings with series as rows, factors as columns
    """
    return pd.DataFrame(
        loading_matrix,
        index=series_names,
        columns=factor_names
    )


def generate_block_loading_matrix(
    n_series: int,
    n_factors: int,
    block_size: int = 3
) -> np.ndarray:
    """
    Generate a block-structured loading matrix.

    Each factor loads strongly on a subset of series.

    Parameters
    ----------
    n_series : int
        Number of observed series
    n_factors : int
        Number of factors
    block_size : int
        Number of series per factor block

    Returns
    -------
    np.ndarray
        Loading matrix (n_series x n_factors)
    """
    loadings = np.zeros((n_series, n_factors))

    series_per_factor = n_series // n_factors

    for k in range(n_factors):
        start_idx = k * series_per_factor
        end_idx = min((k + 1) * series_per_factor, n_series)

        # Strong loadings for this block
        loadings[start_idx:end_idx, k] = np.random.uniform(0.7, 1.0, end_idx - start_idx)

        # Weak cross-loadings
        for j in range(n_factors):
            if j != k:
                loadings[start_idx:end_idx, j] = np.random.uniform(-0.2, 0.2, end_idx - start_idx)

    # Handle any remaining series
    if n_series % n_factors != 0:
        remaining_start = (n_factors) * series_per_factor
        for i in range(remaining_start, n_series):
            loadings[i, :] = np.random.uniform(-0.3, 0.3, n_factors)

    return loadings


def check_factor_identification(loading_matrix: np.ndarray) -> Tuple[bool, str]:
    """
    Check basic identification conditions for factor model.

    Parameters
    ----------
    loading_matrix : np.ndarray
        Loading matrix (n_series x n_factors)

    Returns
    -------
    tuple
        (is_identified, message)
    """
    n_series, n_factors = loading_matrix.shape

    # Basic rule: need at least n_factors + 1 series
    if n_series < n_factors + 1:
        return False, f"Need at least {n_factors + 1} series for {n_factors} factors"

    # Check rank
    rank = np.linalg.matrix_rank(loading_matrix)
    if rank < n_factors:
        return False, f"Loading matrix is rank-deficient (rank={rank})"

    return True, "Model is identified"
