"""
Vector Autoregression (VAR) models.

This module provides functions for simulating and exploring VAR processes.
Relies on statsmodels for core VAR functionality.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy import signal


def simulate_var(
    coef_matrices: list,
    n_samples: int = 500,
    sigma: np.ndarray = None,
    seed: Optional[int] = None,
    burnin: int = 500
) -> pd.DataFrame:
    """
    Simulate a VAR process using coefficient matrices.

    Parameters
    ----------
    coef_matrices : list of np.ndarray
        List of coefficient matrices [A1, A2, ..., Ap] where each Ai is (n_vars x n_vars)
        VAR(p): Y_t = A1*Y_{t-1} + A2*Y_{t-2} + ... + Ap*Y_{t-p} + e_t
    n_samples : int
        Number of samples to generate
    sigma : np.ndarray, optional
        Covariance matrix of innovations (n_vars x n_vars)
        If None, uses identity matrix
    seed : int, optional
        Random seed
    burnin : int
        Number of burn-in samples

    Returns
    -------
    pd.DataFrame
        Simulated VAR series with columns ['Var1', 'Var2', ...]
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = coef_matrices[0].shape[0]
    p = len(coef_matrices)  # VAR order

    if sigma is None:
        sigma = np.eye(n_vars)

    # Generate innovations
    innovations = np.random.multivariate_normal(
        mean=np.zeros(n_vars),
        cov=sigma,
        size=n_samples + burnin
    )

    # Initialize series
    Y = np.zeros((n_samples + burnin, n_vars))
    Y[:p] = innovations[:p]  # Initial values

    # Generate VAR process
    for t in range(p, n_samples + burnin):
        Y[t] = innovations[t]
        for lag in range(p):
            Y[t] += coef_matrices[lag] @ Y[t - lag - 1]

    # Remove burnin
    Y = Y[burnin:]

    # Create DataFrame
    columns = [f'Var{i+1}' for i in range(n_vars)]
    df = pd.DataFrame(Y, columns=columns)
    df.index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    return df


def simulate_var_with_shock(
    coef_matrices: list,
    shock_var: int,
    shock_time: int,
    shock_magnitude: float,
    n_samples: int = 500,
    sigma: np.ndarray = None,
    seed: Optional[int] = None,
    burnin: int = 500
) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate VAR process with a shock to one variable.

    Parameters
    ----------
    coef_matrices : list of np.ndarray
        VAR coefficient matrices
    shock_var : int
        Index of variable to shock (0-based)
    shock_time : int
        Time index for shock
    shock_magnitude : float
        Size of shock in standard deviations
    n_samples : int
        Number of samples
    sigma : np.ndarray, optional
        Innovation covariance matrix
    seed : int, optional
        Random seed
    burnin : int
        Burn-in period

    Returns
    -------
    tuple
        (DataFrame, shock_info dict)
    """
    if seed is not None:
        np.random.seed(seed)

    n_vars = coef_matrices[0].shape[0]
    p = len(coef_matrices)

    if sigma is None:
        sigma = np.eye(n_vars)

    # Generate innovations
    innovations = np.random.multivariate_normal(
        mean=np.zeros(n_vars),
        cov=sigma,
        size=n_samples + burnin
    )

    # Apply shock to specified variable
    shock_std = np.sqrt(sigma[shock_var, shock_var])
    innovations[burnin + shock_time, shock_var] += shock_magnitude * shock_std

    # Initialize series
    Y = np.zeros((n_samples + burnin, n_vars))
    Y[:p] = innovations[:p]

    # Generate VAR process
    for t in range(p, n_samples + burnin):
        Y[t] = innovations[t]
        for lag in range(p):
            Y[t] += coef_matrices[lag] @ Y[t - lag - 1]

    # Remove burnin
    Y = Y[burnin:]

    # Create DataFrame
    columns = [f'Var{i+1}' for i in range(n_vars)]
    df = pd.DataFrame(Y, columns=columns)
    df.index = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')

    shock_info = {
        'time': shock_time,
        'variable': shock_var,
        'variable_name': columns[shock_var],
        'magnitude': shock_magnitude,
        'actual_value': shock_magnitude * shock_std
    }

    return df, shock_info


def compute_irf(
    coef_matrices: list,
    sigma: np.ndarray,
    periods: int = 20,
    shock_var: int = 0
) -> np.ndarray:
    """
    Compute Impulse Response Functions for VAR.

    Parameters
    ----------
    coef_matrices : list of np.ndarray
        VAR coefficient matrices
    sigma : np.ndarray
        Innovation covariance matrix
    periods : int
        Number of periods for IRF
    shock_var : int
        Variable to shock (0-based)

    Returns
    -------
    np.ndarray
        IRF array of shape (periods, n_vars)
        IRF[t, i] = response of variable i at time t to shock in shock_var
    """
    n_vars = coef_matrices[0].shape[0]
    p = len(coef_matrices)

    # Initialize IRF
    irf = np.zeros((periods, n_vars))

    # Initial shock (use Cholesky decomposition for orthogonalized shock)
    chol = np.linalg.cholesky(sigma)
    irf[0] = chol[:, shock_var]

    # Compute IRF recursively
    for t in range(1, periods):
        for lag in range(min(p, t)):
            irf[t] += coef_matrices[lag] @ irf[t - lag - 1]

    return irf


def check_var_stability(coef_matrices: list) -> Tuple[bool, np.ndarray]:
    """
    Check stability of VAR process.

    A VAR(p) is stable if all eigenvalues of the companion matrix
    lie inside the unit circle.

    Parameters
    ----------
    coef_matrices : list of np.ndarray
        VAR coefficient matrices [A1, A2, ..., Ap]

    Returns
    -------
    tuple
        (is_stable, eigenvalues)
    """
    n_vars = coef_matrices[0].shape[0]
    p = len(coef_matrices)

    # Construct companion matrix
    # For VAR(p), companion matrix is (n_vars*p x n_vars*p)
    companion = np.zeros((n_vars * p, n_vars * p))

    # First n_vars rows: [A1, A2, ..., Ap]
    for i, A in enumerate(coef_matrices):
        companion[:n_vars, i*n_vars:(i+1)*n_vars] = A

    # Remaining rows: identity blocks
    if p > 1:
        companion[n_vars:, :n_vars*(p-1)] = np.eye(n_vars * (p - 1))

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(companion)

    # VAR is stable if all eigenvalues are inside unit circle
    is_stable = np.all(np.abs(eigenvalues) < 1.0)

    return is_stable, eigenvalues
