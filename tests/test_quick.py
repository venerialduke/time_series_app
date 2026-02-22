"""
Quick test to verify the ARMA simulation fix.
Run this in your venv to test the changes.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from exploration.univariate.arma import simulate_arma, check_stationarity

print("Testing ARMA simulation with fixed 'scale' parameter...\n")

# Test 1: Simple AR(1) with φ = 0.5 (should be stationary)
print("=" * 50)
print("Test 1: AR(1) with φ = 0.5")
ar_params = np.array([0.5])
ma_params = np.array([])

# Check stationarity
is_stationary, roots = check_stationarity(ar_params)
print(f"  AR parameter: {ar_params}")
print(f"  Characteristic root: {roots}")
print(f"  Root magnitude: {np.abs(roots)}")
print(f"  Is stationary: {is_stationary}")
print(f"  Expected: stationary (root > 1)")

# Simulate
try:
    series = simulate_arma(ar_params, ma_params, n_samples=100, sigma=1.0, seed=42)
    print(f"  ✓ Simulation successful!")
    print(f"  Series length: {len(series)}")
    print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Test 2: AR(1) with φ = 0.99 (should be barely stationary)
print("\n" + "=" * 50)
print("Test 2: AR(1) with φ = 0.99")
ar_params = np.array([0.99])
is_stationary, roots = check_stationarity(ar_params)
print(f"  AR parameter: {ar_params}")
print(f"  Characteristic root: {roots}")
print(f"  Root magnitude: {np.abs(roots)}")
print(f"  Is stationary: {is_stationary}")
print(f"  Expected: stationary (root slightly > 1)")

# Test 3: ARMA(1,1)
print("\n" + "=" * 50)
print("Test 3: ARMA(1,1) with φ = 0.6, θ = 0.4")
ar_params = np.array([0.6])
ma_params = np.array([0.4])

try:
    series = simulate_arma(ar_params, ma_params, n_samples=100, sigma=1.0, seed=42)
    print(f"  ✓ Simulation successful!")
    print(f"  Series length: {len(series)}")
    print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 50)
print("All tests completed!")
