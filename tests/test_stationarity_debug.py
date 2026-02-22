"""
Debug stationarity check to understand the root calculation.
Run this in your venv: python tests/test_stationarity_debug.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from exploration.univariate.arma import check_stationarity, check_invertibility

print("=" * 60)
print("Debugging Stationarity Check")
print("=" * 60)

# Test case that user reported: AR coefficient 0.5
print("\nTest: AR(1) with φ = 0.5")
print("-" * 60)
ar_params = np.array([0.5])

# Manual calculation
print("Manual calculation:")
print(f"  AR parameter: φ = {ar_params[0]}")
print(f"  AR polynomial: 1 - {ar_params[0]}*z")
print(f"  Setting to zero: 1 - {ar_params[0]}*z = 0")
print(f"  Solving for z: z = 1/{ar_params[0]} = {1/ar_params[0]}")
print(f"  Root magnitude: |{1/ar_params[0]}| = {abs(1/ar_params[0])}")
print(f"  Expected: Stationary (root > 1)")

# Using our function
is_stationary, roots = check_stationarity(ar_params)
print("\nFunction output:")
print(f"  Is stationary: {is_stationary}")
print(f"  Roots: {roots}")
print(f"  Root magnitudes: {np.abs(roots)}")

# Check the polynomial we're creating
ar_poly = np.concatenate(([1], -ar_params))
print(f"\nPolynomial used: {ar_poly}")
print(f"  This represents: {ar_poly[0]} + ({ar_poly[1]})*z")

# Use numpy to verify
np_roots = np.roots(ar_poly)
print(f"  NumPy roots: {np_roots}")
print(f"  NumPy root magnitudes: {np.abs(np_roots)}")

print("\n" + "=" * 60)
print("Test: AR(1) with φ = 0.9 (should be barely stationary)")
print("-" * 60)
ar_params = np.array([0.9])
is_stationary, roots = check_stationarity(ar_params)
print(f"  AR parameter: φ = {ar_params[0]}")
print(f"  Is stationary: {is_stationary}")
print(f"  Root magnitude: {np.abs(roots[0]):.4f}")
print(f"  Expected: {1/ar_params[0]:.4f} (stationary)")

print("\n" + "=" * 60)
print("Test: MA(1) with θ = 0.5")
print("-" * 60)
ma_params = np.array([0.5])

print("Manual calculation:")
print(f"  MA parameter: θ = {ma_params[0]}")
print(f"  MA polynomial: 1 + {ma_params[0]}*z")
print(f"  Setting to zero: 1 + {ma_params[0]}*z = 0")
print(f"  Solving for z: z = -1/{ma_params[0]} = {-1/ma_params[0]}")
print(f"  Root magnitude: |{-1/ma_params[0]}| = {abs(-1/ma_params[0])}")
print(f"  Expected: Invertible (root > 1)")

is_invertible, roots = check_invertibility(ma_params)
print("\nFunction output:")
print(f"  Is invertible: {is_invertible}")
print(f"  Roots: {roots}")
print(f"  Root magnitudes: {np.abs(roots)}")

print("\n" + "=" * 60)
print("Conclusion:")
print("-" * 60)
print("If you see root magnitudes < 1, there may be an issue with")
print("how the polynomial is being constructed or interpreted.")
print("=" * 60)
