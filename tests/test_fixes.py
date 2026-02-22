"""
Test script to verify all fixes are working.
Run this in your venv: python tests/test_fixes.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from exploration.univariate.arma import simulate_arma, check_stationarity

print("=" * 60)
print("Testing ARMA Fixes")
print("=" * 60)

# Test 1: AR(1) with no MA (this was failing before)
print("\nTest 1: AR(1,0) - AR only, no MA component")
print("-" * 60)
ar_params = np.array([0.5])
ma_params = np.array([])  # Empty MA

is_stationary, roots = check_stationarity(ar_params)
print(f"AR coefficient: {ar_params[0]}")
print(f"Characteristic root: {roots[0]:.4f}")
print(f"Root magnitude: {np.abs(roots[0]):.4f}")
print(f"Is stationary: {is_stationary} (expected: True, root > 1)")

try:
    series = simulate_arma(ar_params, ma_params, n_samples=100, sigma=1.0, seed=42)
    print(f"✓ Simulation successful!")
    print(f"  Series length: {len(series)}")
    print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: MA(1) with no AR (this should also work)
print("\nTest 2: ARMA(0,1) - MA only, no AR component")
print("-" * 60)
ar_params = np.array([])  # Empty AR
ma_params = np.array([0.6])

try:
    series = simulate_arma(ar_params, ma_params, n_samples=100, sigma=1.0, seed=42)
    print(f"✓ Simulation successful!")
    print(f"  Series length: {len(series)}")
    print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: White noise (no AR, no MA)
print("\nTest 3: ARMA(0,0) - White noise")
print("-" * 60)
ar_params = np.array([])
ma_params = np.array([])

try:
    series = simulate_arma(ar_params, ma_params, n_samples=100, sigma=1.0, seed=42)
    print(f"✓ Simulation successful!")
    print(f"  Series length: {len(series)}")
    print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
    print(f"  (White noise should have mean ≈ 0, std ≈ 1)")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: ARMA(1,1) - Both components
print("\nTest 4: ARMA(1,1) - Both AR and MA")
print("-" * 60)
ar_params = np.array([0.6])
ma_params = np.array([0.4])

try:
    series = simulate_arma(ar_params, ma_params, n_samples=100, sigma=1.0, seed=42)
    print(f"✓ Simulation successful!")
    print(f"  Series length: {len(series)}")
    print(f"  Mean: {series.mean():.4f}, Std: {series.std():.4f}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Non-stationary AR (φ > 1 would be non-stationary, but we can't test that with slider limits)
# Instead, test near unit root
print("\nTest 5: Near unit root AR(1) - φ = 0.99")
print("-" * 60)
ar_params = np.array([0.99])
ma_params = np.array([])

is_stationary, roots = check_stationarity(ar_params)
print(f"AR coefficient: {ar_params[0]}")
print(f"Characteristic root: {roots[0]:.4f}")
print(f"Root magnitude: {np.abs(roots[0]):.4f}")
print(f"Is stationary: {is_stationary} (expected: True, barely)")

try:
    series = simulate_arma(ar_params, ma_params, n_samples=100, sigma=1.0, seed=42)
    print(f"✓ Simulation successful!")
    print(f"  Series should be very persistent with slow decay")
except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
