"""
Test shock functionality.
Run this in your venv: python tests/test_shock.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import matplotlib.pyplot as plt
from exploration.univariate.arma import apply_shock_to_arma, simulate_arma

print("=" * 60)
print("Testing Shock Functionality")
print("=" * 60)

# Test 1: AR(1) with positive coefficient - should show persistent response
print("\nTest 1: AR(1) with φ=0.7 and 3σ shock at t=100")
print("-" * 60)
ar_params = np.array([0.7])
ma_params = np.array([])

try:
    series_with_shock, shock_info = apply_shock_to_arma(
        ar_params=ar_params,
        ma_params=ma_params,
        shock_time=100,
        shock_magnitude=3.0,
        n_samples=200,
        sigma=1.0,
        seed=42
    )

    print(f"✓ Simulation successful!")
    print(f"  Shock info: {shock_info}")
    print(f"  Series length: {len(series_with_shock)}")
    print(f"  Value at shock time: {series_with_shock.iloc[100]:.4f}")
    print(f"  Value 10 periods after shock: {series_with_shock.iloc[110]:.4f}")
    print(f"  Expected: AR(1) should show gradual decay after shock")

    # Compare with baseline (no shock)
    series_no_shock = simulate_arma(
        ar_params=ar_params,
        ma_params=ma_params,
        n_samples=200,
        sigma=1.0,
        seed=42
    )

    print(f"  Baseline value at t=100: {series_no_shock.iloc[100]:.4f}")
    print(f"  Difference: {series_with_shock.iloc[100] - series_no_shock.iloc[100]:.4f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: MA(1) - should show only short-term response
print("\nTest 2: MA(1) with θ=0.6 and 2σ shock at t=50")
print("-" * 60)
ar_params = np.array([])
ma_params = np.array([0.6])

try:
    series_with_shock, shock_info = apply_shock_to_arma(
        ar_params=ar_params,
        ma_params=ma_params,
        shock_time=50,
        shock_magnitude=2.0,
        n_samples=150,
        sigma=1.0,
        seed=42
    )

    print(f"✓ Simulation successful!")
    print(f"  Shock info: {shock_info}")
    print(f"  Value at shock time (t=50): {series_with_shock.iloc[50]:.4f}")
    print(f"  Value 1 period after (t=51): {series_with_shock.iloc[51]:.4f}")
    print(f"  Value 2 periods after (t=52): {series_with_shock.iloc[52]:.4f}")
    print(f"  Value 5 periods after (t=55): {series_with_shock.iloc[55]:.4f}")
    print(f"  Expected: MA(1) should show effect for 1-2 periods only")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: ARMA(1,1) - combined response
print("\nTest 3: ARMA(1,1) with φ=0.5, θ=0.4 and 2.5σ shock")
print("-" * 60)
ar_params = np.array([0.5])
ma_params = np.array([0.4])

try:
    series_with_shock, shock_info = apply_shock_to_arma(
        ar_params=ar_params,
        ma_params=ma_params,
        shock_time=75,
        shock_magnitude=2.5,
        n_samples=200,
        sigma=1.0,
        seed=42
    )

    print(f"✓ Simulation successful!")
    print(f"  Series mean: {series_with_shock.mean():.4f}")
    print(f"  Series std: {series_with_shock.std():.4f}")
    print(f"  Expected: Combined AR persistence + MA immediate response")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Negative shock
print("\nTest 4: Negative shock (-3σ)")
print("-" * 60)
ar_params = np.array([0.6])
ma_params = np.array([])

try:
    series_with_shock, shock_info = apply_shock_to_arma(
        ar_params=ar_params,
        ma_params=ma_params,
        shock_time=80,
        shock_magnitude=-3.0,
        n_samples=150,
        sigma=1.0,
        seed=42
    )

    print(f"✓ Simulation successful!")
    print(f"  Shock magnitude: {shock_info['magnitude']}")
    print(f"  Expected: Negative shock should push series downward")

except Exception as e:
    print(f"✗ FAILED: {e}")

print("\n" + "=" * 60)
print("All shock tests completed!")
print("=" * 60)
print("\n💡 To see visual impulse responses, run the Streamlit app and enable 'Add Shock to Series'")
