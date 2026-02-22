"""
Basic tests for ARMA simulation functionality.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from exploration.univariate.arma import (
    simulate_arma, check_stationarity, check_invertibility,
    get_arma_preset, get_all_presets
)


def test_arma_simulation():
    """Test basic ARMA simulation."""
    print("Testing ARMA simulation...")

    # Test AR(1)
    ar_params = np.array([0.7])
    ma_params = np.array([])
    series = simulate_arma(ar_params, ma_params, n_samples=100, seed=42)

    assert len(series) == 100, "Series length mismatch"
    assert series.name == 'ARMA', "Series name mismatch"
    print("✓ AR(1) simulation successful")

    # Test MA(1)
    ar_params = np.array([])
    ma_params = np.array([0.5])
    series = simulate_arma(ar_params, ma_params, n_samples=100, seed=42)

    assert len(series) == 100, "Series length mismatch"
    print("✓ MA(1) simulation successful")

    # Test ARMA(1,1)
    ar_params = np.array([0.6])
    ma_params = np.array([0.4])
    series = simulate_arma(ar_params, ma_params, n_samples=200, seed=42)

    assert len(series) == 200, "Series length mismatch"
    print("✓ ARMA(1,1) simulation successful")


def test_stationarity_check():
    """Test stationarity checking."""
    print("\nTesting stationarity check...")

    # Stationary AR(1)
    ar_params = np.array([0.7])
    is_stationary, roots = check_stationarity(ar_params)
    assert is_stationary, "Should be stationary"
    print(f"✓ AR(1) with φ=0.7 is stationary (root magnitude: {abs(roots[0]):.3f})")

    # Non-stationary AR(1)
    ar_params = np.array([1.2])
    is_stationary, roots = check_stationarity(ar_params)
    assert not is_stationary, "Should not be stationary"
    print(f"✓ AR(1) with φ=1.2 is non-stationary (root magnitude: {abs(roots[0]):.3f})")


def test_invertibility_check():
    """Test invertibility checking."""
    print("\nTesting invertibility check...")

    # Invertible MA(1)
    ma_params = np.array([0.5])
    is_invertible, roots = check_invertibility(ma_params)
    assert is_invertible, "Should be invertible"
    print(f"✓ MA(1) with θ=0.5 is invertible (root magnitude: {abs(roots[0]):.3f})")


def test_presets():
    """Test preset configurations."""
    print("\nTesting preset configurations...")

    presets = get_all_presets()
    assert len(presets) > 0, "Should have presets"
    print(f"✓ Found {len(presets)} preset configurations")

    # Test loading a specific preset
    preset = get_arma_preset('AR1_positive')
    assert preset['ar_order'] == 1, "AR order mismatch"
    assert preset['ma_order'] == 0, "MA order mismatch"
    print("✓ Preset loading successful")


if __name__ == "__main__":
    print("Running ARMA tests...\n")
    print("=" * 50)

    try:
        test_arma_simulation()
        test_stationarity_check()
        test_invertibility_check()
        test_presets()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
