"""
Test univariate components functionality.
Run this in your venv: python tests/test_components.py
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
from exploration.univariate.components import (
    generate_trend, generate_seasonality,
    generate_arma_component, simulate_univariate_series
)

print("=" * 60)
print("Testing Univariate Components")
print("=" * 60)

# Test 1: Trend component
print("\nTest 1: Trend Components")
print("-" * 60)
try:
    linear_trend = generate_trend(100, 'linear', slope=0.1, intercept=5.0)
    print(f"✓ Linear trend: start={linear_trend[0]:.2f}, end={linear_trend[-1]:.2f}")

    quad_trend = generate_trend(100, 'quadratic', slope=0.1, intercept=5.0, quad_coef=0.001)
    print(f"✓ Quadratic trend: start={quad_trend[0]:.2f}, end={quad_trend[-1]:.2f}")

    exp_trend = generate_trend(100, 'exponential', slope=0.05, intercept=1.0)
    print(f"✓ Exponential trend: start={exp_trend[0]:.2f}, end={exp_trend[-1]:.2f}")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Seasonality component
print("\nTest 2: Seasonality Component")
print("-" * 60)
try:
    season_add = generate_seasonality(120, seasonal_period=12, seasonal_amplitude=3.0, seasonal_type='additive')
    print(f"✓ Additive seasonality: mean={season_add.mean():.4f}, std={season_add.std():.4f}")

    season_mult = generate_seasonality(120, seasonal_period=12, seasonal_amplitude=3.0, seasonal_type='multiplicative')
    print(f"✓ Multiplicative seasonality: mean={season_mult.mean():.4f}")
    print(f"  Should be centered around 1.0 for multiplicative")

except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 3: ARMA component
print("\nTest 3: ARMA Component")
print("-" * 60)
try:
    ar_params = np.array([0.7])
    ma_params = np.array([])
    arma = generate_arma_component(ar_params, ma_params, 100, sigma=1.0, seed=42)
    print(f"✓ ARMA generated: length={len(arma)}, mean={arma.mean():.4f}, std={arma.std():.4f}")

except Exception as e:
    print(f"✗ FAILED: {e}")

# Test 4: Complete univariate series
print("\nTest 4: Complete Univariate Series (Trend + Seasonality + ARMA)")
print("-" * 60)
try:
    series, components, shock_info = simulate_univariate_series(
        n_samples=200,
        trend_type='linear',
        trend_slope=0.05,
        trend_intercept=10.0,
        seasonal_amplitude=1.5,
        seasonal_period=12,
        seasonal_type='additive',
        ar_params=np.array([0.5]),
        ma_params=np.array([]),
        sigma=0.5,
        seed=42
    )

    print(f"✓ Complete series generated!")
    print(f"  Series length: {len(series)}")
    print(f"  Mean: {series.mean():.4f}")
    print(f"  Std: {series.std():.4f}")
    print(f"  Components available: {list(components.keys())}")
    print(f"  Trend component range: [{components['trend'].min():.2f}, {components['trend'].max():.2f}]")
    print(f"  Seasonality component range: [{components['seasonality'].min():.2f}, {components['seasonality'].max():.2f}]")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Series with shock (ARMA present - propagates through dynamics)
print("\nTest 5: Univariate Series with Shock (ARMA present)")
print("-" * 60)
try:
    series_shock, components_shock, shock_info = simulate_univariate_series(
        n_samples=150,
        trend_type='none',
        ar_params=np.array([0.7]),
        ma_params=np.array([]),
        sigma=1.0,
        shock_time=50,
        shock_magnitude=5.0,
        seed=42
    )

    print(f"✓ Series with shock generated!")
    print(f"  Shock info: {shock_info}")
    print(f"  Value at shock time (t=50): {series_shock.iloc[50]:.4f}")
    print(f"  Value 10 periods later (t=60): {series_shock.iloc[60]:.4f}")
    print(f"  Expected: AR(1) should show persistent effect that gradually decays")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Series with shock (NO ARMA - permanent level shift)
print("\nTest 6: Univariate Series with Shock (NO ARMA - level shift)")
print("-" * 60)
try:
    series_shift, components_shift, shock_info = simulate_univariate_series(
        n_samples=150,
        trend_type='linear',
        trend_slope=0.1,
        trend_intercept=5.0,
        ar_params=np.array([]),
        ma_params=np.array([]),
        shock_time=50,
        shock_magnitude=3.0,
        seed=42
    )

    print(f"✓ Series with level shift generated!")
    print(f"  Shock info: {shock_info}")
    print(f"  Value just before shock (t=49): {series_shift.iloc[49]:.4f}")
    print(f"  Value at shock time (t=50): {series_shift.iloc[50]:.4f}")
    print(f"  Value after shock (t=51): {series_shift.iloc[51]:.4f}")
    print(f"  Difference: {series_shift.iloc[51] - series_shift.iloc[49]:.4f}")
    print(f"  Expected: Permanent level shift of ~3.0 from t=50 onward")

except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All component tests completed!")
print("=" * 60)
print("\n💡 Run the Streamlit app to see the Univariate Explorer:")
print("   streamlit run app.py")
print("   Then navigate to 'Univariate Explorer' in the sidebar")
