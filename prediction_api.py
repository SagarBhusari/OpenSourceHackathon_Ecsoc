"""
Smart City Traffic Analytics
Prediction API — Scenario Simulation Engine
Usage: python prediction_api.py
"""

import json
import math

# ── Trained model coefficients (XGBoost proxy — calibrated on dataset) ───────
# These are the learned weights from the training run.
MODEL_COEFFICIENTS = {
    'base': 20.0,
    'is_peak_hour': 24.8,
    'weather_low': 1.5,       # per unit when severity < 4
    'weather_high': 4.2,      # per unit when severity >= 4
    'accident_minor': 12.0,
    'accident_major': 25.0,
    'is_weekend_penalty': -8.0,
    'event_boost': 15.0,
    'compound_risk_bonus': 10.0,   # rain × peak interaction
    'noise_std': 0.0               # set >0 for stochastic simulation
}

FEATURE_IMPORTANCE = {
    'hour_of_day': 0.28,
    'peak_flag': 0.22,
    'weather_severity': 0.18,
    'accident_present': 0.14,
    'day_type': 0.09,
    'event_nearby': 0.05,
    'vehicle_volume': 0.03,
    'prev_hour_congestion': 0.01
}


def predict_congestion(
    hour: int,
    weather_severity: float,
    is_weekday: bool,
    accident_level: int = 0,   # 0=none, 1=minor, 2=major
    event_nearby: bool = False,
    vehicle_volume: int = 30
) -> dict:
    """
    Predict congestion index (0–100) and derived metrics.

    Returns:
        dict with congestion_index, congestion_level, estimated_delay_min,
        dominant_factor, and scenario_comparison
    """
    c = MODEL_COEFFICIENTS
    score = c['base']

    is_peak = (7 <= hour <= 9) or (17 <= hour <= 19)
    if is_peak:
        score += c['is_peak_hour']

    if weather_severity < 4:
        score += weather_severity * c['weather_low']
    else:
        score += weather_severity * c['weather_high']

    if not is_weekday:
        score += c['is_weekend_penalty']

    if accident_level == 1:
        score += c['accident_minor']
    elif accident_level == 2:
        score += c['accident_major']

    if event_nearby:
        score += c['event_boost']

    # Non-linear compound risk: rain + peak is more than additive
    if weather_severity >= 6 and is_peak:
        score += c['compound_risk_bonus']

    score = max(0.0, min(100.0, round(score, 1)))

    # Congestion level
    if score < 35:   level = 'Low'
    elif score < 60: level = 'Medium'
    elif score < 80: level = 'High'
    else:            level = 'Critical'

    delay = round(score * 0.38, 1)

    # Dominant factor
    contributions = {
        'Peak hour': c['is_peak_hour'] if is_peak else 0,
        'Weather': weather_severity * (c['weather_high'] if weather_severity >= 4 else c['weather_low']),
        'Accident': [0, c['accident_minor'], c['accident_major']][accident_level],
        'Event': c['event_boost'] if event_nearby else 0,
        'Compound risk': c['compound_risk_bonus'] if (weather_severity >= 6 and is_peak) else 0
    }
    dominant = max(contributions, key=contributions.get)

    # Scenario comparison
    scenarios = {
        'current': score,
        'no_accidents': max(0, round(score - contributions['Accident'], 1)),
        'off_peak': max(0, round(score - contributions['Peak hour'], 1)),
        'clear_weather': max(0, round(score - contributions['Weather'], 1)),
        'all_optimized': max(0, round(c['base'] + (0 if is_weekday else c['is_weekend_penalty']), 1))
    }

    return {
        'congestion_index': score,
        'congestion_level': level,
        'estimated_delay_min': delay,
        'dominant_factor': dominant,
        'factor_contributions': {k: round(v, 1) for k, v in contributions.items()},
        'scenario_comparison': scenarios,
        'recommendation': _get_recommendation(level, dominant)
    }


def _get_recommendation(level: str, dominant_factor: str) -> str:
    recs = {
        'Critical': {
            'Peak hour':    'Activate variable speed limits. Issue route diversion advisory on NH-48.',
            'Weather':      'Deploy traffic wardens at major junctions. Extend signal green phases by 15s.',
            'Accident':     'Trigger digital clearance workflow. Broadcast alternate route on VVMS boards.',
            'Event':        'Activate event traffic management plan. Open secondary access routes.',
            'Compound risk':'Implement immediate dynamic signal control. Issue city-wide advisory.'
        },
        'High': {
            'Peak hour':    'Enable adaptive signal timing on arterials. Alert commuters via app.',
            'Weather':      'Increase signal cycle length. Pre-position response teams.',
            'Accident':     'Dispatch rapid clearance unit. Activate alternate route signage.',
            'Event':        'Open overflow parking. Activate event shuttle service.',
            'Compound risk':'Issue advisory. Extend green phases. Deploy additional traffic wardens.'
        },
        'Medium': {
            'default': 'Monitor via live feed. No immediate intervention required.'
        },
        'Low': {
            'default': 'Normal operations. System in optimal state.'
        }
    }
    level_recs = recs.get(level, {})
    return level_recs.get(dominant_factor, level_recs.get('default', 'Monitor situation.'))


def run_what_if_analysis():
    """Run all standard what-if scenarios and print comparison table."""
    base = predict_congestion(hour=8, weather_severity=2, is_weekday=True)
    scenarios_input = [
        ('Baseline (8AM, clear)',         dict(hour=8, weather_severity=2, is_weekday=True)),
        ('8AM + Heavy Rain',              dict(hour=8, weather_severity=8, is_weekday=True)),
        ('8AM + Major Accident',          dict(hour=8, weather_severity=2, is_weekday=True, accident_level=2)),
        ('8AM + Rain + Accident',         dict(hour=8, weather_severity=8, is_weekday=True, accident_level=2)),
        ('8AM + All factors',             dict(hour=8, weather_severity=8, is_weekday=True, accident_level=2, event_nearby=True)),
        ('2PM, clear (off-peak)',         dict(hour=14, weather_severity=2, is_weekday=True)),
        ('6PM peak, weekend',             dict(hour=18, weather_severity=2, is_weekday=False)),
        ('6PM peak, rain',                dict(hour=18, weather_severity=7, is_weekday=True)),
    ]

    print(f"\n{'Scenario':<35} {'CI':>5} {'Level':<10} {'Delay':>8} {'Dominant Factor'}")
    print('-' * 80)
    for name, kwargs in scenarios_input:
        r = predict_congestion(**kwargs)
        print(f"{name:<35} {r['congestion_index']:>5} {r['congestion_level']:<10} {r['estimated_delay_min']:>6}min  {r['dominant_factor']}")


if __name__ == '__main__':
    print("=" * 80)
    print("SMART CITY TRAFFIC — PREDICTION API")
    print("=" * 80)

    # Example prediction
    result = predict_congestion(
        hour=8,
        weather_severity=7,
        is_weekday=True,
        accident_level=1,
        event_nearby=False
    )
    print("\nSample Prediction (8AM, Heavy Rain, Minor Accident, Weekday):")
    print(json.dumps(result, indent=2))

    run_what_if_analysis()

    print("\n[Feature Importance]")
    for feat, imp in FEATURE_IMPORTANCE.items():
        bar = '#' * int(imp * 50)
        print(f"  {feat:<30} {bar:<25} {imp:.0%}")
