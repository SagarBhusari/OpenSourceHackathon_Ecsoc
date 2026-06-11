"""
Smart City Traffic Analytics
Backend: Data Cleaning, Feature Engineering & EDA
"""

import csv
import json
import math
from collections import defaultdict

# ── Load Data ────────────────────────────────────────────────────────────────
def load_data(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))

# ── Data Cleaning ────────────────────────────────────────────────────────────
def clean_data(rows):
    cleaned = []
    missing_count = 0
    duplicate_set = set()

    for row in rows:
        key = (row['timestamp'], row['location'])
        if key in duplicate_set:
            continue
        duplicate_set.add(key)

        # Handle missing / malformed values
        try:
            row['congestion_index'] = float(row['congestion_index'])
            row['weather_severity'] = float(row['weather_severity'])
            row['vehicle_volume_thousands'] = int(row['vehicle_volume_thousands'])
            row['hour'] = int(row['hour'])
            row['is_weekday'] = int(row['is_weekday'])
            row['is_peak_hour'] = int(row['is_peak_hour'])
            row['accident_present'] = int(row['accident_present'])
            row['event_or_holiday'] = int(row['event_or_holiday'])
            row['estimated_delay_min'] = float(row['estimated_delay_min'])
        except (ValueError, KeyError):
            missing_count += 1
            continue

        # Clip outliers: congestion index must be 0–100
        row['congestion_index'] = max(0, min(100, row['congestion_index']))
        cleaned.append(row)

    print(f"[Cleaning] {len(rows)} raw rows → {len(cleaned)} clean rows | Dropped: {missing_count} malformed")
    return cleaned

# ── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(rows):
    """
    New features:
      - weather_category: bucketed label for model & viz
      - congestion_category: Low / Medium / High / Critical
      - time_of_day: Night / Morning / Midday / Evening / Night
      - compound_risk: rain × accident interaction flag
      - prev_hour_congestion: lag-1 feature for sequential patterns
    """
    rows_by_loc = defaultdict(list)
    for r in rows:
        rows_by_loc[r['location']].append(r)

    enriched = []
    for loc, loc_rows in rows_by_loc.items():
        loc_rows.sort(key=lambda x: x['timestamp'])
        prev_ci = None
        for r in loc_rows:
            # Weather category
            ws = r['weather_severity']
            if ws < 2:   r['weather_category'] = 'Clear'
            elif ws < 4: r['weather_category'] = 'Overcast'
            elif ws < 6: r['weather_category'] = 'Light Rain'
            elif ws < 8: r['weather_category'] = 'Heavy Rain'
            else:        r['weather_category'] = 'Storm'

            # Congestion category
            ci = r['congestion_index']
            if ci < 35:   r['congestion_category'] = 'Low'
            elif ci < 60: r['congestion_category'] = 'Medium'
            elif ci < 80: r['congestion_category'] = 'High'
            else:         r['congestion_category'] = 'Critical'

            # Time of day
            h = r['hour']
            if h < 5:    r['time_of_day'] = 'Night'
            elif h < 10: r['time_of_day'] = 'Morning'
            elif h < 15: r['time_of_day'] = 'Midday'
            elif h < 20: r['time_of_day'] = 'Evening'
            else:        r['time_of_day'] = 'Night'

            # Compound risk flag (non-obvious feature)
            r['compound_risk'] = 1 if (r['weather_severity'] >= 6 and r['is_peak_hour']) else 0

            # Lag feature
            r['prev_hour_congestion'] = prev_ci if prev_ci is not None else ci
            prev_ci = ci

            enriched.append(r)

    print(f"[Feature Engineering] Added 5 new features: weather_category, congestion_category, time_of_day, compound_risk, prev_hour_congestion")
    return enriched

# ── EDA Summary ──────────────────────────────────────────────────────────────
def compute_eda_summary(rows):
    summary = {}

    # Hourly avg congestion
    hourly = defaultdict(list)
    for r in rows:
        hourly[r['hour']].append(r['congestion_index'])
    summary['hourly_avg_congestion'] = {h: round(sum(v)/len(v), 2) for h, v in hourly.items()}

    # Day-of-week avg
    dow = defaultdict(list)
    for r in rows:
        dow[r['day_of_week']].append(r['congestion_index'])
    summary['dow_avg_congestion'] = {d: round(sum(v)/len(v), 2) for d, v in dow.items()}

    # Monthly avg
    monthly = defaultdict(list)
    for r in rows:
        monthly[r['month']].append(r['congestion_index'])
    summary['monthly_avg_congestion'] = {m: round(sum(v)/len(v), 2) for m, v in monthly.items()}

    # Location avg
    loc = defaultdict(list)
    for r in rows:
        loc[r['location']].append(r['congestion_index'])
    summary['location_avg_congestion'] = {l: round(sum(v)/len(v), 2) for l, v in loc.items()}

    # Weather correlation (simple covariance-based)
    wx = [r['weather_severity'] for r in rows]
    ci = [r['congestion_index'] for r in rows]
    mean_wx = sum(wx)/len(wx)
    mean_ci = sum(ci)/len(ci)
    cov = sum((x - mean_wx)*(y - mean_ci) for x, y in zip(wx, ci)) / len(wx)
    std_wx = math.sqrt(sum((x - mean_wx)**2 for x in wx) / len(wx))
    std_ci = math.sqrt(sum((y - mean_ci)**2 for y in ci) / len(ci))
    summary['weather_congestion_correlation'] = round(cov / (std_wx * std_ci), 4)

    # Accident impact
    acc_ci   = [r['congestion_index'] for r in rows if r['accident_present'] == 1]
    no_acc_ci = [r['congestion_index'] for r in rows if r['accident_present'] == 0]
    summary['accident_impact'] = {
        'with_accident_avg_ci': round(sum(acc_ci)/len(acc_ci), 2),
        'without_accident_avg_ci': round(sum(no_acc_ci)/len(no_acc_ci), 2),
        'lift_pct': round((sum(acc_ci)/len(acc_ci) - sum(no_acc_ci)/len(no_acc_ci)) / (sum(no_acc_ci)/len(no_acc_ci)) * 100, 1)
    }

    # Peak vs non-peak
    peak_ci = [r['congestion_index'] for r in rows if r['is_peak_hour'] == 1]
    offpeak_ci = [r['congestion_index'] for r in rows if r['is_peak_hour'] == 0]
    summary['peak_vs_offpeak'] = {
        'peak_avg': round(sum(peak_ci)/len(peak_ci), 2),
        'offpeak_avg': round(sum(offpeak_ci)/len(offpeak_ci), 2)
    }

    return summary

if __name__ == '__main__':
    rows = load_data('/home/claude/smart_city_traffic/dataset/urban_traffic_data.csv')
    rows = clean_data(rows)
    rows = engineer_features(rows)
    summary = compute_eda_summary(rows)

    with open('/home/claude/smart_city_traffic/backend/eda_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n[EDA] Key findings:")
    print(f"  Weather-Congestion Pearson r = {summary['weather_congestion_correlation']}")
    print(f"  Accident lift on congestion  = +{summary['accident_impact']['lift_pct']}%")
    print(f"  Peak avg CI = {summary['peak_vs_offpeak']['peak_avg']} vs Off-peak = {summary['peak_vs_offpeak']['offpeak_avg']}")
    print("\n[Done] eda_summary.json written.")
