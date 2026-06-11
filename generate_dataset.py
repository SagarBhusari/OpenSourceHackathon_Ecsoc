import csv
import random
import math
from datetime import datetime, timedelta

random.seed(42)

def weather_label(sev):
    labels = ['Clear','Clear','Light Cloud','Light Cloud','Drizzle','Drizzle',
              'Moderate Rain','Moderate Rain','Heavy Rain','Heavy Rain','Storm']
    return labels[min(int(sev), 10)]

def traffic_density(vol):
    if vol < 25: return 'Low'
    elif vol < 45: return 'Medium'
    return 'High'

def congestion_index(hour, weather, is_weekday, accident, event):
    score = 20
    is_peak = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    score += is_peak * 25
    score += 0 if is_weekday else -8
    score += weather * 1.5 if weather < 4 else weather * 4.2
    score += accident * 14
    score += event * 15
    noise = random.gauss(0, 3)
    return max(0, min(100, round(score + noise, 1)))

rows = []
start = datetime(2023, 1, 1, 0, 0)
for i in range(8760):
    dt = start + timedelta(hours=i)
    hour = dt.hour
    weekday = 1 if dt.weekday() < 5 else 0
    month = dt.month
    weather_sev = round(random.triangular(0, 10, 2), 1)
    accident = 1 if random.random() < 0.08 else 0
    event = 1 if random.random() < 0.05 else 0

    base_vol = 15
    if weekday:
        if 7 <= hour <= 9:     base_vol = random.randint(38, 50)
        elif 17 <= hour <= 19: base_vol = random.randint(44, 56)
        elif 10 <= hour <= 16: base_vol = random.randint(28, 38)
        elif 5 <= hour <= 6:   base_vol = random.randint(10, 18)
        elif 20 <= hour <= 22: base_vol = random.randint(18, 26)
        else:                  base_vol = random.randint(2, 8)
    else:
        if 10 <= hour <= 14:   base_vol = random.randint(22, 34)
        elif 15 <= hour <= 18: base_vol = random.randint(18, 28)
        else:                  base_vol = random.randint(3, 14)

    if month in [10, 11]: base_vol = int(base_vol * 1.15)
    elif month in [5, 6]:  base_vol = int(base_vol * 0.92)

    volume = base_vol + random.randint(-3, 3)
    ci = congestion_index(hour, weather_sev, weekday, accident, event)
    delay = round(ci * 0.38, 1)

    rows.append({
        'timestamp': dt.strftime('%Y-%m-%d %H:%M'),
        'date': dt.strftime('%Y-%m-%d'),
        'hour': hour,
        'day_of_week': dt.strftime('%A'),
        'month': dt.strftime('%B'),
        'is_weekday': weekday,
        'is_peak_hour': 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0,
        'vehicle_volume_thousands': volume,
        'weather_severity': weather_sev,
        'weather_label': weather_label(weather_sev),
        'accident_present': accident,
        'event_or_holiday': event,
        'congestion_index': ci,
        'traffic_density': traffic_density(volume),
        'estimated_delay_min': delay,
        'location': random.choice(['NH-48 Stretch','Ring Road North','Central Ave','East Bypass','MG Road'])
    })

with open('/home/claude/smart_city_traffic/dataset/urban_traffic_data.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} rows")
print("Columns:", list(rows[0].keys()))
