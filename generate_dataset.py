"""
Run this ONCE to replace data/disease_data.csv with realistic data.

    python generate_dataset.py

Key improvements over v1:
  - Each region has a unique RISK MULTIPLIER (Delhi=2.8x, Shimla=0.5x)
  - Random OUTBREAK EVENTS injected (1-6 week spikes, bigger in high-risk regions)
  - Cross-regional variance so risk comparisons are meaningful
  - COVID-19 realistically peaks 2020-2021 then declines
"""

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "disease_data.csv"

# Higher = more disease burden (denser, hotter, worse infra)
REGION_RISK = {
    "Delhi": 2.8, "Mumbai": 2.5, "Kolkata": 2.4, "Patna": 2.3,
    "Chennai": 2.2, "Bhubaneswar": 2.0, "Guwahati": 1.9,
    "Hyderabad": 1.8, "Lucknow": 1.7, "Ahmedabad": 1.6,
    "Bengaluru": 1.5, "Raipur": 1.5, "Ranchi": 1.4, "Pune": 1.4,
    "Bhopal": 1.3, "Jaipur": 1.2, "Dehradun": 1.0,
    "Chandigarh": 0.9, "Srinagar": 0.7, "Shimla": 0.5,
}

DISEASES = {
    "Dengue":   {"peak_months": [7,8,9,10], "base": 25, "amplitude": 80},
    "Malaria":  {"peak_months": [6,7,8,9],  "base": 18, "amplitude": 60},
    "Cholera":  {"peak_months": [5,6,7],    "base": 8,  "amplitude": 35},
    "Typhoid":  {"peak_months": [4,5,6],    "base": 12, "amplitude": 40},
    "COVID-19": {"peak_months": [1,2,4,5],  "base": 30, "amplitude": 250},
}

CLIMATE = {
    "Delhi":{"temp":(8,42),"humidity":(30,85),"rainfall":(0,40)},
    "Mumbai":{"temp":(18,36),"humidity":(55,95),"rainfall":(0,80)},
    "Bengaluru":{"temp":(15,35),"humidity":(40,85),"rainfall":(0,50)},
    "Chennai":{"temp":(22,42),"humidity":(60,95),"rainfall":(0,60)},
    "Kolkata":{"temp":(10,40),"humidity":(50,95),"rainfall":(0,60)},
    "Hyderabad":{"temp":(14,42),"humidity":(35,85),"rainfall":(0,45)},
    "Ahmedabad":{"temp":(12,45),"humidity":(25,80),"rainfall":(0,35)},
    "Pune":{"temp":(12,38),"humidity":(30,85),"rainfall":(0,50)},
    "Jaipur":{"temp":(5,45),"humidity":(20,75),"rainfall":(0,30)},
    "Lucknow":{"temp":(5,44),"humidity":(35,90),"rainfall":(0,40)},
    "Bhopal":{"temp":(8,43),"humidity":(30,85),"rainfall":(0,45)},
    "Patna":{"temp":(8,42),"humidity":(40,90),"rainfall":(0,45)},
    "Bhubaneswar":{"temp":(14,42),"humidity":(50,95),"rainfall":(0,65)},
    "Guwahati":{"temp":(10,36),"humidity":(55,95),"rainfall":(0,70)},
    "Chandigarh":{"temp":(4,42),"humidity":(30,85),"rainfall":(0,35)},
    "Srinagar":{"temp":(-5,32),"humidity":(30,75),"rainfall":(0,30)},
    "Shimla":{"temp":(-5,28),"humidity":(35,80),"rainfall":(0,40)},
    "Dehradun":{"temp":(5,38),"humidity":(35,85),"rainfall":(0,50)},
    "Ranchi":{"temp":(8,38),"humidity":(35,90),"rainfall":(0,55)},
    "Raipur":{"temp":(10,44),"humidity":(30,90),"rainfall":(0,50)},
}


def seasonal_mult(month, peaks):
    if month in peaks: return 2.5
    dist = [min(abs(month-p), 12-abs(month-p)) for p in peaks]
    return max(1.0, 2.5 - min(dist) * 0.5)


def year_trend(year, disease):
    if disease == "COVID-19":
        return {2020:2.5,2021:3.2,2022:1.8,2023:0.9,2024:0.6,2025:0.4}.get(year,1.0)
    return {2020:1.0,2021:1.1,2022:1.2,2023:1.15,2024:1.3,2025:1.35}.get(year,1.0)


def outbreak_events(rng, n_dates, region_mult):
    events = {}
    n = rng.integers(1, max(2, int(region_mult * 2)))
    for _ in range(n):
        start = rng.integers(0, n_dates)
        dur   = rng.integers(7, 45)
        mag   = rng.uniform(3.0, 8.0) * region_mult
        for d in range(dur):
            idx = start + d
            if idx < n_dates:
                bell = 1 - abs(d - dur/2) / (dur/2)
                events[idx] = events.get(idx, 1.0) + bell * mag
    return events


def generate():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    rng   = np.random.default_rng(42)
    rows  = []

    print(f"Generating {len(REGION_RISK)} regions × {len(DISEASES)} diseases × {len(dates)} days...")

    for state, rmult in REGION_RISK.items():
        cl = CLIMATE[state]
        for disease, cfg in DISEASES.items():
            ob = outbreak_events(rng, len(dates), rmult)
            for i, date in enumerate(dates):
                m, y = date.month, date.year
                s = seasonal_mult(m, cfg["peak_months"])
                t = year_trend(y, disease)
                noise = rng.normal(0, cfg["base"] * 0.2)
                cases = max(0, int(cfg["base"] * s * t * rmult * ob.get(i, 1.0) + noise))

                tmin, tmax = cl["temp"]
                hmin, hmax = cl["humidity"]
                rmin, rmax = cl["rainfall"]
                temp  = round(float(np.clip(rng.normal((tmin+tmax)/2+(m-6)*1.5,3),tmin,tmax)),1)
                hum   = round(float(np.clip(rng.normal((hmin+hmax)/2+(m in cfg["peak_months"])*10,8),hmin,hmax)),1)
                rain  = round(float(np.clip(rng.exponential(rmax/4 if m in cfg["peak_months"] else rmax/10),rmin,rmax)),1)

                rows.append({"date":date.strftime("%Y-%m-%d"),"region":state,
                             "disease":disease,"cases":cases,"temperature":temp,
                             "humidity":hum,"rainfall":rain,"year":y,"month":m})

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Saved: {OUTPUT_PATH}  ({len(df):,} rows)\n")
    print("COVID-19 avg daily cases by region (2021):")
    sub = df[(df["disease"]=="COVID-19") & (df["year"]==2021)]
    for region, avg in sub.groupby("region")["cases"].mean().sort_values(ascending=False).items():
        print(f"  {region:<15} {avg:>6.0f}  {'█' * int(avg/30)}")


if __name__ == "__main__":
    generate()
