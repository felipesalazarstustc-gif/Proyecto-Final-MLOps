import hashlib, json, os, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "batches"
RAW.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
ZIP = RAW / "seoul_bike.zip"
CSV = RAW / "SeoulBikeData.csv"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()

def load_public_data():
    if not ZIP.exists():
        urllib.request.urlretrieve(URL, ZIP)
    if not CSV.exists():
        with zipfile.ZipFile(ZIP) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            z.extract(name, RAW)
            extracted = RAW / name
            extracted.rename(CSV)
    # UCI file is encoded with latin-1.
    df = pd.read_csv(CSV, encoding="latin1")
    df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values(["Date", "Hour"]).reset_index(drop=True)
    df["timestamp"] = df["Date"] + pd.to_timedelta(df["Hour"], unit="h")
    df["year"] = df["timestamp"].dt.year
    df["month_num"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["weekday_num"] = df["timestamp"].dt.weekday
    df = df.rename(columns={"Rented_Bike_Count":"target"})
    # Keep at least 8 useful variables; target is continuous.
    keep = [
        "timestamp","Hour","TemperatureC","Humidity","Wind_speed_ms",
        "Visibility10m","Dew_point_temperatureC","Solar_RadiationMJm2",
        "Rainfallmm","Snowfall_cm","Seasons","Holiday","Functioning_Day",
        "year","month_num","day","weekday_num","target"
    ]
    df = df[keep].dropna().reset_index(drop=True)
    return df

def inject_drift(df):
    # L0/L1/L2 are untouched. L3/L4 have exactly two numerical variables shifted.
    # Documented magnitudes: Temperature +3.0 C and Humidity +15 percentage points.
    return df

def main():
    df = load_public_data()
    n = len(df)
    # Approximate 40/15/15/15/15, preserving chronological order.
    n0 = int(n * 0.40)
    remaining = n - n0
    cuts = [n0]
    for i in range(1,5):
        cuts.append(n0 + int(remaining*i/4))
    batches = [
        ("L0", df.iloc[:cuts[0]].copy()),
        ("L1", df.iloc[cuts[0]:cuts[1]].copy()),
        ("L2", df.iloc[cuts[1]:cuts[2]].copy()),
        ("L3", df.iloc[cuts[2]:cuts[3]].copy()),
        ("L4", df.iloc[cuts[3]:cuts[4]].copy()),
    ]
    manifest = {
        "source": "UCI Machine Learning Repository - Seoul Bike Sharing Demand",
        "source_url": "https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand",
        "license": "CC BY 4.0",
        "download_date_utc": pd.Timestamp.utcnow().isoformat(),
        "drift": {
            "L3": {"TemperatureC": "+3.0", "Humidity": "+15.0"},
            "L4": {"TemperatureC": "+3.0", "Humidity": "+15.0"},
            "method": "additive shift applied only to the production batch"
        }
    }
    for name, part in batches:
        part = part.copy()
        if name in ("L3","L4"):
            part["TemperatureC"] += 3.0
            part["Humidity"] += 15.0
        part.to_csv(OUT/f"{name}.csv", index=False)
        manifest[name] = {
            "rows": len(part),
            "sha256": sha256(OUT/f"{name}.csv"),
            "min_timestamp": str(part.timestamp.min()),
            "max_timestamp": str(part.timestamp.max())
        }
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
