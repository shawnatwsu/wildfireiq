"""
WildfireIQ Pipeline — Zero keys, zero downloads, fully automated.

What this does every morning:
  1. Generate synthetic fire training data from fire science equations
  2. Train a real MLP neural network on that data
  3. Export the model weights as JSON (browser runs inference)
  4. Fetch live weather for a grid of points (Open-Meteo, free, no key)
  5. Run ML inference to get risk scores for every grid cell
  6. Fetch active fires (NASA FIRMS public CSV, no key)
  7. Fetch NOAA red flag warnings (weather.gov API, no key)
  8. Write docs/data/latest.json and docs/model.json
  9. GitHub Actions commits + deploys everything

Dependencies: pip install requests numpy scikit-learn
"""

import json, math, os, time, sys
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Region ────────────────────────────────────────────────────────────────────
REGION   = "california"
BBOX     = (-124.5, 32.5, -114.0, 42.0)   # west, south, east, north
GRID_DEG = 0.25                            # ~28km grid

OUT_DIR  = Path("docs/data")
MODEL_OUT = Path("docs/model.json")

# ═════════════════════════════════════════════════════════════════════════════
# PART 1 — TRAIN THE ML MODEL
# Physics-informed synthetic data: generate samples from fire science equations,
# then train a neural net so the browser can run real ML inference.
# This is "physics-informed machine learning" — used in real wildfire research.
# ═════════════════════════════════════════════════════════════════════════════

def compute_fwi_scalar(temp, rh, wind_kmh, precip):
    """Canadian FWI system — used to generate ground-truth labels for training."""
    rh = max(5.0, min(100.0, rh))
    ffmc0, dmc0, dc0 = 85.0, 6.0, 15.0

    # FFMC
    mo = 147.2 * (101 - ffmc0) / (59.5 + ffmc0)
    if precip > 0.5:
        rf = precip - 0.5
        mo = min(mo + 42.5 * rf * math.exp(-100/(251-mo)) * (1 - math.exp(-6.93/(rf+1e-9))), 250)
    Ed = 0.942*rh**0.679 + 11*math.exp((rh-100)/10) + 0.18*(21.1-temp)*(1-math.exp(-0.115*rh))
    Ew = 0.618*rh**0.753 + 10*math.exp((rh-100)/10) + 0.18*(21.1-temp)*(1-math.exp(-0.115*rh))
    ko = 0.424*(1-(rh/100)**1.7) + 0.0694*math.sqrt(wind_kmh)*(1-(rh/100)**8)
    kd = ko * 0.581 * math.exp(0.0365*temp)
    m  = Ed + (mo-Ed)*10**(-kd) if mo > Ed else Ew - (Ew-mo)*10**(-kd)
    ffmc = max(0.0, min(101.0, 59.5*(250-m)/(147.2+m)))

    # ISI
    m2 = 147.2*(101-ffmc)/(59.5+ffmc)
    isi = 0.208 * math.exp(0.05039*wind_kmh) * 91.9*math.exp(-0.1386*m2)*(1+m2**5.31/4.93e7)

    # BUI (simplified)
    bui = max(0.0, dmc0 * 0.8 * dc0 / (dmc0 + 0.4*dc0 + 1e-9))

    # FWI
    fd  = 0.626*bui**0.809 + 2 if bui <= 80 else 1000/(25+108.64*math.exp(-0.023*bui))
    B   = 0.1 * isi * fd
    fwi = math.exp(2.72*(0.434*math.log(max(B,1e-9)))**0.647) if B > 1 else B
    return fwi


def generate_training_data(n=15000, seed=42):
    """
    Generate synthetic training samples spanning the full range of fire conditions.
    Each sample = (weather features) → risk score.

    The labels come from the FWI system + probabilistic noise that simulates
    real fire ignition uncertainty (same conditions don't always produce fire).
    This is how physics-informed ML works in research.
    """
    rng = np.random.default_rng(seed)

    X, y = [], []

    for _ in range(n):
        # Realistic ranges for fire-prone western US
        temp      = rng.uniform(-5, 48)          # °C
        rh        = rng.uniform(3, 98)            # %
        wind_ms   = rng.uniform(0, 22)            # m/s
        precip_7d = rng.exponential(3.0)          # mm, skewed low in fire season
        month     = rng.integers(1, 13)           # 1–12
        elev_norm = rng.uniform(0, 1)             # elevation proxy

        wind_kmh  = wind_ms * 3.6
        vpd       = max(0, 6.112 * math.exp(17.67*temp/(temp+243.5)) * (1 - rh/100))

        fwi = compute_fwi_scalar(temp, rh, wind_kmh, precip_7d / 7)

        # Base risk from FWI (sigmoid centered at 20)
        base_risk = 1 / (1 + math.exp(-0.09 * (fwi - 18)))

        # Seasonality: fire season (Jun–Nov) amplifies risk
        season_factor = 0.6 + 0.4 * math.sin(math.pi * (month - 4) / 7) if 4 <= month <= 11 else 0.5

        # Elevation: mid-elevation (500-2000m) has most fuel
        elev_factor = 1.0 if 0.2 < elev_norm < 0.7 else 0.75

        risk = float(np.clip(base_risk * season_factor * elev_factor + rng.normal(0, 0.04), 0, 1))

        # Feature vector — these are what the browser will compute from live weather
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        precip_inv = 1.0 - min(precip_7d / 50.0, 1.0)   # inverted: more precip = lower risk

        X.append([temp, rh, wind_ms, vpd, precip_inv, month_sin, month_cos, elev_norm])
        y.append(risk)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model():
    """
    Train a small MLP neural network on the synthetic fire data.
    Architecture: 8 → 48 → 24 → 1   (sigmoid output)
    ~1,500 parameters — small enough to ship as JSON, fast enough for browser.
    """
    print("Generating training data…")
    X, y = generate_training_data(n=15000)

    print(f"Training MLP on {len(X):,} samples…")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=(48, 24),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=42,
        learning_rate_init=0.003,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
    )
    model.fit(X_scaled, y)

    train_pred = model.predict(X_scaled)
    r2 = float(1 - np.var(y - train_pred) / np.var(y))
    print(f"Model trained — R² = {r2:.4f}")

    return model, scaler


def export_model_json(model, scaler):
    """
    Export the trained model weights + scaler params as JSON.
    The browser loads this and runs the exact same forward pass in JavaScript.
    """
    weights = [layer.tolist() for layer in model.coefs_]
    biases  = [layer.tolist() for layer in model.intercepts_]

    payload = {
        "version":   "1.0",
        "trained_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": {
            "input_dim":    8,
            "hidden_sizes": [48, 24],
            "output_dim":   1,
            "activation":   "relu",
            "output_activation": "sigmoid",
        },
        "feature_names": [
            "temp_c", "rh_pct", "wind_ms", "vpd_kpa",
            "precip_inv", "month_sin", "month_cos", "elev_norm"
        ],
        "scaler": {
            "mean":  scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "weights": weights,
        "biases":  biases,
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUT, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = MODEL_OUT.stat().st_size / 1024
    print(f"Model exported → {MODEL_OUT} ({size_kb:.1f} KB)")
    return payload


# ═════════════════════════════════════════════════════════════════════════════
# PART 2 — FETCH LIVE DATA
# All sources: zero API keys, zero sign-ups, all public.
# ═════════════════════════════════════════════════════════════════════════════

def build_grid():
    west, south, east, north = BBOX
    pts = []
    lat = south
    while lat <= north + 0.001:
        lon = west
        while lon <= east + 0.001:
            pts.append({"lat": round(lat, 3), "lon": round(lon, 3)})
            lon = round(lon + GRID_DEG, 3)
        lat = round(lat + GRID_DEG, 3)
    print(f"Grid: {len(pts)} points at {GRID_DEG}° spacing")
    return pts


def fetch_weather(lat, lon):
    """Open-Meteo — completely free, no API key, ~1km resolution, live + 7d forecast."""
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation,vapor_pressure_deficit",
            "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max,et0_fao_evapotranspiration",
            "past_days": 7, "forecast_days": 7,
            "timezone": "auto", "wind_speed_unit": "ms",
        }, timeout=12)
        r.raise_for_status()
        return r.json()
    except:
        return None


def fetch_all_weather(pts):
    results = {}
    print(f"Fetching weather for {len(pts)} points…")
    with ThreadPoolExecutor(max_workers=25) as ex:
        futs = {ex.submit(fetch_weather, p["lat"], p["lon"]): p for p in pts}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            d = fut.result()
            if d:
                results[(p["lat"], p["lon"])] = d
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(pts)} complete…")
    print(f"Weather fetched: {len(results)}/{len(pts)} points")
    return results


def extract_weather_features(data):
    """Pull current conditions + 7-day history from Open-Meteo response."""
    hourly = data.get("hourly", {})
    daily  = data.get("daily",  {})

    now = datetime.now(timezone.utc)
    times = hourly.get("time", [])
    idx = 0
    for i, t in enumerate(times):
        try:
            if datetime.fromisoformat(t).replace(tzinfo=timezone.utc) <= now:
                idx = i
        except: pass

    def h(k, fb=0):
        v = hourly.get(k, [])
        return float(v[idx]) if idx < len(v) and v[idx] is not None else fb

    def d(k, i=0, fb=0):
        v = daily.get(k, [])
        return float(v[i]) if i < len(v) and v[i] is not None else fb

    temp     = h("temperature_2m")
    rh       = max(5.0, min(100.0, h("relativehumidity_2m", 50)))
    wind_ms  = h("windspeed_10m", 2)
    precip   = h("precipitation", 0)
    vpd      = max(0, h("vapor_pressure_deficit", 0))
    month    = now.month

    # 7-day rolling precip from daily
    precip_7d = sum(d("precipitation_sum", i, 0) for i in range(min(7, len(daily.get("precipitation_sum", [])))))

    # Build forecast risk inputs (next 7 days)
    forecast_raw = []
    n_daily = len(daily.get("temperature_2m_max", []))
    for i in range(7, min(14, n_daily)):
        forecast_raw.append({
            "temp_max":  d("temperature_2m_max", i),
            "precip":    d("precipitation_sum", i, 0),
            "wind_max":  d("windspeed_10m_max", i, 2),
            "et0":       d("et0_fao_evapotranspiration", i, 2),
        })

    return {
        "temp": temp, "rh": rh, "wind_ms": wind_ms,
        "vpd": vpd, "precip": precip, "precip_7d": precip_7d,
        "month": month, "forecast_raw": forecast_raw,
    }


def run_ml_inference_python(features_vec, model, scaler):
    """Run the trained MLP on a single feature vector."""
    x = np.array([features_vec], dtype=np.float32)
    x_scaled = scaler.transform(x)
    risk = float(np.clip(model.predict(x_scaled)[0], 0, 1))
    return risk


def features_to_vec(w, elev_norm=0.4):
    month_sin = math.sin(2 * math.pi * w["month"] / 12)
    month_cos = math.cos(2 * math.pi * w["month"] / 12)
    precip_inv = 1.0 - min(w.get("precip_7d", 0) / 50.0, 1.0)
    return [w["temp"], w["rh"], w["wind_ms"], w["vpd"], precip_inv, month_sin, month_cos, elev_norm]


def compute_forecast(forecast_raw, model, scaler, base_month):
    forecast = []
    for i, day in enumerate(forecast_raw[:7]):
        T   = day["temp_max"]
        rh  = max(5, min(100, 80 - day["et0"] * 6))
        w   = day["wind_max"]
        p7d = day["precip"]
        vpd = max(0, 6.112 * math.exp(17.67*T/(T+243.5)) * (1 - rh/100))
        month = ((base_month - 1 + i + 1) % 12) + 1
        vec = features_to_vec({"temp":T,"rh":rh,"wind_ms":w,"vpd":vpd,"precip_7d":p7d,"month":month})
        risk = run_ml_inference_python(vec, model, scaler)
        level = "Low" if risk < 0.25 else "Moderate" if risk < 0.45 else "High" if risk < 0.65 else "Very High" if risk < 0.82 else "Extreme"
        forecast.append({"day": i+1, "risk": round(risk, 3), "level": level})
    return forecast


def build_risk_grid(pts, weather_data, model, scaler):
    grid = []
    for p in pts:
        raw = weather_data.get((p["lat"], p["lon"]))
        if not raw:
            continue
        w    = extract_weather_features(raw)
        vec  = features_to_vec(w)
        risk = run_ml_inference_python(vec, model, scaler)
        level = "Low" if risk < 0.25 else "Moderate" if risk < 0.45 else "High" if risk < 0.65 else "Very High" if risk < 0.82 else "Extreme"
        fwi  = round(compute_fwi_scalar(w["temp"], w["rh"], w["wind_ms"]*3.6, w["precip"]), 1)
        fc   = compute_forecast(w["forecast_raw"], model, scaler, w["month"])
        grid.append({
            "lat": p["lat"], "lon": p["lon"],
            "risk": round(risk, 3), "level": level, "fwi": fwi,
            "weather": {
                "temp_c":    round(w["temp"], 1),
                "rh_pct":    round(w["rh"], 1),
                "wind_ms":   round(w["wind_ms"], 1),
                "wind_kmh":  round(w["wind_ms"] * 3.6, 1),
                "vpd_kpa":   round(w["vpd"], 2),
                "precip_7d": round(w.get("precip_7d", 0), 1),
            },
            "forecast": fc,
        })
    return grid


def fetch_firms_fires():
    """
    NASA FIRMS public CSV — no API key required.
    Direct public file access, updated every few hours.
    """
    url = "https://firms.modaps.eosdis.nasa.gov/data/active_fire/suomi-npp-viirs-c2/csv/SUOMI_VIIRS_C2_USA_contiguous_and_Hawaii_24h.csv"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return []

        headers = [h.strip() for h in lines[0].split(",")]
        west, south, east, north = BBOX
        fires = []

        for line in lines[1:]:
            vals = line.split(",")
            if len(vals) < len(headers):
                continue
            row = dict(zip(headers, vals))
            try:
                lat = float(row.get("latitude", 0))
                lon = float(row.get("longitude", 0))
                if not (south <= lat <= north and west <= lon <= east):
                    continue
                conf = row.get("confidence", "n").strip().lower()
                if conf == "l":
                    continue
                fires.append({
                    "lat":  round(lat, 4),
                    "lon":  round(lon, 4),
                    "frp":  round(float(row.get("frp", 0)), 1),
                    "bright": round(float(row.get("bright_ti4", 300)), 1),
                    "conf": conf,
                    "dt":   row.get("acq_date", ""),
                })
            except:
                continue

        print(f"Active fires in region: {len(fires)}")
        return fires
    except Exception as e:
        print(f"FIRMS fetch error: {e}")
        return []


def fetch_noaa_alerts():
    """
    NOAA Weather.gov API — completely open, no API key.
    Returns active Red Flag Warnings and Fire Weather Watches for California.
    """
    try:
        r = requests.get(
            "https://api.weather.gov/alerts/active",
            params={"area": "CA", "event": "Red Flag Warning"},
            headers={"User-Agent": "WildfireIQ/1.0 (github.com/wildfireiq)"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        alerts = []
        for feat in data.get("features", []):
            p = feat.get("properties", {})
            alerts.append({
                "event":       p.get("event", ""),
                "headline":    p.get("headline", ""),
                "area":        p.get("areaDesc", ""),
                "severity":    p.get("severity", ""),
                "onset":       p.get("onset", ""),
                "expires":     p.get("expires", ""),
                "description": p.get("description", "")[:300],
            })
        print(f"NOAA alerts: {len(alerts)} active")
        return alerts
    except Exception as e:
        print(f"NOAA alerts error: {e}")
        return []


def fetch_air_quality_sample():
    """
    Open-Meteo Air Quality API — free, no key.
    Sample a few points across the region for smoke/PM2.5.
    """
    sample_pts = [
        (36.7, -119.7),  # Central Valley
        (37.8, -122.4),  # Bay Area
        (34.0, -118.2),  # LA
        (38.5, -121.5),  # Sacramento
    ]
    aq_data = []
    for lat, lon in sample_pts:
        try:
            r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
                "latitude": lat, "longitude": lon,
                "hourly": "pm2_5,us_aqi,dust",
                "forecast_days": 1,
                "timezone": "auto",
            }, timeout=10)
            r.raise_for_status()
            d = r.json()
            pm25_vals = d.get("hourly", {}).get("pm2_5", [None]*24)
            aqi_vals  = d.get("hourly", {}).get("us_aqi", [None]*24)
            pm25 = next((v for v in pm25_vals if v is not None), 0)
            aqi  = next((v for v in aqi_vals if v is not None), 0)
            aq_data.append({"lat": lat, "lon": lon, "pm25": round(pm25, 1), "aqi": round(aqi or 0)})
        except:
            pass
    return aq_data


# ═════════════════════════════════════════════════════════════════════════════
# PART 3 — WRITE OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════

def write_outputs(grid, fires, alerts, aq, model_meta):
    now = datetime.now(timezone.utc).isoformat()
    risks = [p["risk"] for p in grid]
    fwis  = [p["fwi"]  for p in grid]
    avg_risk  = round(sum(risks) / len(risks), 3) if risks else 0
    max_risk  = round(max(risks), 3) if risks else 0
    max_fwi   = round(max(fwis), 1)  if fwis  else 0
    high_n    = sum(1 for r in risks if r > 0.65)
    area_km2  = round(high_n * (GRID_DEG * 111) ** 2)

    level = ("Low" if avg_risk < .25 else "Moderate" if avg_risk < .45 else
             "High" if avg_risk < .65 else "Very High" if avg_risk < .82 else "Extreme")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "meta": {
            "updated_utc": now,
            "region": REGION,
            "grid_deg": GRID_DEG,
            "n_points": len(grid),
            "model_version": model_meta.get("trained_utc", ""),
            "sources": ["Open-Meteo (weather)", "NASA FIRMS (fires)", "NOAA weather.gov (alerts)", "Open-Meteo AQ (smoke)"],
        },
        "summary": {
            "avg_risk": avg_risk, "max_risk": max_risk, "max_fwi": max_fwi,
            "level": level, "high_risk_km2": area_km2,
            "active_fires": len(fires), "red_flag_warnings": len(alerts),
        },
        "grid":   grid,
        "fires":  fires,
        "alerts": alerts,
        "air_quality": aq,
    }

    path = OUT_DIR / "latest.json"
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"\n✅ latest.json → {path} ({path.stat().st_size/1024:.0f} KB)")
    print(f"   Level: {level} | Avg risk: {avg_risk:.1%} | Max FWI: {max_fwi}")
    print(f"   Fires: {len(fires)} | Alerts: {len(alerts)} | AQ points: {len(aq)}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 56)
    print("  WildfireIQ Pipeline")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 56)

    # Train ML model
    model, scaler = train_model()
    model_meta    = export_model_json(model, scaler)

    # Fetch live data (parallel)
    pts          = build_grid()
    weather_data = fetch_all_weather(pts)

    print("Running ML inference on grid…")
    grid   = build_risk_grid(pts, weather_data, model, scaler)

    print("Fetching active fires (NASA FIRMS)…")
    fires  = fetch_firms_fires()

    print("Fetching NOAA red flag warnings…")
    alerts = fetch_noaa_alerts()

    print("Fetching air quality (Open-Meteo AQ)…")
    aq     = fetch_air_quality_sample()

    write_outputs(grid, fires, alerts, aq, model_meta)

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
