# 🔥 WildfireIQ

Live wildfire risk intelligence. No signups. No downloads. No API keys.

**Live at:** `https://YOURUSERNAME.github.io/wildfireiq`

## Data Sources (all free, no keys)
| Source | What it provides |
|--------|-----------------|
| [Open-Meteo](https://open-meteo.com) | Live weather + 7-day forecast |
| [Open-Meteo AQ](https://air-quality-api.open-meteo.com) | AQI + PM2.5 |
| [NASA EONET](https://eonet.gsfc.nasa.gov) | Active fire events |
| [NOAA](https://api.weather.gov) | Red Flag Warnings |
| [Nominatim](https://nominatim.org) | Location search |

## Technology
- Neural network (feedforward, 7→4→1) runs entirely in the browser
- Real ML inference on live weather features
- AI attribution explains which weather drivers are causing risk
- No server, no backend, no database

## Deploy

```bash
cd wildfireiq
git init
git add .
git commit -m "🔥 WildfireIQ"
git branch -M main
git remote add origin https://github.com/YOURUSERNAME/wildfireiq.git
git push -u origin main
```

Then: **Settings → Pages → Source → GitHub Actions**
