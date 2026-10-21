# 🛡️ BorderGuard AI

**Defence Threat Detection & Military Asset Analysis**

YOLOv8 detects military assets in images/videos, a rule-based threat engine scores
them, Gemini writes the intelligence assessment, and the dashboard maps threat
regions over a real CARTO/OpenStreetMap basemap.

This repo now has **two frontends**:

| File | Stack | Use |
|---|---|---|
| `streamlit_app.py` | Streamlit | **Deployment target** (Streamlit Community Cloud) |
| `app.py` | FastAPI + HTML/JS | Original dashboard, kept for reference / self-hosting |

---

## Why Streamlit Cloud (and not Render's free tier)

Measured on this project: loading `best.pt` (50 MB) and running one image peaks at
**~495 MB RSS** before Streamlit, pandas, Altair and pydeck are even counted. On
Linux that typically lands at **700 MB–1 GB**. Render's free instance is 512 MB and
would OOM on model load. Streamlit Community Cloud gives ~2.7 GB, so it is the
right host for this workload.

`requirements.txt` pulls CPU-only torch/torchvision wheels (`+cpu`) so the build is
~200 MB instead of the ~2.5 GB CUDA build.

`packages.txt` (repo root — Streamlit only reads it there, not in `.streamlit/`)
apt-installs the graphics libraries OpenCV needs. This is required, not optional:
`ultralytics` hard-requires `opencv-python` (the GUI build), so pip installs it
alongside the `opencv-python-headless` pin and its `cv2` wins. Without those
libraries the build succeeds and the app then dies at startup on `import cv2`
with `ImportError: libxcb.so.1` / `libGL.so.1`.

---

## Deploy to Streamlit Community Cloud

1. **Rotate the Gemini key first.** The old key was hardcoded in `app.py`; treat it
   as compromised. Create a new one at <https://aistudio.google.com/apikey>. The
   hardcoded key has been removed from the code.

2. **Push this folder to GitHub** (public or private).

   ```bash
   git init
   git add .
   git commit -m "Streamlit deployment"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

   `best.pt` is 50 MB, under GitHub's 100 MB file limit, so no LFS needed.
   `.streamlit/secrets.toml` is gitignored — never commit it.

3. Go to <https://share.streamlit.io> → **New app** → pick the repo/branch.

4. Set **Main file path** to `streamlit_app.py`.

5. **Python version:** leave the default. Streamlit Cloud currently provisions
   **Python 3.14**, and the pinned CPU wheels (`torch==2.10.0+cpu`,
   `torchvision==0.25.0+cpu`) are the oldest pair that supports it. The earlier
   pin (`torch 2.6` / `torchvision 0.21`) is capped at Python ≤3.12 and fails to
   resolve on 3.14, which is what caused the first deploy build to error out.

6. In **Advanced settings → Secrets**, paste:

   ```toml
   GEMINI_API_KEY = "your-new-google-ai-studio-key"
   ```

   CARTO is optional — see [Map](#map) below.

7. **Deploy.** First build takes a few minutes (torch + ultralytics). The model is
   loaded once per app instance and cached with `st.cache_resource`.

---

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then paste your GEMINI_API_KEY into .env
streamlit run streamlit_app.py
```

Either configuration style works — pick one:

| | File | Format |
|---|---|---|
| Environment | `.env` (copied from `.env.example`) | `KEY=value` |
| Streamlit secrets | `.streamlit/secrets.toml` (from `.streamlit/secrets.toml.example`) | TOML |

The app reads **Streamlit secrets first, then the environment**, and `.env` is
loaded automatically (no extra dependency). Real environment variables always win,
so nothing is overwritten.

The app opens on <http://localhost:8501>.

Recognised variables: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `CARTO_API_KEY`,
`CARTO_STYLE_URL`, `MODEL_PATH`.

---

## Map

The hand-drawn India SVG was replaced with a real map:

- **Basemap:** CARTO **dark matter**, built on OpenStreetMap data. The map is
  **dark-only** — the Dashboard switcher was removed, so there is nothing to
  configure and no light/voyager/positron options. Set `CARTO_STYLE_URL` if you
  ever want to override it.
- **Threat regions:** `data/india_states.geojson.gz` — 36 states/UTs, one
  **dissolved outline per state**, ~24k vertices (150 KB gzipped). Built from
  modern district boundaries that are merged at build time, so a state renders as
  a single solid region instead of a patchwork of districts with visible seams.
  31 of the 36 states are one polygon; the exceptions are genuine (island chains,
  West Bengal's Siliguri corridor which is separated by Bangladesh, Puducherry's
  scattered enclaves).
- **Colouring:** each state is filled flat by its seeded threat level
  (red/orange/yellow/green/blue), with the same tooltip fields as before
  (level, detections, latest asset, date). The outline is drawn dark-on-dark so
  the colour reads as one area rather than an outline.
- **Fills are semi-transparent** (0.38 for Safe up to 0.78 for Critical) so the
  basemap and its labels stay readable under the colours.
- **Legend and threat table** are rendered below the map.

### Do you need a CARTO key?

**No.** CARTO's public basemap styles are served without authentication — this is
verified, `basemaps.cartocdn.com` returns the style JSON with HTTP 200 and no key.
Add a key only if you want CARTO's keyed terms or higher tile limits
(<https://carto.com/> → Developers → API keys).

### Configuring the key

Set it in `.env`, or in Streamlit secrets (same names, TOML syntax):

```toml
CARTO_API_KEY = "your-carto-key"
```

That alone is enough — the key is appended to the dark style. To pin a specific
style, add `CARTO_STYLE_URL`:

```toml
# a placeholder, filled from CARTO_API_KEY
CARTO_STYLE_URL = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json?api_key={api_key}"
```

Resolution rules (implemented in `threat_map.build_style_url`):

| `CARTO_API_KEY` | `CARTO_STYLE_URL` | Result |
|---|---|---|
| unset | unset | default dark-matter style, no key |
| set | unset | default style + `?key=<key>` |
| set | plain URL | that URL + `?key=<key>` |
| set | URL with `{api_key}`/`{key}` | placeholder replaced |
| unset | any URL | used as-is |

> **Gotcha:** CARTO's CDN ignores `?api_key=` and only forwards `?key=` to the tile
> endpoint, so a `{api_key}` URL works for style loading but carries no
> authentication. Appending as `?key=` is the form that actually authenticates.
> `build_style_url` handles both, and can never fail — a bad key falls back to the
> keyless default.

The **Settings** page shows the resolved style URL (key masked as `***`) so you can
confirm what the map is actually using.

To rebuild the map data:

```bash
python scripts/build_map_data.py
```

The build script merges 36 per-state files. If a single state fails to download it
reports `!` and continues rather than producing a partial map.

**Full restoration path** if the map ever looks wrong again:

1. `data/india_states.geojson.gz` defines the shapes. Check the detail level:
   ```bash
   python -c "import gzip,json;g=json.load(gzip.open('data/india_states.geojson.gz','rt'));\
   v=sum(len(r) for f in g['features'] for p in f['geometry']['coordinates'] for r in p);\
   print(len(g['features']),'states',v,'vertices')"
   ```
   Expect ~36 states and ~24k vertices (dissolved, one outline per state). If you see
   ~95k vertices the file is the old district-level build, and if you see ~23k the
   original coarse source was used — rerun the build script.
2. The basemap is a browser-side fetch by deck.gl. If the polygons appear on a black
   background, the tile requests are failing: check the browser devtools network tab
   for `basemaps.cartocdn.com`. The map is dark-only by design, so a black canvas
   with visible region colours is the expected result.
3. `threat_map.build_style_url()` controls which style loads. The Settings page prints
   the resolved URL with the key masked.

To rebuild the map data (needs network + `pip install shapely`):

```bash
python scripts/build_map_data.py
```

---

## Dashboard layout

The dashboard is a command-centre layout, not a flat card grid:

1. **Page header** — title, subtitle and a right-aligned meta line.
2. **Alert banner** — full-width posture strip whose colour follows the threat level,
   naming the elevated regions and showing the high-risk count.
3. **Hero + KPI strip** — the largest element is the *Current Threat Posture* score
   (0–100, derived from how many monitored regions are elevated), beside two rows of
   four compact KPI pills.
4. **Regional threat map** — full width, dark CARTO basemap, no style switcher, so the
   map gets the whole row.
5. **Bottom row** — Recent Activity, Posture Breakdown (per-level bars) and System
   info side by side.
6. **Monitored Regions table** — sortable, the same records as the map tooltips.

Reusable pieces live in `streamlit_app.py`: `page_header()`, `alert_banner()`,
`hero_panel()` and `stat_card()`. Design tokens (colours, spacing, radii) are CSS
custom properties at the top of the injected stylesheet, so the look can be retuned
in one place.

## Features

- **Dashboard** — live counters, CARTO threat map, activity feed, threat table.
- **Threat Detection** — image + video upload, YOLO inference, annotated evidence,
  per-object confidence/threat table, threat badge and 0–100 score.
- **Video** — samples every 8th frame (max 40) and encodes a browser-playable H.264
  annotated clip with `imageio-ffmpeg`, with a frame-image fallback.
- **Reports** — auto-created per detection; downloadable PDF with evidence image,
  risk assessment and object analysis.
- **History** — search/filter, inspect, delete, clear.
- **AI Chat** — Gemini 2.5 Flash grounded in a detection session; deterministic
  fallback when the API key is missing or the call fails.
- **Analytics** — asset bars, threat-level donut, 7-day line chart.
- **Settings** — config status, model classes, session data reset.

Threat scoring lives in `threat_core.py`; the level comes from the highest-priority
object (`weapon`/tank/aircraft/artillery/warship → Critical), and the numeric score
is weighted by object counts with a floor that matches the level.

---

## Files

```
streamlit_app.py          Streamlit UI + pages
threat_core.py            inference, threat scoring, PDF, Gemini
threat_map.py             CARTO deck.gl map + seeded state threats
data/india_states.geojson.gz
scripts/build_map_data.py regenerates the map data
scripts/smoke_test.py     detection + PDF + scoring smoke test
.streamlit/config.toml    dark theme, 200 MB upload limit
.streamlit/secrets.toml.example
.env.example              same settings as KEY=value
requirements.txt          minimal Streamlit Cloud deps (UTF-8)
packages.txt              apt graphics libs OpenCV needs (read from repo root)
requirements.lock.txt     original pip freeze, reference only
app.py                    original FastAPI app (kept)
best.pt                   YOLOv8 model (12 classes)
static/                   artefacts from the FastAPI version (unused)
```

Verify the model pipeline locally:

```bash
python scripts/smoke_test.py            # uses static/*.png as fixtures
python scripts/smoke_test.py path/to.jpg
```

---

## Notes and limits

- **Session data is per browser session** (Streamlit `session_state`), not a shared
  database. Restarting the app or opening a new session starts empty. Download PDFs
  you need to keep.
- **CPU-only inference.** Images are resized to ≤1600 px. Video is sampled, not
  analysed frame-by-frame, to stay responsive on shared CPUs.
- **Streamlit Community Cloud sleeps** idle apps; the first request after sleep
  reloads the model (~5–10 s).
- The seeded state threat data is demo data carried over from the original SVG, not
  a live feed.
- The old FastAPI key is gone from the source, but if that key ever reached a Git
  remote before this change, revoke it in Google AI Studio.
# BorderGaurd-AI
