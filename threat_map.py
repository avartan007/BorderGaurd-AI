"""India threat map: CARTO basemap + per-state threat regions.

The original dashboard drew a hand-made SVG. This keeps the same seeded threat
picture (state, level, detection count, latest asset, date) but renders it over
real CARTO/OpenStreetMap tiles.

The basemap is fixed to CARTO **dark matter**; the app exposes no style switcher.
The threat geometry is dissolved to one outline per state (see
``scripts/build_map_data.py``) so each region fills as a single solid area rather
than a patchwork of districts.

CARTO basemap styles
--------------------
CARTO's public dark-matter style JSON is served without authentication, so a key is
*optional* for this app.

If you do have a CARTO key, note the CDN quirk: ``basemaps.cartocdn.com`` ignores
``?api_key=`` and only forwards ``?key=`` to the tile endpoint. ``build_style_url``
therefore handles both forms correctly. Set either:

  CARTO_API_KEY   -> the style URL gets the key appended (or substituted into a
                     ``{api_key}`` / ``{key}`` placeholder in CARTO_STYLE_URL)
  CARTO_STYLE_URL -> a fully custom style URL, used as-is
"""
from __future__ import annotations

import gzip
import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pydeck as pdk

DATA_FILE = Path(__file__).resolve().parent / "data" / "india_states.geojson.gz"

# Fixed dark basemap. The app is dark-only by design: there is no style switcher,
# and the threat fills are tuned to sit on this dark canvas.
DARK_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
DEFAULT_STYLE = DARK_STYLE

# Seeded threat picture from the original dashboard (state -> level, detections,
# latest asset, date). States not listed are Safe.
STATE_THREATS: dict[str, dict] = {
    "Jammu and Kashmir": {"level": "Critical", "detections": 14, "asset": "Military Tank", "date": "2025-01-10"},
    "Himachal Pradesh": {"level": "High", "detections": 7, "asset": "Military Aircraft", "date": "2025-01-08"},
    "Punjab": {"level": "High", "detections": 9, "asset": "Soldier", "date": "2025-01-09"},
    "Haryana": {"level": "Medium", "detections": 4, "asset": "Military Vehicle", "date": "2025-01-05"},
    "Delhi": {"level": "Medium", "detections": 3, "asset": "Camouflage Soldier", "date": "2025-01-07"},
    "Rajasthan": {"level": "Low", "detections": 2, "asset": "Civilian Vehicle", "date": "2025-01-03"},
    "Uttar Pradesh": {"level": "Medium", "detections": 5, "asset": "Weapon", "date": "2025-01-06"},
    "Madhya Pradesh": {"level": "Low", "detections": 1, "asset": "Military Truck", "date": "2024-12-28"},
    "Maharashtra": {"level": "Medium", "detections": 3, "asset": "Military Aircraft", "date": "2024-12-30"},
    "Bihar": {"level": "High", "detections": 6, "asset": "Military Tank", "date": "2025-01-07"},
    "West Bengal": {"level": "High", "detections": 8, "asset": "Military Warship", "date": "2025-01-09"},
    "Odisha": {"level": "Low", "detections": 2, "asset": "Soldier", "date": "2024-12-20"},
    "Telangana": {"level": "Low", "detections": 1, "asset": "Civilian", "date": "2024-12-15"},
    "Karnataka": {"level": "Low", "detections": 2, "asset": "Military Vehicle", "date": "2024-12-22"},
    "Assam": {"level": "Critical", "detections": 12, "asset": "Military Artillery", "date": "2025-01-10"},
    "Sikkim": {"level": "Medium", "detections": 3, "asset": "Soldier", "date": "2025-01-02"},
    "Manipur": {"level": "High", "detections": 5, "asset": "Camouflage Soldier", "date": "2025-01-05"},
    "Arunachal Pradesh": {"level": "Critical", "detections": 18, "asset": "Military Aircraft", "date": "2025-01-10"},
}

_LEVEL_RGBA = {
    "Critical": [255, 59, 48, 200],
    "High": [255, 149, 0, 185],
    "Medium": [255, 214, 10, 170],
    "Low": [48, 209, 88, 160],
    "Safe": [88, 130, 175, 90],
}

# Lower opacity for cooler levels so the basemap stays visible everywhere.
_LEVEL_OPACITY = {
    "Critical": 0.78,
    "High": 0.72,
    "Medium": 0.68,
    "Low": 0.62,
    "Safe": 0.38,
}
_LEVEL_HEX = {
    "Critical": "#ff3b30",
    "High": "#ff9500",
    "Medium": "#ffd60a",
    "Low": "#30d158",
    "Safe": "#2a4a6a",
}


@lru_cache(maxsize=1)
def load_states_geojson() -> dict:
    with gzip.open(DATA_FILE, "rt", encoding="utf-8") as fh:
        geo = json.load(fh)
    for feature in geo["features"]:
        name = feature["properties"].get("st_nm", "Unknown")
        threat = STATE_THREATS.get(name)
        feature["properties"].update(
            {
                "state": name,
                "level": threat["level"] if threat else "Safe",
                "detections": threat["detections"] if threat else 0,
                "asset": threat["asset"] if threat else "—",
                "date": threat["date"] if threat else "—",
            }
        )
    return geo


def build_style_url() -> str:
    """Resolve the map style URL, applying CARTO_API_KEY when one is configured.

    Never raises: a bad/missing key must not take the map down, so we always fall
    back to the keyless public style.
    """
    base = (os.getenv("CARTO_STYLE_URL", "").strip() or DEFAULT_STYLE)
    key = os.getenv("CARTO_API_KEY", "").strip()
    if not key:
        return base

    # Placeholder forms: {api_key} or {key}
    if "{api_key}" in base or "{key}" in base:
        return base.replace("{api_key}", key).replace("{key}", key)

    # Otherwise append. The CARTO CDN forwards 'key', not 'api_key'.
    parts = urlsplit(base)
    query = dict(parse_qsl(parts.query))
    query.setdefault("key", key)
    return urlunsplit(parts._replace(query=urlencode(query)))


def build_threat_deck(height: int = 520, style_url: str | None = None) -> pdk.Deck:
    geo = load_states_geojson()
    features = geo["features"]

    # One layer per threat level so each region gets a flat, constant colour.
    # The geometry has already been dissolved to one outline per state, so there
    # are no district seams and no double-drawn internal borders.
    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            id=f"states-{level.lower()}",
            data={
                "type": "FeatureCollection",
                "features": [f for f in features if f["properties"]["level"] == level],
            },
            filled=True,
            stroked=True,
            extruded=False,
            pickable=True,
            auto_highlight=True,
            highlight_color=[0, 212, 255, 70],
            get_fill_color=_LEVEL_RGBA[level],
            # Only the region outline is drawn, in a dark tone matching the
            # basemap, so states read as solid blocks rather than outlines.
            get_line_color=[10, 20, 32, 180],
            line_width_min_pixels=1,
            opacity=_LEVEL_OPACITY[level],
        )
        for level in ["Safe", "Low", "Medium", "High", "Critical"]
    ]

    return pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(
            latitude=22.8,
            longitude=80.6,
            zoom=3.5,
            pitch=0,
            bearing=0,
            max_zoom=9,
            min_zoom=2.5,
        ),
        map_style=style_url or build_style_url(),
        tooltip={
            "html": (
                "<div style='font-family:Segoe UI,sans-serif'>"
                "<b style='color:#00d4ff;font-size:13px'>{state}</b><br/>"
                "<span style='color:#c9d8e8'>Threat Level: <b>{level}</b><br/>"
                "Detections: {detections}<br/>"
                "Latest Asset: {asset}<br/>"
                "Date: {date}</span></div>"
            ),
            "style": {
                "backgroundColor": "#0b1623",
                "color": "#c9d8e8",
                "border": "1px solid #00d4ff",
                "borderRadius": "8px",
                "padding": "10px 14px",
            },
        },
    )


def legend_html() -> str:
    items = "".join(
        f"<span style='display:inline-flex;align-items:center;gap:6px;margin-right:14px;"
        f"font-size:12px;color:#5a7a9a'>"
        f"<i style='width:10px;height:10px;border-radius:50%;background:{_LEVEL_HEX[lvl]};"
        f"display:inline-block'></i>{lvl}</span>"
        for lvl in ["Critical", "High", "Medium", "Low", "Safe"]
    )
    return f"<div style='padding:6px 2px 0'>{items}</div>"
