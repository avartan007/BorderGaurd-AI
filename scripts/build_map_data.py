"""Build data/india_states.geojson.gz — simplified India state boundaries.

Source: udit-001/india-maps-data, per-state district GeoJSON (modern boundaries:
includes Telangana and Ladakh). Each state file is downloaded, its district
polygons simplified, and everything merged into one feature per state.

Two earlier sources were rejected:
  * udit-001 india.geojson  — only ~32 vertices per ring, which is why the first
    map rendered as blocky irregular polygons; no simplification could fix it.
  * geohacker/india         — good detail but pre-2014, so Telangana was missing
    and that state's threat level would never have appeared.

Run once (needs network, plus shapely for the dissolve):
    pip install shapely
    python scripts/build_map_data.py

shapely is a build-time dependency only — the runtime app just reads the gzipped
GeoJSON, so it is deliberately not in requirements.txt.
"""
from __future__ import annotations

import gzip
import json
import math
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/states"
STATE_FILES = {
    "andaman-and-nicobar-islands": "Andaman and Nicobar Islands",
    "andhra-pradesh": "Andhra Pradesh",
    "arunachal-pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "delhi": "Delhi",
    "dnh-and-dd": "Dadra and Nagar Haveli and Daman and Diu",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal-pradesh": "Himachal Pradesh",
    "jammu-and-kashmir": "Jammu and Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhya-pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "puducherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil-nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar-pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west-bengal": "West Bengal",
}

OUT = Path(__file__).resolve().parent.parent / "data" / "india_states.geojson.gz"

# Simplified boundaries are merged into ONE outline per state so the rendered map
# shows solid regions instead of a patchwork of districts. Set DISSOLVE=True to
# merge (recommended for the dashboard); False keeps district polygons.
DISSOLVE = True

# ~550 m at the equator. Keeps coastlines and the NE states readable while cutting
# the payload substantially.
TOLERANCE = 0.004
MIN_SPAN = 0.02   # bounding-box span in degrees; drops tiny offshore islets
MIN_AREA = 1e-4   # shoelace area in square degrees

# After merging, drop parts smaller than this. At national zoom the Sundarbans
# delta fragments and small islets are sub-pixel noise; 0.01 deg^2 is roughly
# 100 km^2, which keeps every state's mainland and any genuinely large island.
DISSOLVE_MIN_AREA = 0.01


def _ring_area(ring: list[list[float]]) -> float:
    area = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def _point_line_dist(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def simplify(ring: list[list[float]], tol: float) -> list[list[float]]:
    """Iterative Ramer-Douglas-Peucker (kept iterative to bound recursion depth)."""
    if len(ring) < 4:
        return ring
    keep = [False] * len(ring)
    keep[0] = keep[-1] = True
    stack = [(0, len(ring) - 1)]
    while stack:
        first, last = stack.pop()
        best, idx = 0.0, -1
        for i in range(first + 1, last):
            d = _point_line_dist(ring[i], ring[first], ring[last])
            if d > best:
                best, idx = d, i
        if idx != -1 and best > tol:
            keep[idx] = True
            stack.append((first, idx))
            stack.append((idx, last))
    return [p for p, k in zip(ring, keep) if k]


def _round(ring: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for pt in ring:
        rounded = [round(pt[0], 4), round(pt[1], 4)]
        if not out or out[-1] != rounded:
            out.append(rounded)
    if out and out[0] != out[-1]:
        out.append(out[0])
    return out


def _clean_polygon(poly: list[list[list[float]]]) -> list[list[list[float]]]:
    rings = []
    for ring in poly:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        if max(xs) - min(xs) < MIN_SPAN and max(ys) - min(ys) < MIN_SPAN:
            continue
        if _ring_area(ring) < MIN_AREA:
            continue
        simplified = _round(simplify(ring, TOLERANCE))
        if len(simplified) >= 4:
            rings.append(simplified)
    return rings


def _make_valid(geom):
    """Repair a self-intersecting polygon.

    The upstream district data contains rings that cross themselves (a
    TopologyException at ~93.10E 12.24N is one of them). make_valid is available
    in shapely 2.x; buffer(0) is the fallback for older builds.
    """
    try:
        from shapely.validation import make_valid

        fixed = make_valid(geom)
    except Exception:
        fixed = geom.buffer(0)
    if fixed.geom_type == "MultiPolygon":
        parts = list(fixed.geoms)
        if parts:
            return max(parts, key=lambda g: g.area)
    if fixed.geom_type == "GeometryCollection":
        polys = [g for g in fixed.geoms if g.geom_type == "Polygon" and g.area > 0]
        if polys:
            return max(polys, key=lambda g: g.area)
    return fixed


def _dissolve(polys: list[list[list[list[float]]]]) -> list[list[list[list[float]]]]:
    """Merge a state's district polygons into one outline.

    District polygons share edges with their neighbours. Drawn as-is they leave
    visible seams and each ring gets its own border, which is what made the map
    look like a patchwork. unary_union cancels the shared interior edges so the
    state renders as a single solid region with only its outer boundary stroked.
    """
    if not DISSOLVE:
        return polys
    try:
        from shapely.geometry import MultiPolygon, shape
        from shapely.ops import unary_union
    except ImportError:  # build-time only dependency; fall back gracefully
        print("  ! shapely unavailable, keeping district polygons")
        return polys

    geom = unary_union(
        [
            fixed
            for poly in polys
            if not (geom_poly := shape({"type": "Polygon", "coordinates": poly})).is_empty
            and not (fixed := _make_valid(geom_poly)).is_empty
        ]
    )
    if geom.is_empty:
        return polys

    parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    out: list[list[list[list[float]]]] = []
    for part in parts:
        # Drop slivers produced by the merge, but keep real islands.
        if part.area < DISSOLVE_MIN_AREA:
            continue
        minx, miny, maxx, maxy = part.bounds
        if (maxx - minx) < MIN_SPAN and (maxy - miny) < MIN_SPAN:
            continue
        rings = [_round([[x, y] for x, y in part.exterior.coords])]
        # Interior rings are only valid if the hole is genuinely empty; district
        # merges almost never produce real holes, so they are skipped on purpose.
        if len(rings[0]) >= 4:
            out.append(rings)
    return out or polys


def main() -> None:
    merged: dict[str, list[list[list[list[float]]]]] = {}

    for slug, display in sorted(STATE_FILES.items(), key=lambda kv: kv[1]):
        url = f"{BASE}/{slug}.geojson"
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                data = json.load(response)
        except Exception as exc:  # a single missing state must not kill the build
            print(f"  ! {display}: {exc}")
            continue

        cleaned_any = False
        for feature in data.get("features", []):
            geom = feature.get("geometry") or {}
            gtype = geom.get("type")
            if gtype == "Polygon":
                polygons = [geom["coordinates"]]
            elif gtype == "MultiPolygon":
                polygons = geom["coordinates"]
            else:
                continue
            cleaned = [c for poly in polygons if (c := _clean_polygon(poly))]
            if cleaned:
                merged.setdefault(display, []).extend(cleaned)
                cleaned_any = True
        before = len(merged.get(display, []))
        merged[display] = _dissolve(merged.get(display, []))
        print(f"  {'ok' if cleaned_any else 'EMPTY'} {display:42} {before}->{len(merged[display])} parts")

    features = [
        {
            "type": "Feature",
            "properties": {"st_nm": name},
            "geometry": {"type": "MultiPolygon", "coordinates": polys},
        }
        for name, polys in sorted(merged.items())
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"))
    with gzip.open(OUT, "wt", encoding="utf-8") as handle:
        handle.write(payload)

    vertices = sum(
        len(ring) for f in features for poly in f["geometry"]["coordinates"] for ring in poly
    )
    rings = sum(len(poly) for f in features for poly in f["geometry"]["coordinates"])
    print(
        f"\nstates={len(features)} raw={len(payload)/1e6:.2f}MB "
        f"gz={OUT.stat().st_size/1e6:.2f}MB vertices={vertices} avg/ring={vertices/max(rings,1):.0f}"
    )


if __name__ == "__main__":
    main()
