"""BorderGuard AI — Defence Threat Detection & Military Asset Analysis.

Streamlit entrypoint. Run locally with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import os
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

import threat_core as core
import threat_map
from threat_map import build_threat_deck, legend_html, load_states_geojson, STATE_THREATS

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", APP_DIR / "best.pt"))

PAGES = [
    "📊 Dashboard",
    "🎯 Threat Detection",
    "📋 Reports",
    "🕑 Detection History",
    "🤖 AI Intelligence Chat",
    "📈 Analytics",
    "⚙️ Settings",
]

# ── Secrets / config ──────────────────────────────────────────────────────────
def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so local runs work without extra dependencies.

    Existing environment variables always win, so real env vars (and Streamlit
    Cloud secrets pushed into the environment) are never overwritten.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


# Load .env before anything reads configuration.
_load_dotenv(APP_DIR / ".env")


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then the environment (.env included)."""
    try:
        value = st.secrets.get(name, None)  # type: ignore[union-attr]
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


GEMINI_API_KEY = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")

# threat_map reads these from the environment.
for _env_name in ("CARTO_API_KEY", "CARTO_STYLE_URL", "MODEL_PATH"):
    _value = get_secret(_env_name)
    if _value:
        os.environ.setdefault(_env_name, _value)

st.set_page_config(
    page_title="BorderGuard AI — Defence Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "BorderGuard AI"
APP_TAGLINE = "Defence Threat Detection & Military Asset Analysis"

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  /* ── Design tokens ───────────────────────────────────────────── */
  :root{
    --bg:#060d17; --panel:#0e1c2c; --panel-2:#0b1624; --border:#19304e;
    --accent:#00d4ff; --red:#ff3b30; --orange:#ff9500; --green:#30d158;
    --yellow:#ffd60a; --purple:#bf5af2; --text:#eaf6ff; --dim:#5a7a9a;
    --gap:16px;
  }
  /* ── Shell ───────────────────────────────────────────────────── */
  .block-container{padding-top:1.1rem; padding-bottom:2.4rem; max-width:1560px;}
  #MainMenu, footer{visibility:hidden;}
  h1,h2,h3,h4{letter-spacing:.2px; color:var(--text);}
  hr{border-color:var(--border); margin:.6rem 0;}

  /* ── Brand ───────────────────────────────────────────────────── */
  .bg-brand{display:flex;align-items:center;gap:11px;padding:4px 0 14px}
  .bg-brand .mark{width:42px;height:42px;border-radius:12px;flex-shrink:0;
    background:linear-gradient(135deg,#00a8cc,#0066cc);display:flex;align-items:center;
    justify-content:center;font-size:21px;box-shadow:0 6px 20px rgba(0,212,255,.30)}
  .bg-brand .name{font-size:16px;font-weight:800;color:#fff;letter-spacing:1.5px;line-height:1.1}
  .bg-brand .sub{font-size:9.5px;color:var(--dim);letter-spacing:1.5px;text-transform:uppercase;margin-top:3px}

  /* ── Page header ─────────────────────────────────────────────── */
  .pg-head{display:flex;align-items:flex-end;justify-content:space-between;
    gap:16px;padding:0 0 14px;border-bottom:1px solid var(--border);margin-bottom:var(--gap)}
  .pg-head .t{font-size:23px;font-weight:800;color:#fff;letter-spacing:.3px;line-height:1.15}
  .pg-head .s{font-size:12.5px;color:var(--dim);margin-top:4px}
  .pg-head .meta{font-size:11px;color:var(--dim);text-align:right;white-space:nowrap}
  .pg-head .meta b{color:var(--accent);font-weight:700}

  /* ── Alert banner ────────────────────────────────────────────── */
  .alert-bar{display:flex;align-items:center;gap:13px;border-radius:12px;
    padding:13px 18px;margin-bottom:var(--gap);border:1px solid var(--bc);
    background:linear-gradient(90deg,var(--bbg),rgba(11,22,36,.25));position:relative;overflow:hidden}
  .alert-bar::before{content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--bc)}
  .alert-bar .ico{font-size:22px;line-height:1}
  .alert-bar .h{font-size:14px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;color:var(--bc)}
  .alert-bar .p{font-size:12.5px;color:var(--dim);margin-top:2px}
  .alert-bar .st{margin-left:auto;text-align:right;font-size:11px;color:var(--dim);white-space:nowrap}
  .alert-bar .st b{display:block;font-size:19px;color:var(--text);font-weight:800}

  /* ── Hero threat panel ───────────────────────────────────────── */
  .hero{border:1px solid var(--border);border-radius:14px;padding:18px 20px;
    background:linear-gradient(160deg,#102134,#0c1a29 60%,#0b1624);height:100%}
  .hero .lbl{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:1.6px}
  .hero .big{font-size:44px;font-weight:850;line-height:1.05;margin:6px 0 2px;color:var(--hc)}
  .hero .of{font-size:13px;color:var(--dim);font-weight:600}
  .hero .bar{height:9px;border-radius:6px;background:#0a1928;margin:14px 0 8px;overflow:hidden}
  .hero .fill{height:9px;border-radius:6px;background:var(--hc);transition:width .5s}
  .hero .foot{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);margin-top:10px}

  /* ── Compact stat pills ──────────────────────────────────────── */
  .pill{background:var(--panel-2);border:1px solid var(--border);border-radius:11px;
    padding:11px 13px;position:relative;overflow:hidden;height:74px}
  .pill::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--pc)}
  .pill .v{font-size:22px;font-weight:800;color:#fff;line-height:1.1}
  .pill .l{font-size:9.5px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-top:3px}
  .pill .i{position:absolute;right:10px;top:9px;font-size:15px;opacity:.55}

  /* ── Panels ──────────────────────────────────────────────────── */
  .panel{background:var(--panel-2);border:1px solid var(--border);border-radius:13px;padding:15px 17px}
  .panel-title{font-size:11px;font-weight:800;color:var(--accent);text-transform:uppercase;
    letter-spacing:1.6px;margin:0 0 10px;display:flex;align-items:center;gap:7px}
  .panel-sub{font-size:11px;color:var(--dim);font-weight:500;letter-spacing:0;text-transform:none;margin-left:auto}

  /* ── Feed ────────────────────────────────────────────────────── */
  .feed-item{display:flex;gap:10px;align-items:flex-start;padding:8px 0;
    border-bottom:1px solid rgba(25,48,78,.5)}
  .feed-item:last-child{border-bottom:none}
  .feed-ico{width:28px;height:28px;border-radius:8px;background:rgba(0,212,255,.10);
    display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
  .feed-title{font-size:12px;color:var(--text);font-weight:600;line-height:1.35}
  .feed-time{font-size:10px;color:var(--dim);margin-top:2px}

  /* ── Info rows ───────────────────────────────────────────────── */
  .info-row{display:flex;justify-content:space-between;gap:10px;padding:6px 0;
    border-bottom:1px solid rgba(25,48,78,.45);font-size:12px}
  .info-row:last-child{border-bottom:none}
  .info-row .k{color:var(--dim)}
  .info-row .v{color:var(--text);font-weight:600;text-align:right}

  /* ── Badges / tags ───────────────────────────────────────────── */
  .badge{display:inline-block;padding:4px 14px;border-radius:20px;font-size:11.5px;font-weight:800;
    letter-spacing:.8px;text-transform:uppercase}
  .badge-Critical{background:rgba(255,59,48,.16);color:var(--red);border:1px solid var(--red)}
  .badge-High{background:rgba(255,149,0,.16);color:var(--orange);border:1px solid var(--orange)}
  .badge-Medium{background:rgba(255,214,10,.14);color:var(--yellow);border:1px solid var(--yellow)}
  .badge-Low{background:rgba(48,209,88,.14);color:var(--green);border:1px solid var(--green)}
  .badge-Safe{background:rgba(48,209,88,.12);color:var(--green);border:1px solid var(--green)}
  .obj-tag{display:inline-block;background:rgba(0,212,255,.10);border:1px solid rgba(0,212,255,.3);
    color:var(--accent);padding:3px 11px;border-radius:20px;font-size:11.5px;margin:2px 4px 2px 0}
  .pill-ok{color:var(--green);font-weight:700}
  .pill-warn{color:var(--orange);font-weight:700}
  .small-note{font-size:11px;color:var(--dim)}

  /* ── Streamlit widget polish ─────────────────────────────────── */
  div[data-testid="stSidebar"]{background:#091320;border-right:1px solid var(--border)}
  div[data-testid="stSidebar"] .block-container{padding-top:1.1rem;padding-left:1rem;padding-right:1rem}
  .nav-label{font-size:9.5px;color:var(--dim);letter-spacing:1.6px;text-transform:uppercase;
    margin:2px 0 8px}

  /* ── Sidebar nav tiles (stacked one per row) ─────────────────── */
  div[data-testid="stSidebar"] div[data-testid="stButton"]{margin:0 0 8px}
  div[data-testid="stSidebar"] div[data-testid="stButton"]:last-of-type{margin-bottom:0}
  div[data-testid="stSidebar"] div[data-testid="stButton"] > button{
    width:100%; height:46px; justify-content:flex-start; text-align:left;
    padding:0 14px; border-radius:11px; font-size:13.5px; font-weight:650;
    background:#0d1b2b; border:1px solid var(--border); color:#9fb6cd;
    box-shadow:none; transition:all .16s ease; white-space:nowrap; overflow:hidden;
  }
  div[data-testid="stSidebar"] div[data-testid="stButton"] > button p{
    font-size:13.5px; font-weight:650; margin:0; overflow:hidden; text-overflow:ellipsis;
  }
  div[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover{
    background:#12253a; border-color:var(--accent); color:#eaf6ff; transform:translateY(-1px);
  }
  /* Active tile. Streamlit sets data-testid="stBaseButton-<kind>" on the button,
     so the active item (rendered with type="primary") is targeted on that. */
  div[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]{
    background:linear-gradient(135deg,rgba(0,212,255,.20),rgba(0,102,204,.16))!important;
    border:1px solid var(--accent)!important; color:#ffffff!important;
    box-shadow:0 4px 16px rgba(0,212,255,.18), inset 3px 0 0 var(--accent)!important;
  }
  div[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"]:hover{
    background:linear-gradient(135deg,rgba(0,212,255,.30),rgba(0,102,204,.24))!important;
  }
  div[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] p{
    color:#fff!important; font-weight:750!important;
  }

  .stButton>button, .stDownloadButton>button{border-radius:9px;font-weight:650}
  div[data-testid="stMetricValue"]{font-size:24px}
  div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:13px}
  .stDataFrame{border-radius:10px;overflow:hidden}
  /* tighten vertical rhythm of stacked blocks */
  div[data-testid="stVerticalBlock"] > div[style*="gap"]{gap:.85rem}
</style>
""",
    unsafe_allow_html=True,
)


# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLO defence model…")
def get_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path).to("cpu")


@st.cache_resource(show_spinner=False)
def get_class_names(model_path: str) -> list[str]:
    model = get_model(model_path)
    return [model.names[i] for i in sorted(model.names)]


# ── Session state ─────────────────────────────────────────────────────────────
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("nav", PAGES[0])
    ss.setdefault("history", [])          # detection records
    ss.setdefault("reports", [])          # report records (auto-created per detection)
    ss.setdefault("chats", {})            # detection id -> [{role, content}]
    ss.setdefault("activity", [])
    ss.setdefault("last_result_id", None)
    ss.setdefault("pdf_cache", {})


def log_activity(icon: str, title: str) -> None:
    st.session_state.activity.insert(
        0, {"icon": icon, "title": title, "time": datetime.now().strftime("%d %b %H:%M:%S")}
    )
    del st.session_state.activity[8:]


def stats_snapshot() -> dict:
    hist = st.session_state.history
    return {
        "images": sum(1 for h in hist if h["type"] == "image"),
        "videos": sum(1 for h in hist if h["type"] == "video"),
        "threats": sum(1 for h in hist if h["classes"]),
        "high_risk": sum(1 for h in hist if h["threat_level"] in ("High", "Critical")),
        "reports": len(st.session_state.reports),
    }


def find_record(record_id: str | None) -> dict | None:
    if not record_id:
        return None
    for rec in st.session_state.history:
        if rec["id"] == record_id:
            return rec
    return None


def stat_card(col, icon: str, value, label: str, color: str) -> None:
    """Compact KPI pill."""
    col.markdown(
        f"""<div class="pill" style="--pc:{color}">
              <div class="i">{icon}</div>
              <div class="v">{value}</div>
              <div class="l">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, meta: str = "") -> None:
    """Consistent page title block with an optional right-aligned meta line."""
    meta_html = f"<div class='meta'>{meta}</div>" if meta else ""
    st.markdown(
        f"""<div class="pg-head">
              <div><div class="t">{title}</div><div class="s">{subtitle}</div></div>
              {meta_html}
            </div>""",
        unsafe_allow_html=True,
    )


def alert_banner(level: str, headline: str, detail: str, stat_label: str = "", stat_value="") -> None:
    """Full-width posture banner. Colour follows the threat level."""
    color = core.threat_color(level)
    icon = {"Critical": "🚨", "High": "⚠️", "Medium": "🟡", "Low": "🟢", "Safe": "✅"}.get(level, "🛡️")
    stat = (
        f"<div class='st'><b>{stat_value}</b>{stat_label}</div>" if stat_label else ""
    )
    st.markdown(
        f"""<div class="alert-bar" style="--bc:{color};--bbg:{color}1f">
              <div class="ico">{icon}</div>
              <div><div class="h">{headline}</div><div class="p">{detail}</div></div>
              {stat}
            </div>""",
        unsafe_allow_html=True,
    )


def hero_panel(level: str, score: int, headline: str, foot_left: str, foot_right: str) -> None:
    """Large threat-score hero, used at the top of the dashboard."""
    color = core.threat_color(level)
    st.markdown(
        f"""<div class="hero" style="--hc:{color}">
              <div class="lbl">Current Threat Posture</div>
              <div class="big">{level}</div>
              <div class="of">{headline}</div>
              <div class="bar"><div class="fill" style="width:{max(score, 2)}%"></div></div>
              <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--dim)">
                <span>0</span><span>Threat score {score}/100</span><span>100</span>
              </div>
              <div class="foot"><span>{foot_left}</span><span>{foot_right}</span></div>
            </div>""",
        unsafe_allow_html=True,
    )


def badge(level: str) -> str:
    label = "✅ Safe — No Threat" if level == "Safe" else f"⚠ {level} Threat"
    return f'<span class="badge badge-{level}">{label}</span>'


def thumb_bytes(image: Image.Image, width: int = 360) -> bytes:
    img = image.copy()
    if img.width > width:
        img = img.resize((width, int(img.height * width / img.width)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=82)
    return buf.getvalue()


# ── Detection helpers ─────────────────────────────────────────────────────────
def save_detection(
    *,
    filename: str,
    kind: str,
    result: dict,
    annotated: Image.Image,
) -> dict:
    record = {
        "id": uuid.uuid4().hex,
        "filename": filename,
        "date": datetime.now(),
        "type": kind,
        "classes": result["classes"],
        "counts": result["counts"],
        "detections": result["detections"],
        "threat_level": result["threat_level"],
        "threat_score": result["threat_score"],
        "annotated_jpeg": thumb_bytes(annotated, width=1280),
        "thumbnail": thumb_bytes(annotated),
        "clip": result.get("clip"),
        "sampled_frames": result.get("sampled_frames", 0),
        "width": result.get("width", annotated.width),
        "height": result.get("height", annotated.height),
    }
    st.session_state.history.insert(0, record)
    st.session_state.last_result_id = record["id"]
    st.session_state.chats[record["id"]] = []
    st.session_state.reports.insert(
        0,
        {
            "id": record["id"],
            "filename": filename,
            "date": record["date"],
            "threat_level": record["threat_level"],
            "threat_score": record["threat_score"],
            "objects": len(record["classes"]),
        },
    )
    log_activity("🎯", f"{kind.title()} analysed — {len(record['classes'])} object type(s) in {filename}")
    return record


def annotated_image(record: dict) -> Image.Image:
    return Image.open(io.BytesIO(record["annotated_jpeg"])).convert("RGB")


def pdf_for(record: dict) -> bytes:
    cache = st.session_state.pdf_cache
    if record["id"] not in cache:
        cache[record["id"]] = core.build_pdf(
            {
                **record,
                "annotated": annotated_image(record) if record.get("annotated_jpeg") else None,
            }
        )
    return cache[record["id"]]


# ── Sidebar ───────────────────────────────────────────────────────────────────
init_state()

NAV_ICONS = {
    "📊 Dashboard": "📊",
    "🎯 Threat Detection": "🎯",
    "📋 Reports": "📋",
    "🕑 Detection History": "🕑",
    "🤖 AI Intelligence Chat": "🤖",
    "📈 Analytics": "📈",
    "⚙️ Settings": "⚙️",
}


def nav_tiles() -> str:
    """Render the navigation as clickable tiles and return the selected page.

    Tiles are stacked one per row in a single column, which reads more neatly in
    a narrow sidebar than a wrapped grid. Uses real buttons rather than st.radio
    so each entry is a proper tile instead of a radio dot, and so the active page
    can be highlighted.
    """
    current = st.session_state.get("nav", PAGES[0])
    chosen = current
    for page in PAGES:
        label = page.split(" ", 1)[1]
        icon = NAV_ICONS[page]
        active = page == current
        if st.button(
            f"{icon} {label}",
            key=f"navtile_{page}",
            width="stretch",
            type="primary" if active else "secondary",
        ):
            chosen = page
    return chosen


with st.sidebar:
    st.markdown(
        f"""<div class="bg-brand">
             <div class="mark">🛡️</div>
             <div><div class="name">{APP_NAME}</div>
             <div class="sub">Defence Threat Detection</div></div>
           </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)
    selected = nav_tiles()
    if selected != st.session_state.nav:
        st.session_state.nav = selected
        st.rerun()

    st.markdown("---")
    model_ok = MODEL_PATH.exists()
    st.markdown(
        f"<div class='small-note'>{'🟢' if model_ok else '🔴'} Model: "
        f"{'best.pt loaded' if model_ok else 'best.pt missing'}<br>"
        f"{'🟢' if GEMINI_API_KEY else '🟠'} Gemini: "
        f"{'configured' if GEMINI_API_KEY else 'not configured'}</div>",
        unsafe_allow_html=True,
    )
    st.caption("System Online · CPU inference")

page = st.session_state.nav


# ── Dashboard ─────────────────────────────────────────────────────────────────
def render_dashboard() -> None:
    snap = stats_snapshot()
    sev = {
        lvl: sum(1 for t in STATE_THREATS.values() if t["level"] == lvl)
        for lvl in ("Critical", "High", "Medium", "Low")
    }
    region_score = min(
        100,
        sev["Critical"] * 22 + sev["High"] * 12 + sev["Medium"] * 5 + sev["Low"] * 2,
    )
    posture = max(
        ("Critical", "High", "Medium", "Low"),
        key=lambda lvl: (core.LEVEL_RANK[lvl] if sev[lvl] else -1),
    )
    hot = [n for n, t in STATE_THREATS.items() if t["level"] in ("Critical", "High")]

    page_header(
        "🛡️ Command Dashboard",
        "Live detection pipeline, threat posture and regional activity.",
        f"<b>{len(STATE_THREATS)}</b> monitored regions · <b>{len(hot)}</b> elevated",
    )

    alert_banner(
        "Critical",
        f"{sev['Critical']} critical · {sev['High']} high threat regions",
        "Regions requiring immediate attention: " + ", ".join(sorted(hot)[:4]) + ".",
        "high-risk detections" if snap["high_risk"] else "no local alerts",
        snap["high_risk"],
    )

    # ── Hero + KPI strip ──────────────────────────────────────────────────────
    hero_col, kpi_col = st.columns([1, 2.35], gap="medium")
    with hero_col:
        hero_panel(
            posture,
            region_score,
            f"{len(hot)} of {len(STATE_THREATS)} monitored regions elevated",
            "Assessed from regional feed",
            f"{sev['Critical']} critical regions",
        )
    with kpi_col:
        r1 = st.columns(4)
        stat_card(r1[0], "🖼️", snap["images"], "Images", "linear-gradient(180deg,#00d4ff,#0066ff)")
        stat_card(r1[1], "🎬", snap["videos"], "Videos", "linear-gradient(180deg,#30d158,#00b34a)")
        stat_card(r1[2], "⚠️", snap["threats"], "Threats", "linear-gradient(180deg,#ff3b30,#b81f16)")
        stat_card(r1[3], "📄", snap["reports"], "Reports", "linear-gradient(180deg,#bf5af2,#7c3aed)")
        st.write("")
        r2 = st.columns(4)
        stat_card(r2[0], "🚨", snap["high_risk"], "High risk", "linear-gradient(180deg,#ff9500,#cc6a00)")
        stat_card(r2[1], "📡", len(STATE_THREATS), "Regions", "linear-gradient(180deg,#00d4ff,#0066cc)")
        stat_card(r2[2], "🔴", sev["Critical"], "Critical zones", "linear-gradient(180deg,#ff3b30,#b81f16)")
        stat_card(r2[3], "🎖️", len(get_class_names(str(MODEL_PATH))) if MODEL_PATH.exists() else 0,
                  "Asset classes", "linear-gradient(180deg,#ffd60a,#ccac00)")

    # ── Map (full width, dark basemap only) ───────────────────────────────────
    st.write("")
    with st.container(border=True):
        st.markdown(
            "<div class='panel-title'>🗺️ Regional Threat Map"
            "<span class='panel-sub'>CARTO dark matter · OpenStreetMap</span></div>",
            unsafe_allow_html=True,
        )
        st.pydeck_chart(
            build_threat_deck(threat_map.build_style_url()),
            width="stretch",
            height=560,
        )
        st.markdown(legend_html(), unsafe_allow_html=True)
        st.caption(
            "Threat shading is applied once per state — the region outline is drawn "
            "dark-on-dark so the colour reads as one solid area rather than a "
            "patchwork of districts. Hover a region for its threat record."
        )

    # ── Bottom row: activity, posture breakdown, system ───────────────────────
    st.write("")
    b1, b2, b3 = st.columns([1.5, 1, 1], gap="medium")

    with b1:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>⚡ Recent Activity</div>", unsafe_allow_html=True)
            items = st.session_state.activity
            if not items:
                st.markdown(
                    "<div class='small-note' style='padding:12px 0'>System ready — upload an "
                    "image or video to begin analysis.</div>",
                    unsafe_allow_html=True,
                )
            for item in items[:6]:
                st.markdown(
                    f"""<div class="feed-item"><div class="feed-ico">{item['icon']}</div>
                        <div><div class="feed-title">{item['title']}</div>
                        <div class="feed-time">{item['time']}</div></div></div>""",
                    unsafe_allow_html=True,
                )

    with b2:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>🎯 Posture Breakdown</div>", unsafe_allow_html=True)
            for lvl in ("Critical", "High", "Medium", "Low"):
                n = sev[lvl]
                color = core.threat_color(lvl)
                st.markdown(
                    f"""<div style="margin-bottom:9px">
                          <div style="display:flex;justify-content:space-between;font-size:11.5px;
                               color:var(--dim);margin-bottom:4px">
                            <span>{lvl}</span><span style="color:var(--text);font-weight:700">{n}</span>
                          </div>
                          <div style="height:7px;border-radius:5px;background:#0a1928;overflow:hidden">
                            <div style="height:7px;width:{(n / max(len(STATE_THREATS), 1)) * 100:.0f}%;
                                 background:{color};border-radius:5px"></div>
                          </div>
                        </div>""",
                    unsafe_allow_html=True,
                )

    with b3:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>ℹ️ System</div>", unsafe_allow_html=True)
            rows = [
                ("Project", APP_NAME),
                ("Detector", "YOLOv8 · best.pt" if MODEL_PATH.exists() else "model missing"),
                ("AI Analyst", "Gemini 2.5 Flash" if GEMINI_API_KEY else "Fallback mode"),
                ("Threat classes", str(len(get_class_names(str(MODEL_PATH))) if MODEL_PATH.exists() else 0)),
                ("Device", "CPU inference"),
            ]
            for k, v in rows:
                st.markdown(
                    f"<div class='info-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Threat region table ───────────────────────────────────────────────────
    st.write("")
    with st.container(border=True):
        st.markdown(
            "<div class='panel-title'>📋 Monitored Regions"
            "<span class='panel-sub'>hover the map for the same records</span></div>",
            unsafe_allow_html=True,
        )
        rows = [
            {"State / UT": name, "Threat Level": t["level"], "Detections": t["detections"],
             "Latest Asset": t["asset"], "Date": t["date"]}
            for name, t in STATE_THREATS.items()
        ]
        df = (
            pd.DataFrame(rows)
            .sort_values("Threat Level", key=lambda s: s.map(core.LEVEL_RANK), ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=min(420, 38 * len(df) + 42),
            column_config={
                "Threat Level": st.column_config.TextColumn(width="small"),
                "Detections": st.column_config.NumberColumn(width="small"),
                "Date": st.column_config.TextColumn(width="small"),
            },
        )


# ── Detection ─────────────────────────────────────────────────────────────────
def render_detection() -> None:
    page_header(
        "🎯 Threat Detection",
        "Upload an image or video. YOLO finds military assets and the threat engine scores them.",
        f"Detector: <b>YOLOv8</b> · {len(get_class_names(str(MODEL_PATH))) if MODEL_PATH.exists() else 0} asset classes",
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}. Add best.pt next to the app.")
        return

    uploaded = st.file_uploader(
        "Drop an image or video",
        type=["jpg", "jpeg", "png", "webp", "bmp", "mp4", "avi", "mov", "mkv", "webm"],
        help="Images up to 1600 px are analysed directly. Videos are sampled frame-by-frame.",
    )

    if not uploaded:
        st.info("Supported: JPG, PNG, WEBP, BMP, MP4, AVI, MOV, MKV, WEBM.")
        last = find_record(st.session_state.last_result_id)
        if last:
            st.markdown("### Last result")
            render_result(last)
        return

    name = uploaded.name
    ext = Path(name).suffix.lower()
    is_image = ext in core.SUPPORTED_IMAGE_EXT
    is_video = ext in core.SUPPORTED_VIDEO_EXT
    data = uploaded.getvalue()

    if not (is_image or is_video):
        st.error("Unsupported file type.")
        return

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("<div class='panel-title'>File Preview</div>", unsafe_allow_html=True)
        if is_image:
            st.image(Image.open(io.BytesIO(data)), width="stretch")
        else:
            st.video(data)
        st.caption(f"{name} · {len(data) / 1e6:.2f} MB")

    with right:
        st.markdown("<div class='panel-title'>Run Analysis</div>", unsafe_allow_html=True)
        with st.container(border=True):
            conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
            if is_video:
                st.caption("Videos are sampled (every 8th frame, max 40 frames) to keep CPU inference responsive.")
            run = st.button("🎯 Run Detection", type="primary", width="stretch")

        if run:
            model = get_model(str(MODEL_PATH))
            if is_image:
                with st.spinner("Running YOLO inference…"):
                    image = Image.open(io.BytesIO(data)).convert("RGB")
                    if max(image.size) > 1600:
                        image.thumbnail((1600, 1600))
                    result = core.analyze_image(model, image, conf=conf)
                    annotated = Image.fromarray(result["annotated"])
                    record = save_detection(
                        filename=name, kind="image", result=result, annotated=annotated
                    )
                st.toast(
                    f"{record['threat_level']} threat — {len(record['classes'])} object type(s)",
                    icon="⚠️" if record["classes"] else "✅",
                )
                st.rerun()
            else:
                tmp_path = None
                bar = st.progress(0.0, text="Opening video…")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                        tmp.write(data)
                        tmp_path = tmp.name
                    result = core.analyze_video(
                        model,
                        tmp_path,
                        conf=conf,
                        stride=8,
                        max_frames=40,
                        progress=lambda p, t: bar.progress(min(p, 1.0), text=t),
                    )
                    bar.progress(1.0, text="Finalising…")
                    if not result["frames"]:
                        st.error("No frames could be read from this video.")
                        return
                    annotated = Image.fromarray(result["frames"][0])
                    record = save_detection(
                        filename=name, kind="video", result=result, annotated=annotated
                    )
                    st.toast(
                        f"{record['threat_level']} threat — {result['sampled_frames']} frames analysed",
                        icon="⚠️" if record["classes"] else "✅",
                    )
                    st.rerun()
                finally:
                    bar.empty()
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

    last = find_record(st.session_state.last_result_id)
    if last:
        st.write("")
        render_result(last)


def render_result(record: dict) -> None:
    st.markdown("### Detection Results")
    level = record["threat_level"]
    score = record["threat_score"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='stat-card' style='--bar:{core.threat_color(level)}'>"
            f"<div class='lbl'>Threat Level</div>"
            f"<div style='margin-top:8px'>{badge(level)}</div></div>",
            unsafe_allow_html=True,
        )
    c2.metric("Threat score", f"{score}/100")
    c3.metric("Object types", len(record["classes"]))
    c4.metric("Objects", sum(record["counts"].values()) if record["counts"] else 0)

    st.progress(score / 100, text=f"Threat score {score}/100 · {level}")

    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("<div class='panel-title'>Annotated Evidence</div>", unsafe_allow_html=True)
        if record["type"] == "video" and record.get("clip"):
            st.video(record["clip"])
            st.caption(f"Annotated clip from {record['sampled_frames']} sampled frames.")
        elif record.get("annotated_jpeg"):
            st.image(annotated_image(record), width="stretch")

    with right:
        st.markdown("<div class='panel-title'>Detected Objects</div>", unsafe_allow_html=True)
        classes = record["classes"]
        if classes:
            st.markdown(
                "".join(
                    f"<span class='obj-tag'>{c.replace('_', ' ').title()} ×{record['counts'].get(c, 1)}</span>"
                    for c in classes
                ),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No objects detected above the confidence threshold.")

        if record["detections"]:
            df = pd.DataFrame(record["detections"])
            df["confidence"] = (df["confidence"] * 100).round(1)
            df = df.rename(
                columns={
                    "name": "Object",
                    "confidence": "Confidence %",
                    "threat_level": "Threat",
                }
            )
            df["Object"] = df["Object"].str.replace("_", " ").str.title()
            st.dataframe(df, width="stretch", hide_index=True)

        st.write("")
        safe_name = Path(record["filename"]).stem
        d1, d2 = st.columns(2)
        d1.download_button(
            "📄 Threat report (PDF)",
            data=pdf_for(record),
            file_name=f"Threat_Report_{safe_name}_{record['id'][:6]}.pdf",
            mime="application/pdf",
            width="stretch",
            key=f"pdf_dl_{record['id']}",
        )
        if record.get("annotated_jpeg"):
            d2.download_button(
                "🖼️ Annotated image",
                data=record["annotated_jpeg"],
                file_name=f"detection_{safe_name}_{record['id'][:6]}.jpg",
                mime="image/jpeg",
                width="stretch",
                key=f"img_dl_{record['id']}",
            )
        if record["type"] == "video" and record.get("clip"):
            st.download_button(
                "🎬 Annotated clip (MP4)",
                data=record["clip"],
                file_name=f"detection_{safe_name}_{record['id'][:6]}.mp4",
                mime="video/mp4",
                width="stretch",
                key=f"clip_dl_{record['id']}",
            )

        if st.button("🤖 Ask the Defence AI about this", width="stretch"):
            st.session_state.chat_target = record["id"]
            st.session_state.nav = PAGES[4]
            st.rerun()


# ── Reports ───────────────────────────────────────────────────────────────────
def render_reports() -> None:
    page_header(
        "📋 Reports",
        "A report is created automatically for every detection session.",
        f"<b>{len(st.session_state.reports)}</b> on file",
    )

    reports = st.session_state.reports
    if not reports:
        st.info("No reports yet. Run a detection to create one.")
        return

    f1, f2 = st.columns([2, 1])
    query = f1.text_input("🔍 Search reports", placeholder="File name…")
    level_filter = f2.selectbox(
        "Threat level", ["All", "Critical", "High", "Medium", "Low", "Safe"]
    )

    rows = [
        {
            "Report ID": r["id"][:12],
            "File Name": r["filename"],
            "Date": r["date"].strftime("%Y-%m-%d %H:%M"),
            "Threat Level": r["threat_level"],
            "Score": r["threat_score"],
            "Objects": r["objects"],
        }
        for r in reports
        if (level_filter == "All" or r["threat_level"] == level_filter)
        and (not query or query.lower() in r["filename"].lower())
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("")
    options = {f"{r['filename']} — {r['date']:%d %b %H:%M} — {r['threat_level']} — {r['id'][:6]}": r["id"] for r in reports}
    label = st.selectbox("Open a report", list(options))
    record = find_record(options[label])
    if record:
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Download PDF",
            data=pdf_for(record),
            file_name=f"Threat_Report_{record['id'][:8]}.pdf",
            mime="application/pdf",
            width="stretch",
            key=f"report_pdf_{record['id']}",
        )
        if c2.button("🗑 Delete report", width="stretch", key=f"report_del_{record['id']}"):
            st.session_state.reports = [r for r in reports if r["id"] != record["id"]]
            st.toast("Report deleted.")
            st.rerun()
        with st.expander("Report content", expanded=False):
            st.markdown(badge(record["threat_level"]), unsafe_allow_html=True)
            st.write(f"**File:** {record['filename']}")
            st.write(f"**Type:** {record['type'].title()}")
            st.write(f"**Threat score:** {record['threat_score']}/100")
            st.write(
                "**Detected:** "
                + (", ".join(f"{c.replace('_', ' ')} ×{record['counts'].get(c, 1)}" for c in record["classes"]) or "None")
            )
            st.write(core.RISK_TEXT.get(record["threat_level"], ""))


# ── History ───────────────────────────────────────────────────────────────────
def render_history() -> None:
    page_header(
        "🕑 Detection History",
        "Every file analysed in this session, newest first.",
        f"<b>{len(st.session_state.history)}</b> records",
    )

    history = st.session_state.history
    if not history:
        st.info("No detections yet. Upload a file on the Threat Detection page.")
        return

    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("🔍 Search history", placeholder="File name…")
    level = f2.selectbox("Level", ["All", "Critical", "High", "Medium", "Low", "Safe"])
    kind = f3.selectbox("Type", ["All", "image", "video"])

    filtered = [
        h
        for h in history
        if (level == "All" or h["threat_level"] == level)
        and (kind == "All" or h["type"] == kind)
        and (not query or query.lower() in h["filename"].lower())
    ]

    if not filtered:
        st.warning("No records match those filters.")
        return

    rows = [
        {
            "File": h["filename"],
            "Date": h["date"].strftime("%Y-%m-%d %H:%M"),
            "Type": h["type"],
            "Objects": sum(h["counts"].values()) if h["counts"] else 0,
            "Threat Level": h["threat_level"],
            "Score": h["threat_score"],
        }
        for h in filtered
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("")
    label = st.selectbox(
        "Inspect a record",
        [f"{h['filename']} — {h['date']:%d %b %H:%M} — {h['threat_level']} — {h['id'][:6]}" for h in filtered],
    )
    labels_list = [f"{h['filename']} — {h['date']:%d %b %H:%M} — {h['threat_level']} — {h['id'][:6]}" for h in filtered]
    idx = labels_list.index(label)
    record = filtered[idx]

    c1, c2 = st.columns([1.3, 1])
    with c1:
        if record.get("thumbnail"):
            st.image(record["thumbnail"], caption=record["filename"], width=360)
    with c2:
        st.markdown(badge(record["threat_level"]), unsafe_allow_html=True)
        st.write(f"**Score:** {record['threat_score']}/100")
        st.write(f"**Type:** {record['type'].title()}")
        st.write(
            "**Objects:** "
            + (", ".join(c.replace("_", " ") for c in record["classes"]) or "None")
        )
        d1, d2 = st.columns(2)
        d1.download_button(
            "📄 PDF",
            data=pdf_for(record),
            file_name=f"Threat_Report_{record['id'][:8]}.pdf",
            mime="application/pdf",
            width="stretch",
            key=f"hist_pdf_{record['id']}",
        )
        if d2.button("🗑 Delete", width="stretch", key=f"hist_del_{record['id']}"):
            st.session_state.history = [h for h in history if h["id"] != record["id"]]
            st.session_state.reports = [
                r for r in st.session_state.reports if r["id"] != record["id"]
            ]
            st.session_state.chats.pop(record["id"], None)
            st.session_state.pdf_cache.pop(record["id"], None)
            if st.session_state.last_result_id == record["id"]:
                st.session_state.last_result_id = None
            st.toast("Record deleted.")
            st.rerun()

    if st.button("🗑 Clear all history", type="secondary"):
        st.session_state.history = []
        st.session_state.reports = []
        st.session_state.chats = {}
        st.session_state.pdf_cache = {}
        st.session_state.last_result_id = None
        st.toast("History cleared.")
        st.rerun()


# ── AI Chat ───────────────────────────────────────────────────────────────────
def render_chat() -> None:
    page_header(
        "🤖 AI Intelligence Chat",
        "Ask about the assets in a detection session, threat reasoning and countermeasures.",
        "Engine: <b>Gemini 2.5 Flash</b>" if GEMINI_API_KEY else "<b>Fallback mode</b>",
    )

    history = st.session_state.history
    if not history:
        st.info("Run a detection first — the AI answers against a detection session's context.")
        return

    labels = {
        f"{h['filename']} — {h['date']:%d %b %H:%M} — {h['threat_level']} — {h['id'][:6]}": h["id"] for h in history
    }
    default_id = st.session_state.get("chat_target") or st.session_state.last_result_id
    default_labels = [lab for lab, rid in labels.items() if rid == default_id]
    chosen = st.selectbox(
        "Detection session",
        list(labels),
        index=list(labels).index(default_labels[0]) if default_labels else 0,
    )
    chat_id = labels[chosen]
    record = find_record(chat_id)
    st.session_state.chat_target = chat_id

    if record["classes"]:
        st.markdown(
            "".join(f"<span class='obj-tag'>{c.replace('_', ' ').title()}</span>" for c in record["classes"]),
            unsafe_allow_html=True,
        )

    messages = st.session_state.chats.setdefault(chat_id, [])
    if not messages:
        with st.chat_message("assistant"):
            st.markdown(
                f"Hello, Commander. I have the context for **{record['filename']}** — "
                f"threat level **{record['threat_level']}**, score **{record['threat_score']}/100**. "
                "Ask me for an assessment, operational risks or countermeasures."
            )

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    suggestions = [
        "What is the threat level?",
        "Explain the detected assets",
        "Recommend countermeasures",
        "Strategic impact assessment",
    ]
    cols = st.columns(len(suggestions))
    clicked = None
    for col, text in zip(cols, suggestions):
        if col.button(text, width="stretch", key=f"sugg_{abs(hash(text))}"):
            clicked = text

    prompt = st.chat_input("Ask about detected military assets…")
    prompt = clicked or prompt
    if prompt:
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                reply, error = core.gemini_reply(GEMINI_API_KEY, record["classes"], prompt)
            st.markdown(reply)
            if error:
                st.caption(f"Gemini fallback used: {error}")
        messages.append({"role": "assistant", "content": reply})
        log_activity("🤖", f"AI conversation: {prompt[:40]}")
        st.rerun()


# ── Analytics ─────────────────────────────────────────────────────────────────
def render_analytics() -> None:
    page_header(
        "📈 Analytics",
        "Aggregated from the detections in this session.",
        f"<b>{len(st.session_state.history)}</b> sessions analysed",
    )

    history = st.session_state.history
    if not history:
        st.info("No data yet — run a detection to populate analytics.")
        return

    asset_counts: Counter = Counter()
    level_counts = Counter({k: 0 for k in ["Critical", "High", "Medium", "Low", "Safe"]})
    daily: Counter = Counter()
    for h in history:
        for cls, n in h["counts"].items():
            asset_counts[cls] += n
        level_counts[h["threat_level"]] += 1
        daily[h["date"].date()] += 1

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>🎖 Most Detected Assets</div>", unsafe_allow_html=True)
            df = pd.DataFrame(
                [{"Asset": k.replace("_", " ").title(), "Count": v} for k, v in asset_counts.items()]
            ).sort_values("Count", ascending=False).head(10)
            if df.empty:
                st.caption("No objects recorded.")
            else:
                chart = (
                    alt.Chart(df)
                    .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                    .encode(
                        x=alt.X("Count:Q", axis=alt.Axis(grid=False)),
                        y=alt.Y("Asset:N", sort="-x", axis=alt.Axis(title=None)),
                        color=alt.value("#00d4ff"),
                        tooltip=["Asset", "Count"],
                    )
                    .properties(height=260, background="transparent")
                )
                st.altair_chart(chart, width="stretch")

    with right:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>🔴 Threat Level Distribution</div>", unsafe_allow_html=True)
            ldf = pd.DataFrame(
                [{"Level": k, "Sessions": v} for k, v in level_counts.items() if v > 0]
            )
            if ldf.empty:
                st.caption("No sessions recorded.")
            else:
                color_scale = alt.Scale(
                    domain=["Critical", "High", "Medium", "Low", "Safe"],
                    range=["#ff3b30", "#ff9500", "#ffd60a", "#30d158", "#2a4a6a"],
                )
                donut = (
                    alt.Chart(ldf)
                    .mark_arc(innerRadius=58, outerRadius=105, stroke="#0b1623", strokeWidth=2)
                    .encode(
                        theta=alt.Theta("Sessions:Q"),
                        color=alt.Color("Level:N", scale=color_scale, legend=alt.Legend(orient="bottom")),
                        tooltip=["Level", "Sessions"],
                    )
                    .properties(height=260, background="transparent")
                )
                st.altair_chart(donut, width="stretch")

    with st.container(border=True):
        st.markdown("<div class='panel-title'>📈 Daily Detection Count (Last 7 Days)</div>", unsafe_allow_html=True)
        today = datetime.now().date()
        days = [(today - timedelta(days=i)) for i in range(6, -1, -1)]
        ddf = pd.DataFrame(
            [{"Date": d.strftime("%d %b"), "Detections": daily.get(d, 0)} for d in days]
        )
        line = (
            alt.Chart(ddf)
            .mark_line(point=True, color="#00d4ff", strokeWidth=2.5)
            .encode(
                x=alt.X("Date:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Detections:Q", axis=alt.Axis(grid=True, tickMinStep=1)),
                tooltip=["Date", "Detections"],
            )
            .properties(height=220, background="transparent")
        )
        st.altair_chart(line, width="stretch")


# ── Settings ──────────────────────────────────────────────────────────────────
def render_settings() -> None:
    page_header(
        "⚙️ Settings",
        "Runtime configuration, secrets status and session controls.",
        f"<b>{APP_NAME}</b> v2.0",
    )

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>System</div>", unsafe_allow_html=True)
            rows = [
                ("Model path", str(MODEL_PATH.name)),
                ("Model present", "Yes" if MODEL_PATH.exists() else "No"),
                ("Device", "CPU"),
                ("Detector", "Ultralytics YOLO"),
                ("Backend", "Streamlit"),
                ("Map basemap", "CARTO · OpenStreetMap"),
            ]
            for k, v in rows:
                st.markdown(
                    f"<div class='info-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>",
                    unsafe_allow_html=True,
                )

        with st.container(border=True):
            st.markdown("<div class='panel-title'>Secrets</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='info-row'><span class='k'>GEMINI_API_KEY</span>"
                f"<span class='v {'pill-ok' if GEMINI_API_KEY else 'pill-warn'}'>"
                f"{'configured' if GEMINI_API_KEY else 'missing (fallback mode)'}</span></div>"
                f"<div class='info-row'><span class='k'>CARTO_API_KEY</span>"
                f"<span class='v {'pill-ok' if get_secret('CARTO_API_KEY') else 'pill-warn'}'>"
                f"{'configured' if get_secret('CARTO_API_KEY') else 'optional / keyless basemap'}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Set these in Streamlit Cloud → App settings → Secrets, or in "
                ".streamlit/secrets.toml / .env locally. Never commit keys to Git."
            )
            st.markdown(
                "<div class='panel-title' style='margin-top:12px'>Resolved map style</div>",
                unsafe_allow_html=True,
            )
            resolved = threat_map.build_style_url()
            shown = resolved.replace(get_secret("CARTO_API_KEY"), "***") if get_secret("CARTO_API_KEY") else resolved
            st.code(shown, language="text")
            st.caption(
                "CARTO's public styles need no key. With CARTO_API_KEY set, the key is "
                "appended as ?key= (the CDN ignores ?api_key=) or substituted into a "
                "{api_key}/{key} placeholder in CARTO_STYLE_URL."
            )

    with right:
        with st.container(border=True):
            st.markdown("<div class='panel-title'>Model Classes</div>", unsafe_allow_html=True)
            if MODEL_PATH.exists():
                classes = get_class_names(str(MODEL_PATH))
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Class": [c.replace("_", " ").title() for c in classes],
                            "Threat": [core.obj_threat(c) for c in classes],
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.warning("best.pt is missing.")

        with st.container(border=True):
            st.markdown("<div class='panel-title'>Threat Map Regions</div>", unsafe_allow_html=True)
            st.caption(
                f"{len(load_states_geojson()['features'])} district polygons across "
                f"{len({f['properties']['state'] for f in load_states_geojson()['features']})} states/UTs. "
                "Regions outside the seeded list render as Safe."
            )

        with st.container(border=True):
            st.markdown("<div class='panel-title'>Session Data</div>", unsafe_allow_html=True)
            st.caption(
                "Detections, reports and chats live in this browser session only — "
                "restarting the app clears them. Download PDFs you want to keep."
            )
            if st.button("🗑 Clear all session data", type="secondary", width="stretch"):
                st.session_state.history = []
                st.session_state.reports = []
                st.session_state.chats = {}
                st.session_state.activity = []
                st.session_state.pdf_cache = {}
                st.session_state.last_result_id = None
                st.toast("Session data cleared.")
                st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
RENDERERS = {
    PAGES[0]: render_dashboard,
    PAGES[1]: render_detection,
    PAGES[2]: render_reports,
    PAGES[3]: render_history,
    PAGES[4]: render_chat,
    PAGES[5]: render_analytics,
    PAGES[6]: render_settings,
}
RENDERERS[page]()
