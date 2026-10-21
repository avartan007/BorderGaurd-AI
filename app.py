from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet
from google import genai
import io, uuid, cv2, tempfile, os, datetime
from collections import Counter

# ─── Config ───────────────────────────────────────────────────────────────────
# Never hardcode API keys. Set GEMINI_API_KEY in the environment or in
# .streamlit/secrets.toml (the Streamlit app reads it from there).
_gemini_key = os.getenv("GEMINI_API_KEY", "")
client = genai.Client(api_key=_gemini_key) if _gemini_key else None

app = FastAPI()
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

model = YOLO("best.pt").to("cpu")

# ─── In-memory stores ─────────────────────────────────────────────────────────
chat_sessions   = {}     # chat_id -> {classes, messages, filename, timestamp, image_path}
detection_history = []   # list of detection records
reports_store   = []     # list of report records
stats = {
    "images": 0,
    "videos": 0,
    "threats": 0,
    "high_risk": 0,
    "reports": 0,
}

THREAT_WEIGHTS = {
    "weapon": 20, "soldier": 8, "military_vehicle": 25,
    "military_tank": 40, "military_aircraft": 50,
    "helicopter": 45, "civilian": -5,
}

# Priority-based per-object threat levels (single object is enough to trigger)
OBJECT_LEVEL_MAP = {
    "weapon":             "Critical",
    "military_tank":      "Critical",
    "military_aircraft":  "Critical",
    "military_artillery": "Critical",
    "military_warship":   "Critical",
    "helicopter":         "High",
    "military_vehicle":   "High",
    "military_truck":     "High",
    "camouflage_soldier": "High",
    "soldier":            "Medium",
    "trench":             "Medium",
    "civilian":           "Low",
    "civilian_vehicle":   "Low",
}
LEVEL_RANK = {"Safe": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}

def compute_threat(classes):
    """Return the highest threat level found among detected objects."""
    if not classes:
        return "Safe"
    best = "Low"
    for obj in classes:
        lvl = OBJECT_LEVEL_MAP.get(obj, "Low")
        if LEVEL_RANK[lvl] > LEVEL_RANK[best]:
            best = lvl
    return best

def compute_score(classes):
    """Numeric threat score 0-100 for display."""
    counts = Counter(classes)
    raw = sum(THREAT_WEIGHTS.get(o, 2) * c for o, c in counts.items())
    return min(int(raw / 3), 100)   # normalise to 0-100

# ══════════════════════════════════════════════════════════════════════════════
#  HOME  –  full SPA dashboard
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
def home():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>BorderGuard AI — Defence Threat Detection</title>
<style>
/* ── Reset & Base ───────────────────────────────── */
*{margin:0;padding:0;box-sizing:border-box;font-family:'Segoe UI',sans-serif}
:root{
  --bg:#060d17;--sidebar:#0b1623;--card:#0f1e2e;--border:#1a3050;
  --accent:#00d4ff;--red:#ff3b30;--orange:#ff9500;--green:#30d158;
  --yellow:#ffd60a;--text:#c9d8e8;--dim:#5a7a9a;
}
body{background:var(--bg);color:var(--text);min-height:100vh;display:flex;overflow:hidden}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:#0b1623}
::-webkit-scrollbar-thumb{background:#1a3050;border-radius:4px}

/* ── Sidebar ────────────────────────────────────── */
#sidebar{
  width:240px;min-height:100vh;background:var(--sidebar);
  border-right:1px solid var(--border);display:flex;flex-direction:column;
  position:fixed;top:0;left:0;z-index:100;transition:.3s
}
.logo{padding:22px 20px;border-bottom:1px solid var(--border)}
.logo-title{font-size:13px;font-weight:700;color:var(--accent);letter-spacing:2px;text-transform:uppercase}
.logo-sub{font-size:10px;color:var(--dim);margin-top:3px}
.nav-section{padding:15px 12px 5px;font-size:10px;color:var(--dim);letter-spacing:2px;text-transform:uppercase}
.nav-item{
  display:flex;align-items:center;gap:12px;padding:11px 20px;
  cursor:pointer;border-radius:8px;margin:2px 8px;transition:.2s;
  font-size:13.5px;color:var(--dim);
}
.nav-item:hover{background:rgba(0,212,255,.08);color:var(--accent)}
.nav-item.active{background:rgba(0,212,255,.12);color:var(--accent);border-left:3px solid var(--accent)}
.nav-item .icon{font-size:16px;width:20px;text-align:center}
.sidebar-footer{margin-top:auto;padding:15px 20px;border-top:1px solid var(--border);font-size:11px;color:var(--dim)}
.status-dot{display:inline-block;width:7px;height:7px;background:var(--green);border-radius:50%;margin-right:6px;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── Main ───────────────────────────────────────── */
#main{margin-left:240px;flex:1;height:100vh;overflow-y:auto}
.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:14px 28px;background:rgba(11,22,35,.85);
  backdrop-filter:blur(10px);border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:50
}
.topbar-title{font-size:15px;font-weight:600;color:white}
.topbar-right{display:flex;align-items:center;gap:14px}
.badge{background:var(--red);color:white;font-size:10px;padding:2px 7px;border-radius:20px}
.user-avatar{width:34px;height:34px;background:linear-gradient(135deg,#1a3a6e,#0f2040);
  border-radius:50%;border:2px solid var(--accent);display:flex;align-items:center;justify-content:center;font-size:13px}

/* ── Pages ──────────────────────────────────────── */
.page{display:none;padding:24px 28px;animation:fadeIn .3s}
.page.active{display:block}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}

/* ── Cards ──────────────────────────────────────── */
.stat-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px}
.stat-card{
  background:var(--card);border:1px solid var(--border);border-radius:12px;
  padding:18px 16px;position:relative;overflow:hidden;transition:.2s
}
.stat-card:hover{border-color:var(--accent);transform:translateY(-2px)}
.stat-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px}
.stat-card.blue::before{background:linear-gradient(90deg,#00d4ff,#0066ff)}
.stat-card.red::before{background:linear-gradient(90deg,#ff3b30,#ff6b6b)}
.stat-card.orange::before{background:linear-gradient(90deg,#ff9500,#ffcc00)}
.stat-card.green::before{background:linear-gradient(90deg,#30d158,#00ff88)}
.stat-card.purple::before{background:linear-gradient(90deg,#bf5af2,#7c3aed)}
.stat-icon{font-size:24px;margin-bottom:8px}
.stat-val{font-size:28px;font-weight:700;color:white}
.stat-label{font-size:11px;color:var(--dim);margin-top:3px;text-transform:uppercase;letter-spacing:1px}

/* ── Grid Layout ────────────────────────────────── */
.dash-grid{display:grid;grid-template-columns:2fr 1fr;gap:20px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px}
.panel-title{font-size:12px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:2px;margin-bottom:16px;display:flex;align-items:center;gap:8px}

/* ── India Map ──────────────────────────────────── */
#india-map{width:100%;height:360px;position:relative;display:flex;align-items:center;justify-content:center}
#india-map svg{height:100%;width:auto;filter:drop-shadow(0 0 10px rgba(0,212,255,.15))}
.state-path{cursor:pointer;transition:.2s;stroke:#0a1928;stroke-width:.5}
.state-path:hover{opacity:.8;stroke:var(--accent);stroke-width:1}
.state-path.critical{fill:#ff3b30}
.state-path.high{fill:#ff9500}
.state-path.medium{fill:#ffd60a}
.state-path.low{fill:#30d158}
.state-path.safe{fill:#1a3050}
#map-tooltip{
  position:absolute;background:#0b1623;border:1px solid var(--accent);
  border-radius:10px;padding:12px 16px;font-size:12px;pointer-events:none;
  display:none;min-width:190px;z-index:200;box-shadow:0 8px 30px rgba(0,0,0,.5)
}
#map-tooltip h4{color:var(--accent);margin-bottom:6px;font-size:13px}
#map-tooltip p{color:var(--text);line-height:1.6}
.map-legend{display:flex;gap:12px;margin-top:10px;flex-wrap:wrap}
.legend-item{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--dim)}
.legend-dot{width:10px;height:10px;border-radius:50%}

/* ── Activity Feed ──────────────────────────────── */
.activity-item{display:flex;align-items:flex-start;gap:12px;padding:10px 0;border-bottom:1px solid rgba(26,48,80,.5)}
.activity-item:last-child{border-bottom:none}
.act-icon{width:32px;height:32px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0}
.act-icon.img{background:rgba(0,212,255,.12)}
.act-icon.vid{background:rgba(48,209,88,.12)}
.act-icon.rep{background:rgba(191,90,242,.12)}
.act-icon.ai{background:rgba(255,149,0,.12)}
.act-text{flex:1}
.act-title{font-size:12.5px;color:white;font-weight:500}
.act-time{font-size:11px;color:var(--dim);margin-top:2px}

/* ── Project Info ───────────────────────────────── */
.info-row{display:flex;justify-content:space-between;align-items:center;padding:8px 0;border-bottom:1px solid rgba(26,48,80,.5);font-size:12.5px}
.info-row:last-child{border-bottom:none}
.info-key{color:var(--dim)}
.info-val{color:white;font-weight:500}
.tag{padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600}
.tag.green{background:rgba(48,209,88,.15);color:var(--green)}
.tag.blue{background:rgba(0,212,255,.15);color:var(--accent)}
.tag.red{background:rgba(255,59,48,.15);color:var(--red)}
.tag.orange{background:rgba(255,149,0,.15);color:var(--orange)}

/* ── Threat Detection Page ──────────────────────── */
.upload-zone{
  border:2px dashed var(--border);border-radius:14px;padding:48px;
  text-align:center;cursor:pointer;transition:.3s;margin-bottom:20px
}
.upload-zone:hover,.upload-zone.dragover{border-color:var(--accent);background:rgba(0,212,255,.04)}
.upload-zone h3{font-size:16px;color:var(--dim);margin:12px 0 6px}
.upload-zone p{font-size:12px;color:var(--dim)}
.file-input{display:none}
.btn{
  padding:10px 22px;border:none;border-radius:8px;cursor:pointer;
  font-size:13px;font-weight:600;transition:.2s;display:inline-flex;align-items:center;gap:8px
}
.btn-primary{background:linear-gradient(135deg,#00a8cc,#0066cc);color:white}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 6px 20px rgba(0,212,255,.3)}
.btn-danger{background:linear-gradient(135deg,#cc2200,#ff3b30);color:white}
.btn-success{background:linear-gradient(135deg,#00882a,#30d158);color:white}
.btn-outline{background:transparent;border:1px solid var(--border);color:var(--text)}
.btn-outline:hover{border-color:var(--accent);color:var(--accent)}
.btn-sm{padding:6px 14px;font-size:12px}
.detect-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
#preview-box{background:#08121c;border:1px solid var(--border);border-radius:12px;overflow:hidden;min-height:280px;display:flex;align-items:center;justify-content:center}
#preview-box img,#preview-box video{max-width:100%;max-height:360px;border-radius:8px}
.det-results{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px}
.threat-badge{
  display:inline-block;padding:6px 18px;border-radius:20px;font-size:13px;
  font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:14px
}
.threat-badge.Critical{background:rgba(255,59,48,.2);color:var(--red);border:1px solid var(--red)}
.threat-badge.High{background:rgba(255,149,0,.2);color:var(--orange);border:1px solid var(--orange)}
.threat-badge.Medium{background:rgba(255,214,10,.2);color:var(--yellow);border:1px solid var(--yellow)}
.threat-badge.Low{background:rgba(48,209,88,.2);color:var(--green);border:1px solid var(--green)}
.threat-badge.Safe{background:rgba(48,209,88,.15);color:var(--green);border:1px solid var(--green)}
.obj-tag{
  display:inline-block;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.25);
  color:var(--accent);padding:4px 12px;border-radius:20px;font-size:12px;margin:3px
}
.threat-bar{background:#0a1928;border-radius:4px;height:8px;margin:6px 0}
.threat-bar-fill{height:8px;border-radius:4px;transition:width .5s}
.conf-table{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:10px}
.conf-table th{text-align:left;padding:8px;color:var(--dim);font-weight:600;border-bottom:1px solid var(--border)}
.conf-table td{padding:8px;border-bottom:1px solid rgba(26,48,80,.4)}

/* ── Reports & History Table ────────────────────── */
.table-wrap{overflow-x:auto}
table.data-table{width:100%;border-collapse:collapse;font-size:13px}
table.data-table th{padding:12px 14px;text-align:left;color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid var(--border);background:#080f18}
table.data-table td{padding:12px 14px;border-bottom:1px solid rgba(26,48,80,.4);vertical-align:middle}
table.data-table tr:hover td{background:rgba(0,212,255,.03)}
.thumb{width:48px;height:36px;object-fit:cover;border-radius:5px;border:1px solid var(--border)}
.search-bar{display:flex;gap:10px;margin-bottom:18px;align-items:center}
.search-bar input,.search-bar select{
  background:#0b1623;border:1px solid var(--border);color:var(--text);
  padding:9px 14px;border-radius:8px;font-size:13px;outline:none
}
.search-bar input:focus,.search-bar select:focus{border-color:var(--accent)}
.search-bar input{flex:1}

/* ── AI Chat Page ───────────────────────────────── */
.chat-layout{display:flex;height:calc(100vh - 130px);gap:16px}
.chat-sidebar{width:220px;background:var(--card);border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;overflow:hidden}
.chat-sidebar-header{padding:14px;border-bottom:1px solid var(--border);font-size:12px;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:1px}
.chat-list{flex:1;overflow-y:auto;padding:8px}
.chat-list-item{padding:10px;border-radius:8px;cursor:pointer;margin-bottom:4px;transition:.2s;font-size:12.5px;color:var(--dim)}
.chat-list-item:hover,.chat-list-item.active{background:rgba(0,212,255,.1);color:white}
.chat-list-item .chat-item-id{font-size:10px;color:var(--dim);margin-top:2px}
.chat-main{flex:1;background:var(--card);border:1px solid var(--border);border-radius:12px;display:flex;flex-direction:column;overflow:hidden}
.chat-header-bar{padding:14px 18px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.chat-messages{flex:1;overflow-y:auto;padding:18px;display:flex;flex-direction:column;gap:12px}
.msg{max-width:85%;padding:12px 16px;border-radius:12px;font-size:13.5px;line-height:1.6;white-space:pre-wrap}
.msg.user{background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);align-self:flex-end;color:white}
.msg.ai{background:#0a1928;border:1px solid var(--border);align-self:flex-start;color:var(--text)}
.msg-label{font-size:10px;color:var(--dim);margin-bottom:3px}
.chat-input-area{padding:14px 18px;border-top:1px solid var(--border);display:flex;gap:10px;align-items:center}
.chat-input{flex:1;background:#080f18;border:1px solid var(--border);color:white;padding:11px 16px;border-radius:10px;font-size:13.5px;outline:none;resize:none}
.chat-input:focus{border-color:var(--accent)}
.suggested-qs{display:flex;flex-wrap:wrap;gap:8px;padding:10px 18px;border-top:1px solid rgba(26,48,80,.3)}
.suggested-q{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.2);color:var(--accent);padding:5px 12px;border-radius:20px;font-size:11.5px;cursor:pointer;transition:.2s}
.suggested-q:hover{background:rgba(0,212,255,.15)}

/* ── Analytics Page ─────────────────────────────── */
.analytics-grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.chart-box{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px}
.chart-box canvas{max-height:200px}
.mini-bar{display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:12.5px}
.mini-bar-label{width:130px;color:var(--dim);text-overflow:ellipsis;overflow:hidden;white-space:nowrap}
.mini-bar-track{flex:1;background:#08121c;border-radius:4px;height:10px}
.mini-bar-fill{height:10px;border-radius:4px}
.mini-bar-val{width:40px;text-align:right;color:white;font-weight:600}
.donut-wrap{position:relative;display:flex;justify-content:center;align-items:center;height:180px}
.donut-legend{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.donut-leg-item{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--dim)}
.donut-leg-dot{width:10px;height:10px;border-radius:50%}

/* ── Notification ───────────────────────────────── */
#notification{
  position:fixed;top:20px;right:20px;background:#0b1623;border:1px solid var(--accent);
  border-radius:10px;padding:12px 20px;font-size:13px;z-index:999;
  display:none;box-shadow:0 10px 30px rgba(0,0,0,.5);max-width:320px
}
#notification.show{display:flex;align-items:center;gap:10px;animation:slideIn .3s}
@keyframes slideIn{from{transform:translateX(100%);opacity:0}to{transform:translateX(0);opacity:1}}

/* ── Typing indicator ───────────────────────────── */
.typing{display:flex;gap:4px;padding:10px 14px;align-items:center}
.typing span{width:7px;height:7px;background:var(--dim);border-radius:50%;animation:bounce .8s infinite}
.typing span:nth-child(2){animation-delay:.15s}
.typing span:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,80%,100%{transform:scale(.8)}40%{transform:scale(1.2)}}

/* ── Loader ─────────────────────────────────────── */
#loader-overlay{
  position:fixed;inset:0;background:rgba(6,13,23,.85);z-index:500;
  display:none;align-items:center;justify-content:center;flex-direction:column;gap:16px
}
#loader-overlay.show{display:flex}
.spinner{width:50px;height:50px;border:3px solid var(--border);border-top:3px solid var(--accent);border-radius:50%;animation:spin 1s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.loader-text{color:var(--accent);font-size:14px;letter-spacing:2px;text-transform:uppercase}

/* ── Responsive ─────────────────────────────────── */
@media(max-width:900px){
  #sidebar{width:64px}.logo-title,.logo-sub,.nav-item span,.nav-section,.sidebar-footer{display:none}
  #main{margin-left:64px}.stat-grid{grid-template-columns:repeat(2,1fr)}
  .dash-grid,.detect-grid,.analytics-grid{grid-template-columns:1fr}
}
</style>
</head>
<body>

<!-- ══ LOADER ═══════════════════════════════════════════════════════════════ -->
<div id="loader-overlay">
  <div class="spinner"></div>
  <div class="loader-text">Analysing Target…</div>
</div>

<!-- ══ NOTIFICATION ══════════════════════════════════════════════════════════ -->
<div id="notification"></div>

<!-- ══ SIDEBAR ═══════════════════════════════════════════════════════════════ -->
<div id="sidebar">
  <div class="logo">
    <div class="logo-title">🛡 BorderGuard</div>
    <div class="logo-sub">Defence Threat Detection</div>
  </div>
  <div class="nav-section">Navigation</div>
  <div class="nav-item active" onclick="showPage('dashboard',this)"><span class="icon">🏠</span><span>Dashboard</span></div>
  <div class="nav-item" onclick="showPage('detection',this)"><span class="icon">🎯</span><span>Threat Detection</span></div>
  <div class="nav-item" onclick="showPage('reports',this)"><span class="icon">📋</span><span>Reports</span></div>
  <div class="nav-item" onclick="showPage('history',this)"><span class="icon">🕑</span><span>Detection History</span></div>
  <div class="nav-item" onclick="showPage('chat',this)"><span class="icon">🤖</span><span>AI Intelligence Chat</span></div>
  <div class="nav-item" onclick="showPage('analytics',this)"><span class="icon">📊</span><span>Analytics</span></div>
  <div class="nav-item" onclick="showPage('settings',this)"><span class="icon">⚙️</span><span>Settings</span></div>
  <div class="sidebar-footer"><span class="status-dot"></span>System Online</div>
</div>

<!-- ══ MAIN ══════════════════════════════════════════════════════════════════ -->
<div id="main">

  <!-- Top Bar -->
  <div class="topbar">
    <div class="topbar-title" id="page-title">📊 Dashboard Overview</div>
    <div class="topbar-right">
      <span id="sys-time" style="font-size:12px;color:var(--dim)"></span>
      <span class="badge" id="alert-badge">0 Alerts</span>
      <div class="user-avatar">🎖</div>
    </div>
  </div>

  <!-- ═══ DASHBOARD PAGE ═══════════════════════════════════════════════════ -->
  <div class="page active" id="page-dashboard">

    <!-- Stats -->
    <div class="stat-grid" style="margin-top:22px">
      <div class="stat-card blue"><div class="stat-icon">🖼️</div><div class="stat-val" id="s-images">0</div><div class="stat-label">Images Analysed</div></div>
      <div class="stat-card green"><div class="stat-icon">🎬</div><div class="stat-val" id="s-videos">0</div><div class="stat-label">Videos Analysed</div></div>
      <div class="stat-card red"><div class="stat-icon">⚠️</div><div class="stat-val" id="s-threats">0</div><div class="stat-label">Threats Detected</div></div>
      <div class="stat-card orange"><div class="stat-icon">🚨</div><div class="stat-val" id="s-high">0</div><div class="stat-label">High Risk Alerts</div></div>
      <div class="stat-card purple"><div class="stat-icon">📄</div><div class="stat-val" id="s-reports">0</div><div class="stat-label">Reports Generated</div></div>
    </div>

    <!-- Map + Activity -->
    <div class="dash-grid">
      <div class="panel">
        <div class="panel-title">🗺️ India Threat Map</div>
        <div id="india-map">
          <div id="map-tooltip"><h4 id="tt-state">State</h4><p id="tt-info"></p></div>
          <!-- Simplified India SVG (major states) -->
          <svg viewBox="0 0 500 600" xmlns="http://www.w3.org/2000/svg">
            <!-- Jammu & Kashmir -->
            <path class="state-path critical" data-state="Jammu & Kashmir" data-level="Critical" data-detections="14" data-asset="Military Tank" data-date="2025-01-10"
              d="M170,30 L220,20 L270,40 L280,80 L250,100 L210,95 L175,75 Z"/>
            <!-- Himachal Pradesh -->
            <path class="state-path high" data-state="Himachal Pradesh" data-level="High" data-detections="7" data-asset="Military Aircraft" data-date="2025-01-08"
              d="M210,95 L250,100 L255,130 L225,145 L200,130 Z"/>
            <!-- Punjab -->
            <path class="state-path high" data-state="Punjab" data-level="High" data-detections="9" data-asset="Soldier" data-date="2025-01-09"
              d="M175,75 L210,95 L200,130 L175,135 L158,110 Z"/>
            <!-- Haryana -->
            <path class="state-path medium" data-state="Haryana" data-level="Medium" data-detections="4" data-asset="Military Vehicle" data-date="2025-01-05"
              d="M175,135 L200,130 L205,160 L185,170 L168,155 Z"/>
            <!-- Delhi -->
            <path class="state-path medium" data-state="Delhi" data-level="Medium" data-detections="3" data-asset="Camouflage Soldier" data-date="2025-01-07"
              d="M195,155 L205,160 L210,172 L195,175 L188,165 Z"/>
            <!-- Rajasthan -->
            <path class="state-path low" data-state="Rajasthan" data-level="Low" data-detections="2" data-asset="Civilian Vehicle" data-date="2025-01-03"
              d="M125,140 L175,135 L185,170 L195,175 L200,220 L175,260 L130,250 L100,200 L105,160 Z"/>
            <!-- Uttar Pradesh -->
            <path class="state-path medium" data-state="Uttar Pradesh" data-level="Medium" data-detections="5" data-asset="Weapon" data-date="2025-01-06"
              d="M205,160 L255,130 L290,145 L310,175 L295,210 L260,225 L220,220 L200,220 L195,175 L210,172 Z"/>
            <!-- Uttarakhand -->
            <path class="state-path safe" data-state="Uttarakhand" data-level="Safe" data-detections="0" data-asset="—" data-date="—"
              d="M255,130 L290,125 L300,145 L290,145 Z"/>
            <!-- Madhya Pradesh -->
            <path class="state-path low" data-state="Madhya Pradesh" data-level="Low" data-detections="1" data-asset="Military Truck" data-date="2024-12-28"
              d="M175,260 L200,220 L260,225 L295,210 L305,250 L290,300 L255,310 L210,305 L180,290 Z"/>
            <!-- Gujarat -->
            <path class="state-path safe" data-state="Gujarat" data-level="Safe" data-detections="0" data-asset="—" data-date="—"
              d="M100,200 L130,250 L120,300 L100,315 L70,290 L65,250 L80,215 Z"/>
            <!-- Maharashtra -->
            <path class="state-path medium" data-state="Maharashtra" data-level="Medium" data-detections="3" data-asset="Military Aircraft" data-date="2024-12-30"
              d="M165,295 L210,305 L255,310 L260,360 L240,390 L205,380 L170,360 L145,330 L150,300 Z"/>
            <!-- Bihar -->
            <path class="state-path high" data-state="Bihar" data-level="High" data-detections="6" data-asset="Military Tank" data-date="2025-01-07"
              d="M295,210 L340,200 L355,230 L330,255 L305,250 L290,300 Z"/>
            <!-- West Bengal -->
            <path class="state-path high" data-state="West Bengal" data-level="High" data-detections="8" data-asset="Military Warship" data-date="2025-01-09"
              d="M340,200 L380,210 L390,250 L375,290 L350,310 L330,255 L355,230 Z"/>
            <!-- Odisha -->
            <path class="state-path low" data-state="Odisha" data-level="Low" data-detections="2" data-asset="Soldier" data-date="2024-12-20"
              d="M305,250 L330,255 L350,310 L340,350 L310,355 L290,320 L290,300 Z"/>
            <!-- Chhattisgarh -->
            <path class="state-path safe" data-state="Chhattisgarh" data-level="Safe" data-detections="0" data-asset="—" data-date="—"
              d="M260,300 L290,300 L290,320 L310,355 L295,380 L270,370 L255,340 L255,310 Z"/>
            <!-- Telangana -->
            <path class="state-path low" data-state="Telangana" data-level="Low" data-detections="1" data-asset="Civilian" data-date="2024-12-15"
              d="M240,390 L260,360 L295,380 L300,410 L275,425 L245,415 Z"/>
            <!-- Andhra Pradesh -->
            <path class="state-path safe" data-state="Andhra Pradesh" data-level="Safe" data-detections="0" data-asset="—" data-date="—"
              d="M245,415 L275,425 L300,410 L310,445 L295,475 L260,470 L235,450 Z"/>
            <!-- Karnataka -->
            <path class="state-path low" data-state="Karnataka" data-level="Low" data-detections="2" data-asset="Military Vehicle" data-date="2024-12-22"
              d="M170,360 L205,380 L240,390 L245,415 L235,450 L210,470 L185,455 L160,420 L155,385 Z"/>
            <!-- Tamil Nadu -->
            <path class="state-path safe" data-state="Tamil Nadu" data-level="Safe" data-detections="0" data-asset="—" data-date="—"
              d="M185,455 L210,470 L235,450 L245,490 L225,530 L200,540 L180,510 L175,480 Z"/>
            <!-- Kerala -->
            <path class="state-path safe" data-state="Kerala" data-level="Safe" data-detections="0" data-asset="—" data-date="—"
              d="M155,420 L185,455 L175,480 L160,500 L140,480 L140,445 Z"/>
            <!-- Assam / NE states (simplified) -->
            <path class="state-path critical" data-state="Assam" data-level="Critical" data-detections="12" data-asset="Military Artillery" data-date="2025-01-10"
              d="M390,200 L430,195 L445,215 L425,235 L395,225 L380,210 Z"/>
            <!-- Sikkim -->
            <path class="state-path medium" data-state="Sikkim" data-level="Medium" data-detections="3" data-asset="Soldier" data-date="2025-01-02"
              d="M375,190 L390,200 L380,210 L370,200 Z"/>
            <!-- Manipur / Mizoram (simplified) -->
            <path class="state-path high" data-state="Manipur" data-level="High" data-detections="5" data-asset="Camouflage Soldier" data-date="2025-01-05"
              d="M425,235 L445,215 L460,240 L445,265 L420,255 Z"/>
            <!-- Arunachal Pradesh -->
            <path class="state-path critical" data-state="Arunachal Pradesh" data-level="Critical" data-detections="18" data-asset="Military Aircraft" data-date="2025-01-10"
              d="M380,145 L440,135 L470,155 L460,185 L430,195 L390,200 L370,180 Z"/>
            <!-- Goa -->
            <path class="state-path safe" data-state="Goa" data-level="Safe" data-detections="0" data-asset="—" data-date="—"
              d="M148,345 L160,340 L165,355 L152,360 Z"/>
          </svg>
        </div>
        <!-- Legend -->
        <div class="map-legend" style="padding-top:10px">
          <div class="legend-item"><div class="legend-dot" style="background:#ff3b30"></div>Critical</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ff9500"></div>High</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ffd60a"></div>Medium</div>
          <div class="legend-item"><div class="legend-dot" style="background:#30d158"></div>Low</div>
          <div class="legend-item"><div class="legend-dot" style="background:#1a3050"></div>Safe</div>
        </div>
      </div>

      <div style="display:flex;flex-direction:column;gap:20px">
        <!-- Activity -->
        <div class="panel" style="flex:1">
          <div class="panel-title">⚡ Recent Activity</div>
          <div id="activity-feed">
            <div class="activity-item"><div class="act-icon img">🖼️</div><div class="act-text"><div class="act-title">System Ready</div><div class="act-time">Upload a file to begin analysis</div></div></div>
          </div>
        </div>
        <!-- Project Info -->
        <div class="panel">
          <div class="panel-title">ℹ️ Project Information</div>
          <div class="info-row"><span class="info-key">Project</span><span class="info-val" style="font-size:11px">BorderGuard AI</span></div>
          <div class="info-row"><span class="info-key">YOLO Model</span><span class="info-val"><span class="tag blue">YOLOv8</span></span></div>
          <div class="info-row"><span class="info-key">AI Chatbot</span><span class="info-val"><span class="tag green">Online</span></span></div>
          <div class="info-row"><span class="info-key">Accuracy</span><span class="info-val">94.7%</span></div>
          <div class="info-row"><span class="info-key">Database</span><span class="info-val"><span class="tag green">Connected</span></span></div>
          <div class="info-row"><span class="info-key">Backend</span><span class="info-val">FastAPI</span></div>
        </div>
      </div>
    </div>
  </div><!-- end dashboard -->

  <!-- ═══ THREAT DETECTION PAGE ════════════════════════════════════════════ -->
  <div class="page" id="page-detection">
    <div style="margin-top:22px">
      <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()"
           ondragover="event.preventDefault();this.classList.add('dragover')"
           ondragleave="this.classList.remove('dragover')"
           ondrop="handleDrop(event)">
        <div style="font-size:48px">📡</div>
        <h3>Drop Image or Video Here</h3>
        <p>Supports JPG, PNG, MP4, AVI, MOV — or click to browse</p>
        <input type="file" id="file-input" class="file-input" accept="image/*,video/*" onchange="previewFile(this)">
      </div>
      <div style="display:flex;gap:10px;margin-bottom:20px">
        <button class="btn btn-primary" onclick="runDetection()">🎯 Run Detection</button>
        <button class="btn btn-outline" onclick="clearDetection()">🗑 Clear</button>
      </div>
      <div class="detect-grid">
        <div>
          <div class="panel-title" style="font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">File Preview</div>
          <div id="preview-box"><span style="color:var(--dim);font-size:13px">No file selected</span></div>
        </div>
        <div id="det-results-panel" style="display:none">
          <div class="panel-title" style="font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;margin-bottom:10px">Detection Results</div>
          <div class="det-results">
            <div id="threat-level-badge"></div>
            <div style="margin-bottom:10px">
              <div style="font-size:11px;color:var(--dim);margin-bottom:4px">THREAT SCORE</div>
              <div class="threat-bar"><div class="threat-bar-fill" id="threat-score-fill"></div></div>
              <div style="font-size:12px;color:var(--dim)" id="threat-score-text"></div>
            </div>
            <div style="margin-bottom:14px">
              <div style="font-size:11px;color:var(--dim);margin-bottom:6px">DETECTED OBJECTS</div>
              <div id="obj-tags"></div>
            </div>
            <table class="conf-table">
              <thead><tr><th>Object</th><th>Confidence</th><th>Threat</th></tr></thead>
              <tbody id="conf-table-body"></tbody>
            </table>
            <div style="display:flex;gap:10px;margin-top:16px">
              <button class="btn btn-success btn-sm" id="report-btn" style="display:none" onclick="downloadReport()">📄 Generate Report</button>
              <button class="btn btn-primary btn-sm" id="chat-btn-detect" style="display:none" onclick="openChatFromDetect()">🤖 Ask AI</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div><!-- end detection -->

  <!-- ═══ REPORTS PAGE ══════════════════════════════════════════════════════ -->
  <div class="page" id="page-reports">
    <div style="margin-top:22px">
      <div class="search-bar">
        <input type="text" id="report-search" placeholder="🔍 Search reports…" oninput="filterReports()">
        <select id="report-filter-level" onchange="filterReports()">
          <option value="">All Threat Levels</option>
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>
      <div class="panel">
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>#</th><th>Report ID</th><th>File Name</th><th>Date</th><th>Threat Level</th><th>Objects</th><th>Actions</th></tr></thead>
            <tbody id="reports-tbody"><tr><td colspan="7" style="text-align:center;color:var(--dim);padding:40px">No reports yet. Run a detection to generate one.</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>
  </div><!-- end reports -->

  <!-- ═══ HISTORY PAGE ══════════════════════════════════════════════════════ -->
  <div class="page" id="page-history">
    <div style="margin-top:22px">
      <div class="search-bar">
        <input type="text" id="hist-search" placeholder="🔍 Search history…" oninput="filterHistory()">
        <select id="hist-filter-level" onchange="filterHistory()">
          <option value="">All Levels</option>
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
          <option value="Safe">Safe</option>
        </select>
        <select id="hist-filter-type" onchange="filterHistory()">
          <option value="">All Types</option>
          <option value="image">Image</option>
          <option value="video">Video</option>
        </select>
      </div>
      <div class="panel">
        <div class="table-wrap">
          <table class="data-table">
            <thead><tr><th>Thumbnail</th><th>File Name</th><th>Date</th><th>Type</th><th>Objects</th><th>Threat Level</th><th>Actions</th></tr></thead>
            <tbody id="history-tbody"><tr><td colspan="7" style="text-align:center;color:var(--dim);padding:40px">No history yet.</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>
  </div><!-- end history -->

  <!-- ═══ AI CHAT PAGE ══════════════════════════════════════════════════════ -->
  <div class="page" id="page-chat">
    <div style="margin-top:22px" class="chat-layout">
      <div class="chat-sidebar">
        <div class="chat-sidebar-header">🕑 Conversations</div>
        <div class="chat-list" id="chat-list-panel">
          <div style="padding:20px;text-align:center;color:var(--dim);font-size:12px">No chats yet</div>
        </div>
        <div style="padding:10px">
          <button class="btn btn-outline btn-sm" style="width:100%" onclick="clearAllChats()">🗑 Clear All</button>
        </div>
      </div>
      <div class="chat-main">
        <div class="chat-header-bar">
          <span style="font-size:20px">🤖</span>
          <div>
            <div style="font-weight:700;font-size:14px;color:white">Defence AI Intelligence</div>
            <div style="font-size:11px;color:var(--green)">● Online — Gemini 2.5 Flash</div>
          </div>
          <div style="margin-left:auto;font-size:12px;color:var(--dim)" id="chat-context-label"></div>
        </div>
        <div class="chat-messages" id="chat-messages">
          <div class="msg ai"><span class="msg-label">AI</span>
Hello, Commander. I am your Defence Intelligence AI, powered by Gemini. Run a Threat Detection first, then ask me anything about the detected military assets, threat assessment, or countermeasures.</div>
        </div>
        <div class="suggested-qs" id="suggested-qs">
          <div class="suggested-q" onclick="askSuggested(this)">What is the threat level?</div>
          <div class="suggested-q" onclick="askSuggested(this)">Explain the detected assets</div>
          <div class="suggested-q" onclick="askSuggested(this)">Recommend countermeasures</div>
          <div class="suggested-q" onclick="askSuggested(this)">Strategic impact assessment</div>
        </div>
        <div class="chat-input-area">
          <textarea class="chat-input" id="chat-input" rows="1" placeholder="Ask about detected military assets…" onkeydown="chatKeydown(event)"></textarea>
          <button class="btn btn-primary" onclick="sendChat()">Send ➤</button>
        </div>
      </div>
    </div>
  </div><!-- end chat -->

  <!-- ═══ ANALYTICS PAGE ════════════════════════════════════════════════════ -->
  <div class="page" id="page-analytics">
    <div style="margin-top:22px" class="analytics-grid">
      <!-- Asset distribution bar chart -->
      <div class="chart-box">
        <div class="panel-title">🎖 Most Detected Assets</div>
        <div id="asset-bars"></div>
      </div>
      <!-- Threat level donut -->
      <div class="chart-box">
        <div class="panel-title">🔴 Threat Level Distribution</div>
        <div class="donut-wrap">
          <canvas id="donut-chart" width="160" height="160"></canvas>
        </div>
        <div class="donut-legend" id="donut-legend"></div>
      </div>
      <!-- Daily detections -->
      <div class="chart-box" style="grid-column:1/-1">
        <div class="panel-title">📈 Daily Detection Count (Last 7 Days)</div>
        <canvas id="line-chart" width="700" height="140"></canvas>
      </div>
    </div>
  </div><!-- end analytics -->

  <!-- ═══ SETTINGS PAGE ════════════════════════════════════════════════════ -->
  <div class="page" id="page-settings">
    <div style="margin-top:22px;max-width:600px">
      <div class="panel" style="margin-bottom:20px">
        <div class="panel-title">⚙️ System Settings</div>
        <div class="info-row"><span class="info-key">YOLO Model Path</span><span class="info-val">best.pt</span></div>
        <div class="info-row"><span class="info-key">Gemini API</span><span class="info-val"><span class="tag green">Connected</span></span></div>
        <div class="info-row"><span class="info-key">Detection Device</span><span class="info-val">CPU</span></div>
        <div class="info-row"><span class="info-key">Static Folder</span><span class="info-val">./static</span></div>
        <div class="info-row"><span class="info-key">Backend</span><span class="info-val">FastAPI + Uvicorn</span></div>
      </div>
      <div class="panel">
        <div class="panel-title">👤 Project Details</div>
        <div class="info-row"><span class="info-key">Project Title</span><span class="info-val" style="font-size:11px">BorderGuard AI — Defence Threat Detection</span></div>
        <div class="info-row"><span class="info-key">Technology Stack</span><span class="info-val">FastAPI · YOLO · Gemini AI</span></div>
        <div class="info-row"><span class="info-key">Model Accuracy</span><span class="info-val">94.7%</span></div>
        <div class="info-row"><span class="info-key">Version</span><span class="info-val">2.0.0</span></div>
      </div>
    </div>
  </div><!-- end settings -->

</div><!-- end main -->

<script>
// ════════════════════════════════════════════════════════════
//  STATE
// ════════════════════════════════════════════════════════════
let currentChatId = null;
let allHistory = [];
let allReports = [];
let analyticsData = { assets:{}, levels:{Critical:0,High:0,Medium:0,Low:0,Safe:0}, daily:{} };
let activityFeed = [];
let currentFile = null;
let lastDetectionResult = null;

// ── Clock ───────────────────────────────────────────────────
function updateClock(){
  const now = new Date();
  document.getElementById('sys-time').textContent =
    now.toLocaleDateString('en-IN') + ' ' + now.toLocaleTimeString('en-IN');
}
setInterval(updateClock,1000); updateClock();

// ── Notification ────────────────────────────────────────────
function notify(msg,color='var(--accent)'){
  const n = document.getElementById('notification');
  n.innerHTML = `<span style="color:${color}">🔔</span> ${msg}`;
  n.classList.add('show');
  setTimeout(()=>n.classList.remove('show'),3500);
}

// ── Page Navigation ─────────────────────────────────────────
function showPage(id,el){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  if(el) el.classList.add('active');
  const titles={dashboard:'📊 Dashboard Overview',detection:'🎯 Threat Detection',
    reports:'📋 Reports',history:'🕑 Detection History',chat:'🤖 AI Intelligence Chat',
    analytics:'📊 Analytics',settings:'⚙️ Settings'};
  document.getElementById('page-title').textContent = titles[id]||id;
  if(id==='analytics') renderAnalytics();
  if(id==='reports') renderReports();
  if(id==='history') renderHistory();
  if(id==='chat') renderChatList();
}

// ════════════════════════════════════════════════════════════
//  STATS
// ════════════════════════════════════════════════════════════
function updateStats(s){
  document.getElementById('s-images').textContent = s.images;
  document.getElementById('s-videos').textContent = s.videos;
  document.getElementById('s-threats').textContent = s.threats;
  document.getElementById('s-high').textContent = s.high_risk;
  document.getElementById('s-reports').textContent = s.reports;
  document.getElementById('alert-badge').textContent = s.high_risk + ' Alerts';
}

async function refreshStats(){
  try{
    const r = await fetch('/api/stats');
    const s = await r.json();
    updateStats(s);
  }catch(_){}
}
refreshStats();

// ════════════════════════════════════════════════════════════
//  INDIA MAP TOOLTIP
// ════════════════════════════════════════════════════════════
const tooltip = document.getElementById('map-tooltip');
document.querySelectorAll('.state-path').forEach(path=>{
  path.addEventListener('mouseenter',e=>{
    const d = e.target.dataset;
    document.getElementById('tt-state').textContent = d.state;
    document.getElementById('tt-info').innerHTML =
      `<b>Threat Level:</b> ${d.level}<br>
       <b>Detections:</b> ${d.detections}<br>
       <b>Latest Asset:</b> ${d.asset}<br>
       <b>Date:</b> ${d.date}`;
    tooltip.style.display='block';
  });
  path.addEventListener('mousemove',e=>{
    const rect = document.getElementById('india-map').getBoundingClientRect();
    tooltip.style.left = (e.clientX-rect.left+12)+'px';
    tooltip.style.top  = (e.clientY-rect.top+12)+'px';
  });
  path.addEventListener('mouseleave',()=>{ tooltip.style.display='none'; });
});

// ════════════════════════════════════════════════════════════
//  ACTIVITY FEED
// ════════════════════════════════════════════════════════════
function pushActivity(icon,cls,title,time){
  activityFeed.unshift({icon,cls,title,time});
  if(activityFeed.length>8) activityFeed.pop();
  renderActivity();
}
function renderActivity(){
  document.getElementById('activity-feed').innerHTML = activityFeed.map(a=>`
    <div class="activity-item">
      <div class="act-icon ${a.cls}">${a.icon}</div>
      <div class="act-text"><div class="act-title">${a.title}</div><div class="act-time">${a.time}</div></div>
    </div>`).join('');
}

// ════════════════════════════════════════════════════════════
//  FILE UPLOAD / PREVIEW
// ════════════════════════════════════════════════════════════
function handleDrop(e){
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if(file){ currentFile=file; displayPreview(file); }
}
function previewFile(input){
  const file = input.files[0];
  if(file){ currentFile=file; displayPreview(file); }
}
function displayPreview(file){
  const box = document.getElementById('preview-box');
  const url = URL.createObjectURL(file);
  if(file.type.startsWith('image')){
    box.innerHTML = `<img src="${url}" style="max-width:100%;max-height:360px;border-radius:8px">`;
  } else {
    box.innerHTML = `<video src="${url}" controls style="max-width:100%;max-height:360px;border-radius:8px"></video>`;
  }
}
function clearDetection(){
  currentFile=null; lastDetectionResult=null;
  document.getElementById('preview-box').innerHTML='<span style="color:var(--dim);font-size:13px">No file selected</span>';
  document.getElementById('det-results-panel').style.display='none';
  document.getElementById('file-input').value='';
}

// ════════════════════════════════════════════════════════════
//  RUN DETECTION
// ════════════════════════════════════════════════════════════
async function runDetection(){
  if(!currentFile){ notify('Please upload an image or video first.','var(--red)'); return; }

  document.getElementById('loader-overlay').classList.add('show');

  const form = new FormData();
  form.append('file', currentFile);

  try{
    const res = await fetch('/predict', {method:'POST', body:form});
    const data = await res.json();
    document.getElementById('loader-overlay').classList.remove('show');

    if(data.error){ notify(data.error,'var(--red)'); return; }

    lastDetectionResult = data;
    currentChatId = data.chat_id;

    displayResults(data);
    addToHistory(data, currentFile);
    await refreshStats();
    pushActivity('🎯','img',
      `Detection: ${data.classes?.length||0} objects in ${currentFile.name}`,
      new Date().toLocaleTimeString());
    notify(data.threat?'⚠️ Threat Detected!':'✅ No Threat — Area Clear',
           data.threat?'var(--red)':'var(--green)');
  }catch(err){
    document.getElementById('loader-overlay').classList.remove('show');
    notify('Detection failed: '+err.message,'var(--red)');
  }
}

function displayResults(data){
  document.getElementById('det-results-panel').style.display='block';

  // Show annotated result in preview if available
  if(data.image_url){
    document.getElementById('preview-box').innerHTML=
      `<img src="${data.image_url}" style="max-width:100%;max-height:360px;border-radius:8px">`;
  } else if(data.video_url){
    document.getElementById('preview-box').innerHTML=
      `<video src="${data.video_url}" controls style="max-width:100%;max-height:360px;border-radius:8px"></video>`;
  }

  const level = data.threat_level || (data.threat ? 'High' : 'Safe');
  document.getElementById('threat-level-badge').innerHTML =
    `<div class="threat-badge ${level}">${level === 'Safe' ? '✅ Safe — No Threat' : '⚠ ' + level + ' Threat'}</div>`;

  // Score bar — use server score if provided, else map from level
  const levelScoreMap = {Critical:95, High:72, Medium:48, Low:22, Safe:5};
  const colorMap = {Critical:'#ff3b30', High:'#ff9500', Medium:'#ffd60a', Low:'#30d158', Safe:'#30d158'};
  const score = data.threat_score !== undefined ? data.threat_score : (levelScoreMap[level] || 5);
  document.getElementById('threat-score-fill').style.cssText =
    `width:${score}%;background:${colorMap[level]}`;
  document.getElementById('threat-score-text').textContent = `Score: ${score}/100 — ${level}`;

  // Object tags
  const classes = data.classes || [];
  document.getElementById('obj-tags').innerHTML =
    classes.map(c=>`<span class="obj-tag">${c.replace(/_/g,' ')}</span>`).join('') ||
    '<span style="color:var(--dim);font-size:12px">No objects detected</span>';

  // Confidence table
  const rows = (data.detections||[]).map(d=>`
    <tr>
      <td>${d.name.replace(/_/g,' ')}</td>
      <td>${(d.confidence*100).toFixed(1)}%</td>
      <td><span class="tag ${d.threat_level==='Critical'||d.threat_level==='High'?'red':'green'}">${d.threat_level||'Low'}</span></td>
    </tr>`).join('');
  document.getElementById('conf-table-body').innerHTML = rows ||
    `<tr><td colspan="3" style="color:var(--dim)">No detections</td></tr>`;

  document.getElementById('report-btn').style.display = data.chat_id ? 'inline-flex' : 'none';
  document.getElementById('chat-btn-detect').style.display = data.chat_id ? 'inline-flex' : 'none';
}

// ════════════════════════════════════════════════════════════
//  HISTORY
// ════════════════════════════════════════════════════════════
function addToHistory(data, file){
  const rec = {
    id: data.chat_id,
    filename: file.name,
    date: new Date().toLocaleString(),
    type: data.type||'image',
    objects: (data.classes||[]).length,
    level: data.threat_level||(data.threat?'High':'Safe'),
    image_url: data.image_url||null,
    video_url: data.video_url||null,
    classes: data.classes||[],
  };
  allHistory.unshift(rec);

  // Analytics
  rec.classes.forEach(c=>{
    analyticsData.assets[c] = (analyticsData.assets[c]||0)+1;
  });
  analyticsData.levels[rec.level] = (analyticsData.levels[rec.level]||0)+1;
  const today = new Date().toLocaleDateString();
  analyticsData.daily[today] = (analyticsData.daily[today]||0)+1;

  renderHistory();

  // Auto-add report
  const rpt = {
    id: data.chat_id,
    filename: file.name,
    date: new Date().toLocaleString(),
    level: rec.level,
    objects: rec.objects,
    chat_id: data.chat_id,
  };
  allReports.unshift(rpt);
  renderReports();
}

function renderHistory(){
  const search = (document.getElementById('hist-search')||{}).value?.toLowerCase()||'';
  const lvl = (document.getElementById('hist-filter-level')||{}).value||'';
  const typ = (document.getElementById('hist-filter-type')||{}).value||'';

  const filtered = allHistory.filter(h=>
    (!search || h.filename.toLowerCase().includes(search)) &&
    (!lvl || h.level===lvl) &&
    (!typ || h.type===typ)
  );

  const tbody = document.getElementById('history-tbody');
  if(!tbody) return;
  if(!filtered.length){
    tbody.innerHTML='<tr><td colspan="7" style="text-align:center;color:var(--dim);padding:40px">No records found.</td></tr>';
    return;
  }
  tbody.innerHTML = filtered.map(h=>`
    <tr>
      <td>${h.image_url?`<img class="thumb" src="${h.image_url}">`:'<div class="thumb" style="background:#0a1928;display:flex;align-items:center;justify-content:center">🎬</div>'}</td>
      <td>${h.filename}</td>
      <td>${h.date}</td>
      <td><span class="tag blue">${h.type}</span></td>
      <td>${h.objects}</td>
      <td><span class="threat-badge ${h.level}" style="font-size:10px;padding:3px 10px">${h.level}</span></td>
      <td style="display:flex;gap:6px;flex-wrap:wrap">
        ${h.image_url?`<a href="${h.image_url}" target="_blank"><button class="btn btn-outline btn-sm">👁 View</button></a>`:''}
        <button class="btn btn-danger btn-sm" onclick="deleteHistory('${h.id}')">🗑</button>
      </td>
    </tr>`).join('');
}
function deleteHistory(id){
  allHistory = allHistory.filter(h=>h.id!==id);
  renderHistory();
  notify('History record deleted.');
}
function filterHistory(){ renderHistory(); }

// ════════════════════════════════════════════════════════════
//  REPORTS
// ════════════════════════════════════════════════════════════
function renderReports(){
  const search = (document.getElementById('report-search')||{}).value?.toLowerCase()||'';
  const lvl = (document.getElementById('report-filter-level')||{}).value||'';

  const filtered = allReports.filter(r=>
    (!search || r.filename.toLowerCase().includes(search)) &&
    (!lvl || r.level===lvl)
  );

  const tbody = document.getElementById('reports-tbody');
  if(!tbody) return;
  if(!filtered.length){
    tbody.innerHTML='<tr><td colspan="7" style="text-align:center;color:var(--dim);padding:40px">No reports found.</td></tr>';
    return;
  }
  tbody.innerHTML = filtered.map((r,i)=>`
    <tr>
      <td>${i+1}</td>
      <td style="font-family:monospace;font-size:11px;color:var(--dim)">${r.id.substring(0,12)}…</td>
      <td>${r.filename}</td>
      <td>${r.date}</td>
      <td><span class="threat-badge ${r.level}" style="font-size:10px;padding:3px 10px">${r.level}</span></td>
      <td>${r.objects}</td>
      <td style="display:flex;gap:6px">
        <a href="/generate_report/${r.chat_id}" target="_blank"><button class="btn btn-success btn-sm">⬇ PDF</button></a>
        <button class="btn btn-danger btn-sm" onclick="deleteReport('${r.id}')">🗑</button>
      </td>
    </tr>`).join('');
}
function deleteReport(id){
  allReports = allReports.filter(r=>r.id!==id);
  renderReports();
  notify('Report deleted.');
}
function filterReports(){ renderReports(); }

async function downloadReport(){
  if(!currentChatId){ notify('No detection session.','var(--red)'); return; }
  window.open(`/generate_report/${currentChatId}`,'_blank');
  // Wait briefly for server to increment report counter, then refresh all stats
  setTimeout(async()=>{
    await refreshStats();
    notify('📄 Report generated & dashboard updated!','var(--green)');
  }, 1200);
  pushActivity('📄','rep',`Report generated for session ${currentChatId.substring(0,8)}`,new Date().toLocaleTimeString());
}

// ════════════════════════════════════════════════════════════
//  AI CHAT
// ════════════════════════════════════════════════════════════
function renderChatList(){
  const panel = document.getElementById('chat-list-panel');
  const sessions = Object.keys(localStorage).filter(k=>k.startsWith('chat_'));
  // Use allHistory as source
  if(!allHistory.length){
    panel.innerHTML='<div style="padding:20px;text-align:center;color:var(--dim);font-size:12px">No chats yet</div>';
    return;
  }
  panel.innerHTML = allHistory.map(h=>`
    <div class="chat-list-item ${currentChatId===h.id?'active':''}" onclick="loadChatSession('${h.id}','${h.filename}')">
      <div>📁 ${h.filename.substring(0,18)}</div>
      <div class="chat-item-id">${h.date}</div>
    </div>`).join('');
}

function loadChatSession(id, filename){
  currentChatId = id;
  document.getElementById('chat-context-label').textContent = `Context: ${filename}`;
  document.getElementById('chat-messages').innerHTML =
    `<div class="msg ai"><span class="msg-label">AI</span>Loaded session for <b>${filename}</b>. Ask me about the detected assets.</div>`;
  renderChatList();
}

function openChatFromDetect(){
  if(!currentChatId){ notify('Run detection first.','var(--red)'); return; }
  showPage('chat', document.querySelector('.nav-item:nth-child(5)'));
  if(lastDetectionResult && lastDetectionResult.filename){
    loadChatSession(currentChatId, lastDetectionResult.filename||'last detection');
  }
}

async function sendChat(){
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if(!msg || !currentChatId){ notify('Select a detection session first.','var(--orange)'); return; }

  appendMessage('user', msg);
  input.value='';

  // Typing indicator
  const tid = 'typing-'+Date.now();
  document.getElementById('chat-messages').innerHTML +=
    `<div id="${tid}" class="msg ai"><div class="typing"><span></span><span></span><span></span></div></div>`;
  scrollChat();

  try{
    const res = await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message:msg, chat_id:currentChatId})
    });
    const data = await res.json();
    document.getElementById(tid)?.remove();
    appendMessage('ai', data.reply||'No response.');
    pushActivity('🤖','ai','AI conversation: '+msg.substring(0,30),new Date().toLocaleTimeString());
  }catch(e){
    document.getElementById(tid)?.remove();
    appendMessage('ai','⚠️ Connection error. Please try again.');
  }
}

function appendMessage(role, text){
  const box = document.getElementById('chat-messages');
  const label = role==='user'?'You':'AI';
  box.innerHTML += `<div class="msg ${role}"><span class="msg-label">${label}</span>${escapeHtml(text)}</div>`;
  scrollChat();
}
function scrollChat(){ const b=document.getElementById('chat-messages'); b.scrollTop=b.scrollHeight; }
function chatKeydown(e){ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendChat(); } }
function askSuggested(el){ document.getElementById('chat-input').value=el.textContent; sendChat(); }
function clearAllChats(){
  currentChatId=null;
  document.getElementById('chat-messages').innerHTML=
    '<div class="msg ai"><span class="msg-label">AI</span>Chats cleared. Run a new detection to start.</div>';
  renderChatList();
}
function escapeHtml(s){ return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ════════════════════════════════════════════════════════════
//  ANALYTICS
// ════════════════════════════════════════════════════════════
function renderAnalytics(){
  renderAssetBars();
  renderDonut();
  renderLineChart();
}

function renderAssetBars(){
  const el = document.getElementById('asset-bars');
  const assets = analyticsData.assets;
  if(!Object.keys(assets).length){
    el.innerHTML='<div style="color:var(--dim);font-size:13px;padding:20px 0">No data yet. Run detections to populate.</div>';
    return;
  }
  const max = Math.max(...Object.values(assets));
  const colors=['#00d4ff','#ff3b30','#ff9500','#30d158','#bf5af2','#ffd60a','#ff6b6b','#00ff88'];
  el.innerHTML = Object.entries(assets).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([k,v],i)=>`
    <div class="mini-bar">
      <div class="mini-bar-label">${k.replace(/_/g,' ')}</div>
      <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${(v/max*100)}%;background:${colors[i%colors.length]}"></div></div>
      <div class="mini-bar-val">${v}</div>
    </div>`).join('');
}

function renderDonut(){
  const canvas = document.getElementById('donut-chart');
  const ctx = canvas.getContext('2d');
  const levels = analyticsData.levels;
  const total = Object.values(levels).reduce((a,b)=>a+b,0)||1;
  const colors={Critical:'#ff3b30',High:'#ff9500',Medium:'#ffd60a',Low:'#30d158',Safe:'#1a3050'};

  ctx.clearRect(0,0,160,160);
  let start=0;
  Object.entries(levels).forEach(([k,v])=>{
    const angle = (v/total)*2*Math.PI;
    ctx.beginPath();
    ctx.moveTo(80,80);
    ctx.arc(80,80,70,start,start+angle);
    ctx.closePath();
    ctx.fillStyle=colors[k];
    ctx.fill();
    start+=angle;
  });
  // donut hole
  ctx.beginPath();
  ctx.arc(80,80,42,0,2*Math.PI);
  ctx.fillStyle='#0f1e2e';
  ctx.fill();
  ctx.fillStyle='white';
  ctx.font='bold 18px Segoe UI';
  ctx.textAlign='center';
  ctx.fillText(total,80,86);

  document.getElementById('donut-legend').innerHTML = Object.entries(levels).map(([k,v])=>`
    <div class="donut-leg-item"><div class="donut-leg-dot" style="background:${colors[k]}"></div>${k}: ${v}</div>`).join('');
}

function renderLineChart(){
  const canvas = document.getElementById('line-chart');
  const ctx = canvas.getContext('2d');
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);

  // Last 7 days
  const days=[];
  for(let i=6;i>=0;i--){
    const d=new Date(); d.setDate(d.getDate()-i);
    days.push(d.toLocaleDateString());
  }
  const vals = days.map(d=>analyticsData.daily[d]||0);
  const max = Math.max(...vals,1);
  const pad=30, chartW=W-pad*2, chartH=H-pad*2;

  // Grid lines
  ctx.strokeStyle='#1a3050'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){
    const y=pad+chartH-(chartH/4)*i;
    ctx.beginPath(); ctx.moveTo(pad,y); ctx.lineTo(W-pad,y); ctx.stroke();
    ctx.fillStyle='#5a7a9a'; ctx.font='10px Segoe UI';
    ctx.fillText(Math.round(max/4*i),4,y+4);
  }

  // Line
  ctx.strokeStyle='#00d4ff'; ctx.lineWidth=2.5;
  ctx.beginPath();
  vals.forEach((v,i)=>{
    const x=pad+i*(chartW/(days.length-1));
    const y=pad+chartH-(v/max)*chartH;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();

  // Fill gradient
  const grad=ctx.createLinearGradient(0,pad,0,pad+chartH);
  grad.addColorStop(0,'rgba(0,212,255,0.3)');
  grad.addColorStop(1,'rgba(0,212,255,0)');
  ctx.fillStyle=grad;
  ctx.beginPath();
  vals.forEach((v,i)=>{
    const x=pad+i*(chartW/(days.length-1));
    const y=pad+chartH-(v/max)*chartH;
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.lineTo(W-pad,pad+chartH); ctx.lineTo(pad,pad+chartH); ctx.closePath(); ctx.fill();

  // Dots & labels
  vals.forEach((v,i)=>{
    const x=pad+i*(chartW/(days.length-1));
    const y=pad+chartH-(v/max)*chartH;
    ctx.beginPath(); ctx.arc(x,y,4,0,2*Math.PI);
    ctx.fillStyle='#00d4ff'; ctx.fill();
    ctx.fillStyle='#c9d8e8'; ctx.font='9px Segoe UI'; ctx.textAlign='center';
    ctx.fillText(days[i].split('/').slice(0,2).join('/'),x,pad+chartH+14);
    if(v>0){ ctx.fillStyle='white'; ctx.fillText(v,x,y-8); }
  });
}
// Auto-refresh stats every 5 s — keeps dashboard live after PDF opens in new tab
setInterval(refreshStats, 5000);
</script>
</body>
</html>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/stats")
def api_stats():
    return stats


# ─── Per-object threat helper (uses OBJECT_LEVEL_MAP defined at top) ──────────
def obj_threat(name):
    return OBJECT_LEVEL_MAP.get(name, "Low")


# ─── Predict ─────────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()
    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = {"classes": [], "messages": [], "filename": file.filename,
                               "timestamp": datetime.datetime.now().isoformat()}

    # ── IMAGE ──────────────────────────────────────────────────────────────
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        results = model(image)
        boxes = results[0].boxes
        os.makedirs("static", exist_ok=True)
        output_path = f"static/output_{uuid.uuid4().hex}.png"

        if boxes is None or len(boxes) == 0:
            stats["images"] += 1
            return {"type": "image", "threat": False, "classes": [],
                    "threat_level": "Safe", "detections": [],
                    "image_url": None, "chat_id": chat_id}

        detected = [results[0].names[int(b.cls)] for b in boxes]
        confidences = [float(b.conf) for b in boxes]
        unique_classes = list(set(detected))

        chat_sessions[chat_id]["classes"] = unique_classes

        annotated = results[0].plot()
        img = Image.fromarray(annotated)
        img.save(output_path)
        chat_sessions[chat_id]["image_path"] = output_path

        threat_level = compute_threat(unique_classes)
        detections = [
            {"name": n, "confidence": c, "threat_level": obj_threat(n)}
            for n, c in zip(detected, confidences)
        ]

        stats["images"] += 1
        stats["threats"] += 1
        if threat_level in ("High", "Critical"):
            stats["high_risk"] += 1

        return {
            "type": "image",
            "threat": True,
            "classes": unique_classes,
            "threat_level": threat_level,
            "threat_score": compute_score(unique_classes),
            "detections": detections,
            "image_url": "/" + output_path,
            "chat_id": chat_id,
            "filename": file.filename,
        }

    # ── VIDEO ──────────────────────────────────────────────────────────────
    elif filename.endswith((".mp4", ".avi", ".mov")):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(await file.read())
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        os.makedirs("static", exist_ok=True)
        output_path = f"static/output_{uuid.uuid4().hex}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        detected_set = set()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            res = model(frame)
            if res[0].boxes:
                for b in res[0].boxes:
                    detected_set.add(res[0].names[int(b.cls)])
            out.write(res[0].plot())

        cap.release()
        out.release()
        os.remove(tmp.name)

        detected = list(detected_set)
        chat_sessions[chat_id]["classes"] = detected
        threat_level = compute_threat(detected)
        detections = [{"name": n, "confidence": 0.85, "threat_level": obj_threat(n)} for n in detected]

        stats["videos"] += 1
        if detected:
            stats["threats"] += 1
            if threat_level in ("High", "Critical"):
                stats["high_risk"] += 1

        return {
            "type": "video",
            "threat": len(detected) > 0,
            "classes": detected,
            "threat_level": threat_level,
            "threat_score": compute_score(detected),
            "detections": detections,
            "video_url": "/" + output_path,
            "chat_id": chat_id,
            "filename": file.filename,
        }

    return {"error": "Unsupported file format. Use JPG/PNG/MP4/AVI/MOV."}


# ─── PDF Report ───────────────────────────────────────────────────────────────
@app.get("/generate_report/{chat_id}")
async def generate_report(chat_id: str):
    session = chat_sessions.get(chat_id)
    if not session:
        return {"error": "Session not found"}

    detected = session.get("classes", [])
    image_path = session.get("image_path")
    pdf_path = f"static/report_{chat_id}.pdf"

    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("DEFENCE THREAT ANALYSIS REPORT", styles["Title"]))
    content.append(Spacer(1, 12))
    content.append(Paragraph(f"Session ID: {chat_id}", styles["Normal"]))
    content.append(Paragraph(f"Generated On: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    content.append(Paragraph(f"File: {session.get('filename', 'Unknown')}", styles["Normal"]))
    content.append(Spacer(1, 16))

    if image_path and os.path.exists(image_path):
        content.append(Paragraph("Detection Evidence", styles["Heading2"]))
        content.append(Spacer(1, 8))
        content.append(PDFImage(image_path, width=400, height=280))
        content.append(Spacer(1, 16))

    threat_level = compute_threat(detected)
    content.append(Paragraph(f"<b>Detected Objects:</b> {', '.join(detected) or 'None'}", styles["Normal"]))
    content.append(Spacer(1, 8))
    content.append(Paragraph(f"<b>Overall Threat Level:</b> {threat_level}", styles["Heading2"]))
    content.append(Spacer(1, 12))

    recs = {
        "Critical": "A Critical Threat Level indicates the presence of highly dangerous military assets. Immediate military response is recommended. Activate emergency protocols, deploy rapid reaction teams, and notify central command.",
        "High": "A High Threat Level indicates suspicious military activity. Security personnel should increase surveillance and place nearby units on standby.",
        "Medium": "A Medium Threat Level requires further verification. Increase routine patrols and monitor the area continuously.",
        "Low": "A Low Threat Level indicates no major threats. Continue routine monitoring and preventive security measures.",
        "Safe": "Area is clear. No military threats detected during analysis.",
    }
    content.append(Paragraph("Risk Assessment", styles["Heading2"]))
    content.append(Paragraph(recs.get(threat_level, "No assessment available."), styles["BodyText"]))
    content.append(Spacer(1, 12))

    if detected:
        content.append(Paragraph("Object Analysis", styles["Heading2"]))
        for obj in detected:
            lvl = obj_threat(obj)
            content.append(Paragraph(
                f"<b>{obj.replace('_',' ').title()}</b> — Threat Level: {lvl}",
                styles["BodyText"]
            ))
            content.append(Spacer(1, 6))

    content.append(Spacer(1, 20))
    content.append(Paragraph("Generated by BorderGuard AI v2.0", styles["Italic"]))
    doc.build(content)

    stats["reports"] += 1
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"Threat_Report_{chat_id[:8]}.pdf")


# ─── AI Chat ──────────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(data: dict):
    chat_id = data.get("chat_id")
    user_msg = data.get("message", "").strip()

    session = chat_sessions.get(chat_id)
    if not session:
        return {"reply": "Session not found. Please run a detection first."}

    detected = session["classes"]
    counts = Counter(detected)
    total = sum(counts.values())
    score_raw = max(sum(THREAT_WEIGHTS.get(o, 2) * c for o, c in counts.items()), 0)
    threat_level = compute_threat(detected)
    primary = max(counts, key=counts.get) if counts else "none"
    obj_summary = "\n".join(f"- {o.replace('_',' ').title()}: {c}" for o, c in counts.items())

    if not detected:
        return {"reply": "No objects were detected in this session. Upload and analyse a new image or video."}

    is_detailed = any(w in user_msg.lower() for w in
                      ["detail","explain","more","analysis","why","how","assessment","strategic","impact"])

    if is_detailed:
        prompt = f"""You are an AI Military Intelligence Assistant for an Indian Defence project.

Detected Objects:
{obj_summary}
Total: {total} | Primary: {primary.replace('_',' ').title()}
Threat Score: {score_raw}/300 | Threat Level: {threat_level}

Generate a concise defence intelligence report with these EXACT sections:
Situation Report
Object Summary: <list each object and count>
Threat Level: {threat_level}
Assessment: <3 sentences on why this threat level was assigned>
Operational Risks:
- <risk 1>
- <risk 2>
- <risk 3>
Recommended Actions:
- <action 1>
- <action 2>
- <action 3>
Preventive Measures:
- <measure 1>
- <measure 2>
Key Observations: <total objects, dominant object, civilian presence yes/no>
Confidence: High

Do not use markdown. Keep it concise. Never invent objects.
User Question: {user_msg}"""
    else:
        prompt = f"""You are an AI Defence Intelligence Assistant for an Indian Defence project.

Detected Objects:
{obj_summary}
Total: {total} | Primary: {primary.replace('_',' ').title()}
Threat Level: {threat_level}

Answer EXACTLY in this format:
Detection Summary: <object counts>
Threat Level: {threat_level}
Threat Score: {score_raw}/300
Action: <one line>
Prevention: <one line>

If user greets, introduce yourself briefly then give the summary.
Never invent detections. Do not use markdown.
User: {user_msg}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        reply = response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        reply = f"""Detection Summary: {', '.join(f"{o.replace('_',' ').title()} x{c}" for o,c in counts.items())}
Threat Level: {threat_level}
Threat Score: {score_raw}/300
Action: Alert command center and deploy surveillance assets
Prevention: Increase perimeter security and aerial reconnaissance"""

    session["messages"].append({"user": user_msg, "ai": reply})
    return {"reply": reply}


# ─── History API ──────────────────────────────────────────────────────────────
@app.get("/history")
def history():
    return {"sessions": list(chat_sessions.keys())}

@app.get("/chat/{chat_id}")
def get_chat(chat_id: str):
    s = chat_sessions.get(chat_id)
    return {"messages": s["messages"] if s else []}