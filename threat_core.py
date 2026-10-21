"""Detection, threat scoring, PDF and Gemini helpers for BorderGuard AI.

Kept free of Streamlit imports so it can be unit-tested and reused by a worker.
"""
from __future__ import annotations

import io
import os
from collections import Counter
from datetime import datetime
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# ── Threat model ──────────────────────────────────────────────────────────────
THREAT_WEIGHTS = {
    "weapon": 20,
    "soldier": 8,
    "military_vehicle": 25,
    "military_tank": 40,
    "military_aircraft": 50,
    "helicopter": 45,
    "camouflage_soldier": 45,
    "military_artillery": 50,
    "military_warship": 50,
    "military_truck": 30,
    "trench": 20,
    "civilian": -5,
    "civilian_vehicle": -3,
}

# The weighted formula alone makes a single high-severity object look harmless
# (one camouflage soldier => 45/3 = 15). Keep the numeric score consistent with
# the level assigned by the priority map.
LEVEL_FLOOR = {"Critical": 75, "High": 50, "Medium": 30, "Low": 12, "Safe": 0}

OBJECT_LEVEL_MAP = {
    "weapon": "Critical",
    "military_tank": "Critical",
    "military_aircraft": "Critical",
    "military_artillery": "Critical",
    "military_warship": "Critical",
    "helicopter": "High",
    "military_vehicle": "High",
    "military_truck": "High",
    "camouflage_soldier": "High",
    "soldier": "Medium",
    "trench": "Medium",
    "civilian": "Low",
    "civilian_vehicle": "Low",
}

LEVEL_RANK = {"Safe": 0, "Low": 1, "Medium": 2, "High": 3, "Critical": 4}
LEVEL_COLORS = {
    "Critical": "#ff3b30",
    "High": "#ff9500",
    "Medium": "#ffd60a",
    "Low": "#30d158",
    "Safe": "#2a4a6a",
}
LEVEL_SCORE = {"Critical": 95, "High": 72, "Medium": 48, "Low": 22, "Safe": 5}

RISK_TEXT = {
    "Critical": (
        "A Critical Threat Level indicates the presence of highly dangerous military assets. "
        "Immediate military response is recommended. Activate emergency protocols, deploy rapid "
        "reaction teams, and notify central command."
    ),
    "High": (
        "A High Threat Level indicates suspicious military activity. Security personnel should "
        "increase surveillance and place nearby units on standby."
    ),
    "Medium": (
        "A Medium Threat Level requires further verification. Increase routine patrols and monitor "
        "the area continuously."
    ),
    "Low": (
        "A Low Threat Level indicates no major threats. Continue routine monitoring and preventive "
        "security measures."
    ),
    "Safe": "Area is clear. No military threats detected during analysis.",
}

SUPPORTED_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SUPPORTED_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def obj_threat(name: str) -> str:
    return OBJECT_LEVEL_MAP.get(name, "Low")


def compute_threat(classes: Iterable[str]) -> str:
    """Highest threat level found among detected objects."""
    best = "Safe"
    for obj in classes:
        lvl = obj_threat(obj)
        if LEVEL_RANK[lvl] > LEVEL_RANK[best]:
            best = lvl
    return best


def compute_score(classes: Iterable[str]) -> int:
    """Numeric threat score 0-100, weighted by object counts."""
    classes = list(classes)
    counts = Counter(classes)
    raw = sum(THREAT_WEIGHTS.get(o, 2) * c for o, c in counts.items())
    weighted = max(0, min(int(raw / 3), 100))
    return max(weighted, LEVEL_FLOOR[compute_threat(classes)])


def threat_color(level: str) -> str:
    return LEVEL_COLORS.get(level, LEVEL_COLORS["Safe"])


# ── Inference ─────────────────────────────────────────────────────────────────
def _annotated_rgb(result) -> np.ndarray:
    """Ultralytics plot() returns BGR; convert for PIL/Streamlit."""
    plotted = result.plot()
    return cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)


def _collect(result) -> tuple[list[str], list[float]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []
    names = result.names
    classes = [names[int(b.cls)] for b in boxes]
    confs = [float(b.conf) for b in boxes]
    return classes, confs


def _unique(seq: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def analyze_image(model, image: Image.Image, conf: float = 0.25) -> dict:
    """Run detection on one PIL image and return a UI-ready result dict."""
    result = model.predict(image.convert("RGB"), conf=conf, verbose=False)[0]
    raw_classes, confs = _collect(result)
    classes = _unique(raw_classes)
    detections = [
        {"name": n, "confidence": c, "threat_level": obj_threat(n)}
        for n, c in zip(raw_classes, confs)
    ]
    level = compute_threat(classes)
    annotated = _annotated_rgb(result) if raw_classes else np.asarray(image.convert("RGB"))
    return {
        "type": "image",
        "classes": classes,
        "detections": detections,
        "counts": dict(Counter(raw_classes)),
        "threat": bool(raw_classes),
        "threat_level": level,
        "threat_score": compute_score(raw_classes),
        "annotated": annotated,
        "width": int(annotated.shape[1]),
        "height": int(annotated.shape[0]),
    }


def analyze_video(
    model,
    path: str,
    conf: float = 0.25,
    stride: int = 5,
    max_frames: int = 48,
    progress: Callable[[float, str], None] | None = None,
) -> dict:
    """Sample frames from a video, detect on each and build an annotated clip.

    Encoding the full source video with H.264 on a shared CPU box is slow and
    fragile, so we analyse at most ``max_frames`` evenly spaced frames and encode
    those into a short annotated preview clip.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open the uploaded video.")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 20.0
    stride = max(1, stride)
    all_classes: list[str] = []
    best_conf: dict[str, float] = {}
    frames: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    idx = 0

    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            result = model.predict(frame, conf=conf, verbose=False)[0]
            classes, confs = _collect(result)
            for name, c in zip(classes, confs):
                all_classes.append(name)
                best_conf[name] = max(best_conf.get(name, 0.0), c)
            rgb = _annotated_rgb(result) if classes else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if size is None:
                size = (rgb.shape[1], rgb.shape[0])
            frames.append(rgb)
            if progress and total:
                pct = min(0.95, (idx + 1) / total)
                progress(pct, f"Analysing frame {idx + 1} of {total}")
        idx += 1
    cap.release()

    if progress:
        progress(0.97, "Encoding annotated preview clip")

    clip = encode_clip(frames, fps=max(2.0, min(12.0, fps / stride))) if frames else None

    classes = _unique(all_classes)
    detections = [
        {"name": n, "confidence": best_conf[n], "threat_level": obj_threat(n)} for n in classes
    ]
    level = compute_threat(classes)
    return {
        "type": "video",
        "classes": classes,
        "detections": detections,
        "counts": dict(Counter(all_classes)),
        "threat": bool(classes),
        "threat_level": level,
        "threat_score": compute_score(all_classes),
        "frames": frames,
        "clip": clip,
        "sampled_frames": len(frames),
        "width": size[0] if size else 0,
        "height": size[1] if size else 0,
    }


def encode_clip(frames: list[np.ndarray], fps: float) -> bytes | None:
    """Encode RGB frames to a browser-friendly H.264 MP4. Returns None on failure."""
    if not frames:
        return None
    try:
        import tempfile

        import imageio_ffmpeg

        h, w = frames[0].shape[:2]
        w -= w % 2
        h -= h % 2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            out_path = tmp.name
        writer = imageio_ffmpeg.write_frames(
            out_path,
            (w, h),
            fps=fps,
            codec="libx264",
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=2,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        for frame in frames:
            writer.send(np.ascontiguousarray(frame[:h, :w]))
        writer.close()
        with open(out_path, "rb") as fh:
            data = fh.read()
        os.unlink(out_path)
        return data
    except Exception:
        return None


# ── PDF report ────────────────────────────────────────────────────────────────
def build_pdf(record: dict) -> bytes:
    """Build the threat report PDF in memory and return the bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        title="Defence Threat Analysis Report",
        author="BorderGuard AI — Defence Threat Detection",
    )
    styles = getSampleStyleSheet()
    classes = record.get("classes", [])
    counts = Counter(record.get("counts", {}))
    level = record.get("threat_level") or compute_threat(classes)

    story = [
        Paragraph("DEFENCE THREAT ANALYSIS REPORT", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Session ID: {record.get('id', 'unknown')}", styles["Normal"]),
        Paragraph(
            f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]
        ),
        Paragraph(f"File: {record.get('filename', 'Unknown')}", styles["Normal"]),
        Paragraph(f"Detection Type: {record.get('type', 'image').title()}", styles["Normal"]),
        Spacer(1, 16),
    ]

    annotated = record.get("annotated")
    if annotated is not None:
        try:
            img = Image.fromarray(annotated) if isinstance(annotated, np.ndarray) else annotated
            img.thumbnail((900, 900))
            story += [
                Paragraph("Detection Evidence", styles["Heading2"]),
                Spacer(1, 8),
                PDFImage(ImageReader(img), width=420, height=280 * img.height / max(img.width, 1)),
                Spacer(1, 16),
            ]
        except Exception:
            pass

    summary = ", ".join(f"{k.replace('_', ' ').title()} x{v}" for k, v in counts.items()) or "None"
    story += [
        Paragraph(f"<b>Detected Objects:</b> {summary}", styles["Normal"]),
        Spacer(1, 8),
        Paragraph(f"<b>Overall Threat Level:</b> {level}", styles["Heading2"]),
        Paragraph(f"<b>Threat Score:</b> {record.get('threat_score', 0)}/100", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Risk Assessment", styles["Heading2"]),
        Paragraph(RISK_TEXT.get(level, "No assessment available."), styles["BodyText"]),
        Spacer(1, 12),
    ]

    if classes:
        story.append(Paragraph("Object Analysis", styles["Heading2"]))
        for obj in classes:
            story += [
                Paragraph(
                    f"<b>{obj.replace('_', ' ').title()}</b> — Threat Level: {obj_threat(obj)}",
                    styles["BodyText"],
                ),
                Spacer(1, 6),
            ]

    story += [
        Spacer(1, 20),
        Paragraph(
            "Generated by BorderGuard AI v2.0 — Defence Threat Detection", styles["Italic"]
        ),
    ]
    doc.build(story)
    return buf.getvalue()


# ── Gemini ────────────────────────────────────────────────────────────────────
def build_prompt(classes: list[str], user_msg: str) -> str:
    counts = Counter(classes)
    total = sum(counts.values())
    level = compute_threat(classes)
    primary = max(counts, key=counts.get) if counts else "none"
    obj_summary = "\n".join(f"- {o.replace('_', ' ').title()}: {c}" for o, c in counts.items())
    score_raw = max(sum(THREAT_WEIGHTS.get(o, 2) * c for o, c in counts.items()), 0)
    detailed = any(
        w in user_msg.lower()
        for w in ["detail", "explain", "more", "analysis", "why", "how", "assessment", "strategic", "impact"]
    )

    if detailed:
        return f"""You are an AI Military Intelligence Assistant for an Indian Defence project.

Detected Objects:
{obj_summary}
Total: {total} | Primary: {primary.replace('_', ' ').title()}
Threat Score: {score_raw}/300 | Threat Level: {level}

Generate a concise defence intelligence report with these EXACT sections:
Situation Report
Object Summary: <list each object and count>
Threat Level: {level}
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

    return f"""You are an AI Defence Intelligence Assistant for an Indian Defence project.

Detected Objects:
{obj_summary}
Total: {total} | Primary: {primary.replace('_', ' ').title()}
Threat Level: {level}

Answer EXACTLY in this format:
Detection Summary: <object counts>
Threat Level: {level}
Threat Score: {score_raw}/300
Action: <one line>
Prevention: <one line>

If user greets, introduce yourself briefly then give the summary.
Never invent detections. Do not use markdown.
User: {user_msg}"""


def fallback_reply(classes: list[str]) -> str:
    counts = Counter(classes)
    level = compute_threat(classes)
    score_raw = max(sum(THREAT_WEIGHTS.get(o, 2) * c for o, c in counts.items()), 0)
    summary = ", ".join(f"{o.replace('_', ' ').title()} x{c}" for o, c in counts.items())
    return (
        f"Detection Summary: {summary or 'None'}\n"
        f"Threat Level: {level}\n"
        f"Threat Score: {score_raw}/300\n"
        "Action: Alert command center and deploy surveillance assets\n"
        "Prevention: Increase perimeter security and aerial reconnaissance"
    )


def gemini_reply(api_key: str, classes: list[str], user_msg: str) -> tuple[str, str | None]:
    """Return (reply, error). Falls back to a deterministic summary on failure."""
    if not api_key:
        return fallback_reply(classes), "GEMINI_API_KEY is not configured"
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=build_prompt(classes, user_msg),
        )
        text = response.candidates[0].content.parts[0].text.strip()
        return text, None
    except Exception as exc:  # network/quota/model errors must not break the UI
        return fallback_reply(classes), str(exc)[:300]
