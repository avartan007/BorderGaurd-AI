# Smoke test: run detection + threat scoring + PDF on the bundled model.
#
#   python scripts/smoke_test.py [path/to/image]
#
# Uses static/*.png from the old FastAPI run when no path is given.

from __future__ import annotations

import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import threat_core as core  # noqa: E402


def main() -> int:
    model_path = ROOT / "best.pt"
    if not model_path.exists():
        print(f"FAIL: {model_path} is missing")
        return 1

    candidates = sys.argv[1:] or sorted(glob.glob(str(ROOT / "static" / "output_*.png")))
    if not candidates:
        print("FAIL: no test image supplied and none found in static/")
        return 1

    print(f"Loading {model_path.name} …")
    model = YOLO(str(model_path)).to("cpu")
    print(f"Classes: {len(model.names)}")

    hits = 0
    for path in candidates[:8]:
        image = Image.open(path).convert("RGB")
        result = core.analyze_image(model, image, conf=0.15)
        pdf = core.build_pdf(
            {
                "id": "smoke-test",
                "filename": Path(path).name,
                "type": "image",
                "classes": result["classes"],
                "counts": result["counts"],
                "threat_level": result["threat_level"],
                "threat_score": result["threat_score"],
                "annotated": result["annotated"],
            }
        )
        assert pdf.startswith(b"%PDF-"), "PDF generation failed"
        flag = "✓" if result["classes"] else "·"
        print(
            f"{flag} {Path(path).name}: {result['classes']} "
            f"level={result['threat_level']} score={result['threat_score']} "
            f"pdf={len(pdf)}B"
        )
        hits += bool(result["classes"])

    print(
        f"\nScoring checks: weapon={core.compute_score(['weapon'])} "
        f"tank={core.compute_score(['military_tank'])} "
        f"camouflage_soldier={core.compute_score(['camouflage_soldier'])} "
        f"empty={core.compute_score([])}"
    )
    assert core.compute_threat([]) == "Safe"
    assert core.compute_threat(["soldier"]) == "Medium"
    assert core.compute_threat(["soldier", "weapon"]) == "Critical"
    assert core.compute_score([]) == 0
    print(f"Result: {'PASS' if hits else 'PASS (no detections in fixtures)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
