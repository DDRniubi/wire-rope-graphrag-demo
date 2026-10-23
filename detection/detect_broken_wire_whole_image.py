from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTS else []
    return sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def xyxy_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return cx, cy, bw, bh


def draw_predictions(image: np.ndarray, boxes: list[dict[str, object]]) -> np.ndarray:
    vis = image.copy()
    for item in boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in item["xyxy"]]
        conf = float(item["conf"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"broken_wire {conf:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Whole-image YOLO broken-wire detection without ROI preprocessing.")
    parser.add_argument("--source", required=True, help="Image file or folder.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--conf", type=float, default=0.25, help="Balanced default. Use 0.10 for high-recall review.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save-txt", action="store_true", help="Also save YOLO-format prediction labels.")
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out)
    image_out_dir = out_dir / "images"
    label_out_dir = out_dir / "labels"
    summary_path = out_dir / "prediction_summary.csv"

    images = collect_images(source)
    if not images:
        raise FileNotFoundError(f"No images found in {source}")

    model = YOLO(str(args.model))
    rows: list[dict[str, object]] = []

    for image_path in images:
        image = imread(image_path)
        if image is None:
            rows.append({"image": str(image_path), "status": "read_failed", "detections": 0})
            continue

        height, width = image.shape[:2]
        result = model.predict(
            source=image,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            classes=[0],
            verbose=False,
        )[0]

        detections: list[dict[str, object]] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy().tolist()
            confs = result.boxes.conf.cpu().numpy().tolist()
            clss = result.boxes.cls.cpu().numpy().astype(int).tolist()
            for box, conf, cls_id in zip(xyxy, confs, clss):
                detections.append({"xyxy": box, "conf": conf, "cls": cls_id})

        rel_name = image_path.name if source.is_file() else str(image_path.relative_to(source))
        safe_rel = Path(rel_name)
        vis = draw_predictions(image, detections)
        imwrite(image_out_dir / safe_rel, vis)

        if args.save_txt:
            txt_path = label_out_dir / safe_rel.with_suffix(".txt")
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for item in detections:
                cx, cy, bw, bh = xyxy_to_yolo(item["xyxy"], width, height)
                lines.append(f'{int(item["cls"])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(item["conf"]):.6f}')
            txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        rows.append(
            {
                "image": rel_name,
                "status": "ok",
                "detections": len(detections),
                "max_conf": f"{max((float(d['conf']) for d in detections), default=0.0):.6f}",
                "output_image": str(image_out_dir / safe_rel),
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "status", "detections", "max_conf", "output_image"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"images: {len(images)}")
    print(f"images_with_detections: {sum(1 for r in rows if int(r.get('detections') or 0) > 0)}")
    print(f"detections: {sum(int(r.get('detections') or 0) for r in rows)}")
    print(f"summary: {summary_path}")
    print(f"annotated_images: {image_out_dir}")


if __name__ == "__main__":
    main()
