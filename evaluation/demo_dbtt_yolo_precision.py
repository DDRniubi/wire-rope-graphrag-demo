"""Calibrate broken-wire precision on DBTT validation sources, then lock test."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def images(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in EXTENSIONS)


def source_key(path: Path) -> str:
    stem = path.stem.split(".rf.", 1)[0]
    return re.sub(r"-\d+$", "", stem)


def read_break_labels(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid label: {path}")
        if int(fields[0]) != 0:
            continue
        cx, cy, bw, bh = map(float, fields[1:])
        boxes.append([(cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height])
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def score(rows: list[dict], threshold: float) -> dict:
    tp = fp = fn = 0
    per_image = []
    for row in rows:
        labels = row["gt_boxes"]
        predictions = sorted((box for box in row["pred_boxes"] if box["confidence"] >= threshold),
                             key=lambda box: -box["confidence"])
        matched: set[int] = set()
        image_tp = image_fp = 0
        for prediction in predictions:
            overlaps = [(iou(prediction["xyxy"], label), index) for index, label in enumerate(labels)
                        if index not in matched]
            best = max(overlaps, default=(0.0, -1))
            if best[0] >= 0.5:
                matched.add(best[1])
                image_tp += 1
            else:
                image_fp += 1
        image_fn = len(labels) - len(matched)
        tp += image_tp
        fp += image_fp
        fn += image_fn
        per_image.append({"image": row["image"], "gt": len(labels), "pred": len(predictions),
                          "tp": image_tp, "fp": image_fp, "fn": image_fn})
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "images": per_image}


def wilson_lower(tp: int, fp: int) -> float | None:
    n = tp + fp
    if n == 0:
        return None
    p = tp / n
    z = 1.96
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def predict_split(model: YOLO, dataset: Path, split: str, paths: list[Path], device: str) -> list[dict]:
    rows = []
    for path in paths:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {path}")
        height, width = image.shape[:2]
        label = dataset / split / "labels" / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        result = model.predict(source=image, imgsz=800, conf=0.001, iou=0.7, classes=[0],
                               device=device, verbose=False)[0]
        predictions = []
        if result.boxes is not None:
            for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                predictions.append({"xyxy": [float(x) for x in box], "confidence": float(confidence)})
        rows.append({"image": path.name, "image_path": str(path), "gt_boxes": read_break_labels(label, width, height),
                     "pred_boxes": predictions})
    return rows


def format_percent(value: float | None) -> str:
    return "未定义" if value is None else f"{value * 100:.1f}%"


def draw(image: np.ndarray, boxes: list[list[float]], color: tuple[int, int, int]) -> np.ndarray:
    output = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
    return output


def save_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise ValueError(f"Could not encode {path}")
    encoded.tofile(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="DBTT YOLO broken-wire precision demo")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT.parent / "models/dbtt_yolo/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "demo_runs" / f"dbtt_precision_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--device", default="0")
    parser.add_argument("--calibration-target", type=float, default=0.90,
                        help="Minimum calibration precision; higher values reduce false positives at the cost of recall")
    args = parser.parse_args()
    if not 0 < args.calibration_target <= 1:
        raise ValueError("--calibration-target must be in (0, 1]")
    dataset, model_path, output = args.dataset.resolve(), args.model.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    train_keys = {source_key(path) for path in images(dataset / "train/images")}
    valid_keys = {source_key(path) for path in images(dataset / "valid/images")}
    calibration_paths = [path for path in images(dataset / "valid/images") if source_key(path) not in train_keys]
    test_paths = [path for path in images(dataset / "test/images") if source_key(path) not in train_keys | valid_keys]
    if not calibration_paths or not test_paths:
        raise ValueError("No source-name-screened calibration or test images")
    output.mkdir(parents=True)
    (output / "examples").mkdir()
    model = YOLO(str(model_path))
    if model.names.get(0) != "break":
        raise ValueError(f"Model class 0 is not break: {model.names}")
    calibration_rows = predict_split(model, dataset, "valid", calibration_paths, args.device)
    test_rows = predict_split(model, dataset, "test", test_paths, args.device)
    thresholds = [round(i / 100, 2) for i in range(5, 100, 5)] + [0.99]
    sweep = [score(calibration_rows, threshold) for threshold in thresholds]
    eligible = [row for row in sweep if row["precision"] is not None and row["precision"] >= args.calibration_target and row["tp"] > 0]
    if eligible:
        selected = max(eligible, key=lambda row: (row["recall"], row["tp"], row["precision"]))
        selection_rule = f"Highest calibration recall among thresholds with calibration precision >= {args.calibration_target:.2f}"
    else:
        selected = max(sweep, key=lambda row: (row["f1"], row["precision"] or 0.0))
        selection_rule = f"No threshold reached {args.calibration_target:.2f} precision on calibration; selected best F1 instead"
    test = score(test_rows, selected["threshold"])
    summary = {
        "model": str(model_path), "dataset": str(dataset), "task": "break-class bounding-box localization",
        "match_iou": 0.5, "image_size": 800,
        "calibration_split": "valid names unseen in train", "calibration_images": len(calibration_rows),
        "test_split": "test names unseen in train and valid", "test_images": len(test_rows),
        "source_screen_limitation": "Filename grouping only; checkpoint was selected using the original validation set; visual near-duplicates not fully excluded",
        "evaluation_status": "Exploratory: threshold policy was compared after viewing this test set; fresh locked test required for independent acceptance",
        "selection_rule": selection_rule, "calibration_target": args.calibration_target,
        "selected_confidence": selected["threshold"],
        "calibration": {k: v for k, v in selected.items() if k != "images"},
        "test": {**{k: v for k, v in test.items() if k != "images"},
                 "precision_wilson_95_lower": wilson_lower(test["tp"], test["fp"])},
        "test_precision_reached_90_percent": test["precision"] is not None and test["precision"] >= 0.9,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "calibration_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["threshold", "tp", "fp", "fn", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in sweep)
    with (output / "test_per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "gt", "pred", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(test["images"])
    (output / "calibration_source_names.txt").write_text("\n".join(path.name for path in calibration_paths), encoding="utf-8")
    (output / "test_source_names.txt").write_text("\n".join(path.name for path in test_paths), encoding="utf-8")
    gallery = []
    for index, row in enumerate(test_rows[:12], start=1):
        image = cv2.imdecode(np.fromfile(row["image_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        ground_truth = draw(image, row["gt_boxes"], (0, 140, 0))
        predicted = draw(image, [box["xyxy"] for box in row["pred_boxes"] if box["confidence"] >= selected["threshold"]], (0, 0, 220))
        save_image(output / "examples" / f"{index:02d}_gt.jpg", ground_truth)
        save_image(output / "examples" / f"{index:02d}_pred.jpg", predicted)
        stats = test["images"][index - 1]
        gallery.append(f"<tr><td>{html.escape(row['image'])}<br>TP {stats['tp']} / FP {stats['fp']} / FN {stats['fn']}</td>"
                       f"<td><img src='examples/{index:02d}_gt.jpg'></td><td><img src='examples/{index:02d}_pred.jpg'></td></tr>")
    status = "本次测试 precision 达到90%" if summary["test_precision_reached_90_percent"] else "本次测试 precision 未达到90%"
    page = f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>DBTT YOLO断丝定位演示</title>
<style>body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #bbb;padding:8px;vertical-align:top}}td:first-child{{width:20%;overflow-wrap:anywhere}}td:not(:first-child){{width:40%}}img{{max-width:100%;height:auto}}</style>
<h1>DBTT公开集 YOLO 断丝定位</h1>
<p>模型：dbtt_broken_wire_pretrain2/best.pt；只评估break类。绿色为数据集断丝标注，红色为模型断丝候选框。每个预测框与一个标注框IoU≥0.5才计为TP；其余预测框计FP，未匹配标注计FN。</p>
<p>验证集筛选{len(calibration_rows)}张调阈值：confidence={selected['threshold']:.2f}，precision={format_percent(selected['precision'])}，recall={format_percent(selected['recall'])}。固定阈值后测试集筛选{len(test_rows)}张：precision=<strong>{format_percent(test['precision'])}</strong>，recall={format_percent(test['recall'])}，TP={test['tp']}，FP={test['fp']}，FN={test['fn']}。<strong>{status}</strong>。</p>
<p><strong>漏检代价：</strong>本次test断丝标注共{test['tp'] + test['fn']}个，其中{test['fn']}个未被检出。precision的95% Wilson下界约为{format_percent(summary['test']['precision_wilson_95_lower'])}，样本量不足以稳健保证总体precision≥90%。</p>
<p><strong>探索性结果，非独立验收：</strong>当前阈值策略是在看过同一test集结果后比较的；模型权重也曾依据原valid集选取。筛选只依赖文件名与视频帧前缀，尚不能保证视觉完全独立。公开集结果不代表港口现场准确度。完整阈值扫描、逐图统计和来源名单见同目录CSV/TXT。</p>
<table><tr><th>图片与计数</th><th>原标注</th><th>模型预测</th></tr>{''.join(gallery)}</table></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(f"Demo: {output / 'index.html'}")
    print(f"Selected conf={selected['threshold']:.2f}; calibration P={format_percent(selected['precision'])}, R={format_percent(selected['recall'])}")
    print(f"Test P={format_percent(test['precision'])}, R={format_percent(test['recall'])}; TP={test['tp']} FP={test['fp']} FN={test['fn']}")


if __name__ == "__main__":
    main()
