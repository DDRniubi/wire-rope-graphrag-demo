from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from http import HTTPStatus
from pathlib import Path
from typing import Any

from dashscope import MultiModalConversation


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FIELDS = [
    "scene",
    "rope_orientation",
    "surface_condition",
    "lighting_condition",
    "contamination",
    "candidate_visibility",
    "detection_difficulty",
    "short_caption",
    "detailed_caption",
]


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTS else []
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def load_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["image"].replace("\\", "/"): row for row in csv.DictReader(handle)}


def read_locations(path: Path) -> list[str]:
    locations = []
    if not path.exists():
        return locations
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy = float(parts[1]), float(parts[2])
        horizontal = "左侧" if cx < 1 / 3 else "右侧" if cx > 2 / 3 else "中部"
        vertical = "上部" if cy < 1 / 3 else "下部" if cy > 2 / 3 else "中部"
        locations.append("图像中部" if horizontal == vertical == "中部" else f"图像{vertical}{horizontal}")
    return locations


def make_prompt(image_name: str, count: int, locations: list[str]) -> str:
    location_text = "、".join(locations) if locations else "无"
    return f"""你正在为钢丝绳断丝检测系统生成报告辅助描述。

YOLO模型输出信息：
- 图片：{image_name}
- 断丝候选框数量：{count}
- 候选框大致位置：{location_text}

要求：
1. 红框是YOLO模型检出的断丝目标位置。不要自行增加或删除框，也不要写成人工现场确认。
2. 重点客观描述现场环境、钢丝绳方向、表面状态、光照、污染和候选区域可见性。
3. 不猜测污染物化学成分，不推断事故等级、安全风险或设备用途。
4. candidate_visibility和detection_difficulty只能填写“高”“中”“低”。
5. short_caption不超过50字，detailed_caption控制在80至150字。
6. 直接描述图像，不写“需复核”“无法判断”等固定套话。
7. 仅输出合法JSON对象，不要输出Markdown或解释。

{{
  "scene": "客观场景",
  "rope_orientation": "钢丝绳方向",
  "surface_condition": "表面状态",
  "lighting_condition": "光照条件",
  "contamination": "污染或附着物描述",
  "candidate_visibility": "高/中/低",
  "detection_difficulty": "高/中/低",
  "short_caption": "简短描述",
  "detailed_caption": "80至150字详细描述"
}}"""


def response_text(response: Any) -> str:
    content = response.output.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    texts = []
    for item in content if isinstance(content, list) else []:
        if isinstance(item, dict) and item.get("text"):
            texts.append(str(item["text"]))
        elif getattr(item, "text", None):
            texts.append(str(item.text))
    return "\n".join(texts).strip()


def parse_json(text: str) -> dict[str, str]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value = json.loads(cleaned[cleaned.find("{") : cleaned.rfind("}") + 1])
    missing = [field for field in FIELDS if field not in value]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")
    return {field: str(value[field]).strip() for field in FIELDS}


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["image"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate optional Qwen scene descriptions for new-image YOLO results.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--detection-dir", required=True)
    parser.add_argument("--model", default="qwen3.5-omni-flash")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing. Configure the caller's own Bailian API key first.")

    source = Path(args.source).resolve()
    detection_dir = Path(args.detection_dir).resolve()
    output = detection_dir / "qwen_descriptions.jsonl"
    raw_dir = detection_dir / "qwen_raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(detection_dir / "prediction_summary.csv")
    completed = load_completed(output)
    images = collect_images(source)

    for index, image_path in enumerate(images, start=1):
        rel = image_path.name if source.is_file() else image_path.relative_to(source).as_posix()
        if rel in completed:
            continue
        prediction = predictions[rel]
        label_path = detection_dir / "labels" / Path(rel).with_suffix(".txt")
        locations = read_locations(label_path)
        count = int(prediction["detections"])
        marked_image = Path(prediction["output_image"])
        if not marked_image.is_file():
            raise FileNotFoundError(f"YOLO-marked image is missing: {marked_image}")
        prompt = make_prompt(rel, count, locations)
        prompt += "\n输入图像中的红框由YOLO绘制。请特别描述红框周围可见纹理、遮挡和易混淆因素。"
        last_error = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                response = MultiModalConversation.call(
                    api_key=api_key,
                    model=args.model,
                    messages=[{"role": "user", "content": [{"image": marked_image.as_uri()}, {"text": prompt}]}],
                )
                if response.status_code != HTTPStatus.OK:
                    raise RuntimeError(f"HTTP {response.status_code}: {getattr(response, 'message', '')}")
                raw_text = response_text(response)
                parsed = parse_json(raw_text)
                record = {
                    "image": rel,
                    "model": args.model,
                    "yolo_candidate_count": count,
                    "yolo_candidate_locations": locations,
                    "vision_input": "yolo_marked_image",
                    "vision_input_path": str(marked_image),
                    **parsed,
                    "request_id": str(getattr(response, "request_id", "")),
                    "manual_review_required": True,
                }
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                (raw_dir / f"{index:04d}.json").write_text(
                    json.dumps({"input_image": str(marked_image), "prompt": prompt, "response": raw_text, "parsed": parsed}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[{index}/{len(images)}] success: {rel}", flush=True)
                time.sleep(0.8)
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                print(f"[{index}/{len(images)}] attempt {attempt} failed: {last_error}", flush=True)
                if attempt < args.max_retries:
                    time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Qwen description failed for {rel}: {last_error}")

    print(f"descriptions: {output}")


if __name__ == "__main__":
    main()
