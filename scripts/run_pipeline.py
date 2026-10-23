"""Portable image -> YOLO -> optional Qwen -> manual graph -> report runner."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/sample_images')
    parser.add_argument('--output', default='')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--model-set', choices=['site', 'dbtt'], default='site')
    parser.add_argument('--confidence', type=float, default=0.25)
    parser.add_argument('--image-size', type=int, default=1280)
    parser.add_argument('--use-qwen', action='store_true')
    args = parser.parse_args()
    missing = [name for name in ('ultralytics', 'cv2', 'numpy', 'PIL', 'docx', 'pandas', 'pyarrow', 'dashscope')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Missing dependencies: {missing}. Run setup_env.ps1 first.')
    if args.use_qwen and not os.environ.get('DASHSCOPE_API_KEY', '').strip():
        raise RuntimeError('Set DASHSCOPE_API_KEY before using -UseQwen.')
    if not 0 <= args.confidence <= 1 or args.image_size < 32:
        raise ValueError('Invalid confidence or image size')
    source = resolve_path(args.input)
    images = [source] if source.is_file() else sorted(source.rglob('*'))
    images = [p for p in images if p.is_file() and p.suffix.lower() in EXTENSIONS]
    if not images:
        raise ValueError(f'No supported images in {source}')
    model = ROOT / 'models' / f'{args.model_set}_yolo' / 'best.pt'
    if not model.is_file():
        raise FileNotFoundError(model)
    output = resolve_path(args.output) if args.output else ROOT / 'runs' / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    staged = output / 'source_images'
    for image in images:
        relative = Path(image.name) if source.is_file() else image.relative_to(source)
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, destination)
    detection = output / 'detection'
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['YOLO_CONFIG_DIR'] = str(output / '.ultralytics')
    env['YOLO_AUTOINSTALL'] = 'false'
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    (output / '.ultralytics').mkdir()

    def run(script: Path, *arguments: object) -> None:
        subprocess.run([sys.executable, str(script), *map(str, arguments)], cwd=ROOT, env=env, check=True)

    print('[1/5] YOLO detection', flush=True)
    run(ROOT / 'detection/detect_broken_wire_whole_image.py', '--source', staged, '--model', model,
        '--out', detection, '--conf', args.confidence, '--imgsz', args.image_size, '--device', args.device, '--save-txt')
    if args.use_qwen:
        print('[2/5] Qwen visual description (API)', flush=True)
        run(ROOT / 'detection/generate_qwen_descriptions_for_new_images.py', '--source', staged,
            '--detection-dir', detection, '--model', 'qwen3.5-omni-flash')
    else:
        print('[2/5] Offline mode: no Qwen calls', flush=True)
    print('[3/5] Detection report and image events', flush=True)
    run(ROOT / 'detection/build_new_image_detection_report.py', '--detection-dir', detection,
        '--model', model, '--conf', args.confidence, '--imgsz', args.image_size)
    records = [json.loads(line) for line in (detection / 'new_image_results.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(records) != len(images) or any(row['status'] != 'ok' for row in records):
        raise RuntimeError('Some input images were not successfully detected')
    if args.use_qwen and not all(row['qwen_used'] for row in records):
        raise RuntimeError('Some input images have no Qwen description')
    events = []
    for row in records:
        boxes = row['yolo_boxes']
        if len(boxes) != row['yolo_detection_count']:
            raise ValueError('Detection count and boxes disagree')
        events.append({'event_id': 'image-' + hashlib.sha256(row['image'].encode()).hexdigest()[:12],
                       'image': row['image'], 'source_image': 'source_images/' + row['image'],
                       'candidate_count': len(boxes), 'candidate_boxes': boxes})
    (output / 'image_events.json').write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[4/5] Manual GraphRAG retrieval and report', flush=True)
    options = ['--use-qwen'] if args.use_qwen else []
    run(ROOT / 'knowledge/integrate_detection_run.py', '--run', output, *options)
    print('[5/5] Verify outputs', flush=True)
    run(ROOT / 'knowledge/verify_complete_demo.py', '--run', output)
    summary = {'status': 'completed', 'images': len(records), 'model_set': args.model_set,
               'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(),
               'confidence': args.confidence, 'image_size': args.image_size,
               'qwen_enabled': args.use_qwen, 'detection_reused': False,
               'entry': 'complete_demo.html'}
    (output / 'pipeline_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Completed: {output / "complete_demo.html"}', flush=True)


if __name__ == '__main__':
    main()
