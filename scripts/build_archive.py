"""Create a shareable ZIP, excluding environments and temporary state."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.venv', '.index-venv', '__pycache__', '.ultralytics', '.git', 'cache', 'logs'}
SECRET_PATTERNS = [
    rb'\bsk-[A-Za-z0-9]{20,}\b',
    rb'\bgh[pousr]_[A-Za-z0-9]{20,}\b',
    rb'\bgithub_pat_[A-Za-z0-9_]{20,}\b',
    rb'\bAKIA[A-Z0-9]{16}\b',
    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
]


def release_files():
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or EXCLUDE.intersection(relative.parts):
            continue
        if (path.name == 'manifest.json' or path.suffix in {'.pyc', '.pyo', '.tmp'}
                or path.name.startswith(('.env', '~$'))):
            continue
        if relative.parts[0] == 'runs' and relative.parts[1] != 'online_demo':
            continue
        yield path


def main():
    files = list(release_files())
    records = []
    for path in files:
        data = path.read_bytes()
        if path.suffix.lower() in {'.py', '.ps1', '.json', '.jsonl', '.yaml', '.yml', '.md', '.txt', '.js', '.html', '.csv', '.toml', '.xml'}:
            if any(re.search(pattern, data) for pattern in SECRET_PATTERNS):
                raise RuntimeError(f'Possible credential in {path.relative_to(ROOT)}')
        records.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
                        'sha256': hashlib.sha256(data).hexdigest()})
    manifest = ROOT / 'manifest.json'
    manifest.write_text(json.dumps({'files': records, 'file_count': len(records),
                                   'total_bytes': sum(r['bytes'] for r in records)}, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = ROOT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
        for path in [*files, manifest]:
            bundle.write(path, ROOT.name + '/' + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        bad_file = bundle.testzip()
        if bad_file:
            raise RuntimeError(f'ZIP validation failed: {bad_file}')
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size, 'files': len(files) + 1}))


if __name__ == '__main__':
    main()
