"""Export a source-only repository without changing the runnable bundle."""

import argparse
import shutil
from pathlib import Path

from build_archive import release_files


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOTS = {"models", "data", "runs"}
KNOWLEDGE_ASSETS = {"source", "input", "output", "demo_artifacts"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new export directory: {output}")
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Export must be outside the runnable bundle")

    selected = []
    for source in release_files():
        relative = source.relative_to(ROOT)
        if relative.parts[0] in ASSET_ROOTS:
            continue
        if (relative.parts[0] == "knowledge" and len(relative.parts) > 1
                and relative.parts[1] in KNOWLEDGE_ASSETS):
            continue
        selected.append((source, relative))

    output.mkdir(parents=True)
    for source, relative in selected:
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Exported {len(selected)} source/documentation files to {output}")


if __name__ == "__main__":
    main()
