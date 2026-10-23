"""Check a consolidated image-to-document-GraphRAG demo without API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


ROOT = Path(__file__).resolve().parent


class Links(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for name in ("href", "src"):
            if values.get(name):
                self.links.append(values[name])


def verify_html(path: Path) -> int:
    parser = Links()
    parser.feed(path.read_text(encoding="utf-8"))
    checked = 0
    for link in parser.links:
        if link.startswith(("http:", "https:", "data:", "mailto:")):
            continue
        target = (path.parent / unquote(link.split("#", 1)[0].split("?", 1)[0])).resolve()
        if not target.is_file():
            raise AssertionError(f"Broken HTML link: {path}: {link}")
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run / "document_graphrag" / "summary.json").read_text(encoding="utf-8"))
    events = json.loads((run / "image_events.json").read_text(encoding="utf-8"))
    graph_index = run / "document_graphrag" / "graph_index"
    valid_relations = set(pd.read_parquet(graph_index / "output" / "relationships.parquet")["id"].astype(str))
    manifest = json.loads((graph_index / "source_manifest.json").read_text(encoding="utf-8"))
    pdf = run / "document_graphrag" / "graph_demo" / "source" / manifest["pdf_file"]
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != manifest["pdf_sha256"]:
        raise AssertionError("Copied source PDF differs from GraphRAG source manifest")
    for name in ("documents", "text_units", "entities", "communities", "community_reports"):
        if not (graph_index / "output" / f"{name}.parquet").is_file():
            raise AssertionError(f"Missing GraphRAG index table: {name}")
    if len(summary["rows"]) != len(events) or not summary["rows"]:
        raise AssertionError("Missing per-image rows")
    for index, row in enumerate(summary["rows"], 1):
        if not (run / row["marked"]).is_file():
            raise AssertionError(f"Missing YOLO image: {row['marked']}")
        report = run / "document_graphrag" / "reports" / row["report"]
        trace_path = run / "document_graphrag" / "traces" / row["trace"]
        if not report.is_file() or not trace_path.is_file():
            raise AssertionError(f"Missing report or trace for image {index}")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if (run / trace["graph_index"]).resolve() != graph_index.resolve():
            raise AssertionError(f"Trace does not point to the copied graph index at row {index}")
        if trace["image"] != row["image"] or trace["event_id"] != events[index - 1]["event_id"]:
            raise AssertionError(f"Image/trace mismatch at row {index}")
        if not all(relation["id"] in valid_relations for relation in trace["relations"]):
            raise AssertionError(f"Graph relation missing from actual index at row {index}")
        pages = {page["page"] for page in trace["source_pages"]}
        answer = trace["generation"]["answer"] if summary["qwen_document_generation"] else report.read_text(encoding="utf-8")
        cited = {int(page) for page in re.findall(r"\[P(\d+)\]", answer)}
        if not cited or not cited.issubset(pages):
            raise AssertionError(f"Invalid source page citation at row {index}")
        if summary["qwen_document_generation"] and trace["generation"]["model"] != "qwen-plus":
            raise AssertionError(f"Qwen generation not recorded at row {index}")
    links = verify_html(run / "complete_demo.html")
    links += verify_html(run / "document_graphrag" / "graph_demo" / "demo_artifacts" / "index.html")
    print(f"Verified {len(events)} images, graph relations, citations, and {links} local HTML links")


if __name__ == "__main__":
    main()
