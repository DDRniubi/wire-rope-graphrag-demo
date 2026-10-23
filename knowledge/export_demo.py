"""Export the actual Microsoft GraphRAG index into a browsable evidence demo."""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
ARTIFACTS = ROOT / "demo_artifacts"


def clean(value: object) -> str:
    return str(value).strip() if value is not None else ""


def ids(value: object) -> list[str]:
    if value is None or isinstance(value, float):
        return []
    return [str(item) for item in value]


def main() -> None:
    names = ("documents", "text_units", "entities", "relationships", "communities", "community_reports")
    tables = {name: pd.read_parquet(OUTPUT / f"{name}.parquet") for name in names}
    documents = tables["documents"]
    text_units = tables["text_units"]
    page_by_document = {str(row.id): int(re.search(r"page_(\d+)", row.title).group(1))
                        for row in documents.itertuples() if re.search(r"page_(\d+)", row.title)}
    page_by_unit = {str(row.id): page_by_document.get(str(row.document_id))
                    for row in text_units.itertuples()}

    def pages(unit_ids: object) -> list[int]:
        return sorted({page_by_unit[unit_id] for unit_id in ids(unit_ids)
                       if unit_id in page_by_unit and page_by_unit[unit_id] is not None})

    entities = tables["entities"].sort_values(["degree", "frequency"], ascending=False)
    selected = entities.head(55)
    node_names = {clean(row.title) for row in selected.itertuples()}
    nodes = [{"id": clean(row.title), "type": clean(row.type), "description": clean(row.description),
              "degree": int(row.degree), "pages": pages(row.text_unit_ids)} for row in selected.itertuples()]
    relationships = tables["relationships"]
    edges = [{"source": clean(row.source), "target": clean(row.target),
              "description": clean(row.description), "weight": float(row.weight),
              "pages": pages(row.text_unit_ids)} for row in relationships.itertuples()
             if clean(row.source) in node_names and clean(row.target) in node_names]
    edges.sort(key=lambda row: (-row["weight"], row["source"], row["target"]))
    edges = edges[:100]

    graph = nx.Graph()
    graph.add_nodes_from(node["id"] for node in nodes)
    graph.add_weighted_edges_from((edge["source"], edge["target"], max(1, edge["weight"])) for edge in edges)
    positions = nx.spring_layout(graph, seed=14, k=1.25 / max(len(nodes), 1) ** 0.5, iterations=180)
    for node in nodes:
        x, y = positions[node["id"]]
        node["x"] = round(float(x), 5)
        node["y"] = round(float(y), 5)

    reports = tables["community_reports"].sort_values("rank", ascending=False)
    communities = [{"id": int(row.community), "title": clean(row.title),
                    "summary": clean(row.summary), "rank": float(row.rank)}
                   for row in reports.head(8).itertuples()]
    answers_path = ARTIFACTS / "answers.json"
    answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else []
    data = {"source": json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8")),
            "stats": {"documents": len(documents), "text_units": len(text_units),
                      "entities": len(entities), "relationships": len(relationships),
                      "communities": len(tables["communities"])},
            "nodes": nodes, "edges": edges, "communities": communities, "answers": answers}
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "graph_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "demo_data.js").write_text(
        "window.DEMO_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(json.dumps(data["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
