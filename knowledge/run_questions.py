"""Ask two page-grounded questions using the indexed manual's graph."""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd

from graph_retrieval import indexed_pages, select_pages, select_relations


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "demo_artifacts"
QUESTIONS = [
    {"question": "钢丝绳为什么会发生断丝和疲劳？检查时应关注什么？",
     "terms": ("broken wire", "fatigue", "inspection", "bending"),
     "allowed_pages": {78, 130, 131}},
    {"question": "表面检查与内部检测分别能发现什么？润滑维护有什么作用？",
     "terms": ("visual", "internal", "inspection", "lubrication"),
     "paths": [
         (("visual", "inspection"), {121}),
         (("non-destructive", "magnetic", "internal"), {131}),
         (("lubrication", "wear"), {136, 137}),
     ]},
]


def main() -> None:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY is required for paid Qwen questions")
    relations = pd.read_parquet(ROOT / "output" / "relationships.parquet")
    units = pd.read_parquet(ROOT / "output" / "text_units.parquet")
    documents = pd.read_parquet(ROOT / "output" / "documents.parquet")
    _, page_by_unit, source_pages = indexed_pages(documents, units)
    ARTIFACTS.mkdir(exist_ok=True)
    answers = []
    for spec in QUESTIONS:
        if "paths" in spec:
            selected = [relation for terms, allowed in spec["paths"]
                        for relation in select_relations(relations, page_by_unit, terms,
                                                         limit=2, allowed_pages=allowed)]
        else:
            selected = select_relations(relations, page_by_unit, spec["terms"], limit=4,
                                        allowed_pages=spec["allowed_pages"])
        pages = select_pages(selected, source_pages, spec["terms"], limit=4)
        if not selected or not pages:
            raise RuntimeError(f"No source-backed graph path for: {spec['question']}")
        source_text = "\n\n".join(f"[P{page}] {source_pages[page]}" for page in pages)
        paths = "\n".join(f"{row['source']} -> {row['target']}" for row in selected)
        payload = {
            "model": "qwen-plus", "temperature": 0, "max_tokens": 350,
            "messages": [
                {"role": "system", "content": "请用中文写一段不超过180字的回答，不要标题、列表、表格或扩展总结。只依据提供的手册原文；图谱路径仅用于检索，不能充当事实依据。每个事实句末标[P页码]，页码必须来自提供的原文。不要添加原文未说的数值、检测能力或设备适用结论。术语 equalizing sheave 译为均衡滑轮，不能译为等径滑轮。"},
                {"role": "user", "content": f"问题：{spec['question']}\n\nGraphRAG图谱路径：\n{paths}\n\n手册原文：\n{source_text}"},
            ],
        }
        request = urllib.request.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        answer = result["choices"][0]["message"]["content"].strip()
        cited = sorted({int(page) for page in re.findall(r"\[P(\d+)\]", answer)})
        if not cited or not set(cited).issubset(pages):
            raise RuntimeError(f"Invalid source page citations: {answer}")
        answers.append({"method": "graph-guided", "question": spec["question"],
                        "answer": answer, "pages": cited, "retrieved_pages": pages,
                        "relations": selected, "model": "qwen-plus",
                        "review_status": "requires_source_check"})
        (ARTIFACTS / "answers.json").write_text(
            json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved answer {len(answers)}: {len(selected)} relations, pages {cited}", flush=True)


if __name__ == "__main__":
    main()
