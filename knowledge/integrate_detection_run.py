"""Connect an existing YOLO/Qwen run to the indexed wire-rope document."""

from __future__ import annotations

import argparse
from html import escape
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from graph_retrieval import TOPIC_TERMS, indexed_pages, select_pages, select_relations


ROOT = Path(__file__).resolve().parent
PDF_NAME = json.loads((ROOT / "source_manifest.json").read_text(encoding="utf-8"))["pdf_file"]
REVIEWED_REPORT_PAGES = {78, 80, 121, 130, 131, 136}
REPORT_FACTS = {
    "fatigue_bending": (78, "钢丝绳经过滑轮、卷筒或滚轮时的反复弯曲与疲劳有关。",
                        "Fatigue usually results from moving over sheaves, drums, or rollers. A rope moving over drums, sheaves, or rollers is subjected to cyclic bending stresses."),
    "visual_outside": (121, "手册将外层钢丝的断丝、腐蚀和磨蚀列为目视检查内容。",
                       "visual inspection of the outside of the wire rope that would cover condition of the outside wires such as wire breaks, corrosion, abrasive wear"),
    "ndt_limit": (131, "手册指出，无损检测可评估钢丝绳内部状态，但对细小或密集断丝及断丝与点蚀的区分存在局限。",
                  "NDT can be used to determine the internal condition of wire rope. However, there are conditions that cannot or may not be detected, such as breaks in small wires, closely-spaced broken wires, broken wire versus pit corrosion"),
    "lubrication": (136, "手册指出，润滑有助于减少钢丝绳经过滑轮或卷筒时的磨蚀，并提供防腐保护。",
                    "Reduce abrasion as wire rope goes over sheaves or drums. (3) To provide corrosion protection"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_ids(value: object) -> list[str]:
    return [] if value is None or isinstance(value, float) else [str(item) for item in value]


def topics(result: dict) -> list[str]:
    found = ["visible_break", "fatigue"]
    visual = " ".join(str(result.get(key, "")) for key in
                      ("surface_condition", "contamination", "short_caption", "detailed_caption")).lower()
    if any(word in visual for word in ("油", "污", "润滑", "oil", "grease", "污染")):
        found.append("lubrication")
    if result.get("detection_difficulty") in ("高", "中") or result.get("yolo_detection_count", 0) == 0:
        found.append("internal_limit")
    return found


def format_answer(image_result: str, fact_ids: list[str]) -> str:
    knowledge = "；".join(f"{REPORT_FACTS[name][1].rstrip('。')}[P{REPORT_FACTS[name][0]}]"
                         for name in dict.fromkeys(fact_ids)) + "。"
    return f"图像结果：{image_result}\n\n文档依据：{knowledge}"


def qwen_answer(event: dict, result: dict, evidence: list[dict], pages: dict[int, str],
                page_numbers: list[int]) -> tuple[str, dict]:
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for --use-qwen")
    available = {name: (page, chinese, excerpt)
                 for name, (page, chinese, excerpt) in REPORT_FACTS.items() if page in page_numbers}
    if not available:
        raise ValueError(f"No reviewed fact on selected pages: {page_numbers}")
    for page, _, excerpt in available.values():
        normalized_page = " ".join(pages[page].split()).lower()
        if " ".join(excerpt.split()).lower() not in normalized_page:
            raise ValueError(f"Reviewed fact excerpt not found on PDF page {page}")
    facts = [{"id": name, "pdf_page": page, "verified_fact": chinese, "source_excerpt": excerpt}
             for name, (page, chinese, excerpt) in available.items()]
    prompt = ("根据图像检测结果和图谱检索到的手册原文依据，只返回JSON对象，"
              "字段为image_result和fact_ids。image_result用一句中文陈述YOLO检出的断丝候选框数量"
              "及已有千问视觉描述，不要添加没有给出的图像细节，不要把候选框说成人工确认断丝。"
              "fact_ids从给出的id中选1到3个与图像最相关的，不能新增id；无须自行改写手册事实。"
              "不要输出停用、报废或继续使用决定。\n\n"
              f"YOLO候选框数：{event['candidate_count']}\n"
              f"千问视觉描述：{result.get('short_caption', '')}；{result.get('surface_condition', '')}\n"
              f"图谱路径：{'; '.join(item['source'] + ' -> ' + item['target'] for item in evidence)}\n"
              f"可选的核对依据：{json.dumps(facts, ensure_ascii=False)}")
    attempts = []
    for attempt in range(2):
        instruction = prompt if attempt == 0 else prompt + "\n请严格返回所要求的JSON对象和已有事实id。"
        payload = {"model": "qwen-plus", "temperature": 0,
                   "response_format": {"type": "json_object"},
                   "messages": [{"role": "user", "content": instruction}]}
        request = urllib.request.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.load(response)
        raw_answer = raw["choices"][0]["message"]["content"].strip()
        findings = []
        try:
            chosen = json.loads(raw_answer)
            image_result = chosen["image_result"].strip()
            fact_ids = chosen["fact_ids"]
            if not isinstance(fact_ids, list) or not 1 <= len(fact_ids) <= 3 or any(
                    fact_id not in available for fact_id in fact_ids):
                findings.append("invalid_fact_ids")
            if not image_result or re.search(r"报废|停用|继续使用|判废", image_result):
                findings.append("invalid_image_result")
            if str(event["candidate_count"]) not in image_result:
                findings.append("missing_candidate_count")
        except (ValueError, KeyError, TypeError, AttributeError):
            findings.append("invalid_json")
        attempts.append({"attempt": attempt + 1, "raw_answer": raw_answer, "request_id": raw.get("id"),
                         "findings": findings})
        if not findings:
            answer = format_answer(image_result, fact_ids)
            cited = sorted({available[name][0] for name in fact_ids})
            return answer, {"model": "qwen-plus", "prompt": instruction, "answer": answer,
                            "request_id": raw.get("id"), "cited_pages": cited,
                            "selected_fact_ids": fact_ids, "attempts": attempts}
    raise RuntimeError(f"Qwen answer failed validation after retry: {attempts[-1]['findings']}")


def offline_answer(event: dict, result: dict, pages: list[int]) -> str:
    page_label = f"[P{pages[0]}]" if pages else ""
    return (f"图像结果：YOLO检出{event['candidate_count']}个断丝目标框；本次未调用千问视觉模型。\n\n"
            f"文档依据：已定位至手册原文{page_label}；本次未调用千问生成文档分析。")


def write_dashboard(run: Path, rows: list[dict], cards: list[dict], qwen_used: bool) -> None:
    style = """
    :root{font-family:'Microsoft YaHei',Arial,sans-serif;color:#20292b;background:#f4f6f4}
    *{box-sizing:border-box}body{margin:0}header{background:#1e3032;color:#fff;padding:18px 28px;display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}
    h1{font-size:20px;margin:0}h2{font-size:16px;margin:0 0 12px}h3{font-size:13px;margin:0 0 7px}p{line-height:1.65}a{color:#116d71;text-decoration:none}a:hover{text-decoration:underline}
    .counts{display:flex;gap:24px;color:#d6e0dc;font-size:12px}.counts strong{font-size:20px;color:#fff;margin-right:5px}
    main{display:grid;grid-template-columns:250px minmax(0,1fr);min-height:calc(100vh - 76px)}nav{background:#fff;border-right:1px solid #d9e2df;padding:20px 14px}nav h2{padding:0 8px}
    nav button{font:inherit;display:block;text-align:left;width:100%;border:1px solid transparent;background:transparent;padding:11px 9px;cursor:pointer;color:#273537;border-radius:4px;line-height:1.45;overflow-wrap:anywhere}
    nav button.active,nav button:hover{background:#eaf4ef;border-color:#bad5c9}.small{font-size:12px;color:#667578}.workspace{padding:24px 30px;min-width:0}
    .toplinks{display:flex;gap:16px;flex-wrap:wrap;border-bottom:1px solid #d9e2df;padding-bottom:15px;margin-bottom:20px}
    .panel{display:none}.panel.active{display:block}.grid{display:grid;grid-template-columns:minmax(300px,1.3fr) minmax(300px,1fr);gap:26px}
    figure{margin:0}figure img{display:block;width:100%;max-height:570px;object-fit:contain;background:#e9edeb;border:1px solid #d9e2df}figcaption{font-size:12px;color:#667578;margin-top:8px}
    .analysis{white-space:pre-wrap;line-height:1.8;background:#fff;border-left:3px solid #23776d;padding:16px;margin:14px 0;overflow-wrap:anywhere}
    .relation{border-top:1px solid #d9e2df;padding:10px 0;font-size:13px}.relation b{color:#245e57}.relation a{margin-left:8px;font-size:12px}
    .meta{color:#5c6a6d;font-size:12px;margin:6px 0 18px}.section{margin-top:25px;border-top:1px solid #d9e2df;padding-top:18px}
    @media(max-width:1050px){main{grid-template-columns:1fr}nav{border-right:0;border-bottom:1px solid #d9e2df;display:flex;flex-wrap:wrap;gap:6px}nav h2{width:100%}nav button{width:auto;max-width:260px}.grid{grid-template-columns:1fr}.workspace{padding:18px}}
    """
    parts = ["<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>钢丝绳检测与GraphRAG综合演示</title><style>", style, "</style></head><body>",
             "<header><h1>钢丝绳检测与 GraphRAG 综合演示</h1><div class='counts'>",
             f"<span><strong>{len(rows)}</strong>图像</span><span><strong>{sum(row['boxes'] for row in rows)}</strong>目标框</span>",
             f"<span><strong>{sum(row['relations'] for row in rows)}</strong>图谱关系引用</span></div></header>",
             "<main><nav><h2>检测图像</h2>"]
    for index, row in enumerate(rows):
        parts.append(f"<button class='pick{' active' if index == 0 else ''}' data-index='{index}'>{escape(row['image'])}<br><span class='small'>{row['boxes']} 个框 · {row['relations']} 条关系</span></button>")
    parts.append("</nav><div class='workspace'><div class='toplinks'>"
                 "<a href='document_graphrag/graph_demo/demo_artifacts/index.html'>打开 GraphRAG 图谱</a>"
                 f"<a href='document_graphrag/graph_demo/source/{PDF_NAME}'>打开来源 PDF</a>"
                 "<a href='complete_demo.md'>文件索引</a></div>")
    for index, (row, card) in enumerate(zip(rows, cards)):
        focus = quote(json.dumps([{"source": relation["source"], "target": relation["target"]}
                                  for relation in card["relations"]], ensure_ascii=False), safe="")
        parts.append(f"<article class='panel{' active' if index == 0 else ''}' id='panel-{index}'>"
                     f"<h2>{escape(row['image'])}</h2><p class='meta'>YOLO 目标框 {row['boxes']} 个 · 文档分析："
                     f"{'千问 qwen-plus' if qwen_used else '离线模板'}</p><div class='grid'><div>"
                     f"<figure><img src='{escape(row['marked'], quote=True)}' alt='YOLO 标框图'>"
                     "<figcaption>YOLO 标框结果</figcaption></figure>"
                     f"<p><b>千问视觉描述：</b>{escape(card['caption'])}</p></div><div>"
                     "<h2>检索增强分析</h2>"
                     f"<div class='analysis'>{escape(card['answer'])}</div>"
                     "<div class='section'><h2>本图使用的图谱关系</h2>")
        for relation in card["relations"]:
            page_links = " ".join(
                f"<a href='document_graphrag/graph_demo/source/{PDF_NAME}#page={page}'>原文 P{page}</a>"
                for page in relation["pages"])
            parts.append(f"<div class='relation'><b>{escape(relation['source'])} → {escape(relation['target'])}</b>{page_links}</div>")
        parts.append("</div><p class='meta'>图谱关系用于定位原文；回答中的事实应以所链接的 PDF 页为准。</p>"
                     f"<p><a href='document_graphrag/graph_demo/demo_artifacts/index.html?focus={focus}'>在图谱中高亮本图关系</a></p>"
                     f"<p><a href='document_graphrag/reports/{row['report']}'>逐图报告</a> · "
                     f"<a href='document_graphrag/traces/{row['trace']}'>完整检索与千问提示词</a></p></div></div></article>")
    parts.append("</div></main><script>document.querySelectorAll('.pick').forEach(button=>button.addEventListener('click',()=>{"
                 "document.querySelectorAll('.pick').forEach(x=>x.classList.remove('active'));"
                 "document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));"
                 "button.classList.add('active');document.getElementById('panel-'+button.dataset.index).classList.add('active');"
                 "}));</script></body></html>")
    (run / "complete_demo.html").write_text("".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Existing delivery run with detection and image_events.json")
    parser.add_argument("--use-qwen", action="store_true", help="Send detection metadata and public text to DashScope")
    parser.add_argument("--reuse-qwen", action="store_true", help="Reuse prior Qwen traces without network calls")
    args = parser.parse_args()
    if args.reuse_qwen:
        args.use_qwen = True
    run = args.run.resolve()
    events = json.loads((run / "image_events.json").read_text(encoding="utf-8"))
    results = read_jsonl(run / "detection" / "new_image_results.jsonl")
    by_name = {row["image"]: row for row in results}
    if len(events) != len(results) or len(by_name) != len(results):
        raise ValueError("Detection events and result rows do not match")
    output = ROOT / "output"
    relations = pd.read_parquet(output / "relationships.parquet")
    units = pd.read_parquet(output / "text_units.parquet")
    documents = pd.read_parquet(output / "documents.parquet")
    _, page_by_unit, source_pages = indexed_pages(documents, units)
    result_dir = run / "document_graphrag"
    traces_dir = result_dir / "traces"
    reports_dir = result_dir / "reports"
    traces_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    index_copy = result_dir / "graph_index"
    index_copy.mkdir(parents=True, exist_ok=True)
    for name in ("input", "output", "prompts"):
        shutil.copytree(ROOT / name, index_copy / name, dirs_exist_ok=True)
    for name in ("settings.yaml", "source_manifest.json"):
        shutil.copy2(ROOT / name, index_copy / name)
    graph_demo = result_dir / "graph_demo"
    (graph_demo / "source").mkdir(parents=True, exist_ok=True)
    (graph_demo / "demo_artifacts").mkdir(parents=True, exist_ok=True)
    for source, destination in [
        (ROOT / "source" / PDF_NAME, graph_demo / "source" / PDF_NAME),
        (ROOT / "demo_artifacts" / "index.html", graph_demo / "demo_artifacts" / "index.html"),
        (ROOT / "demo_artifacts" / "demo_data.js", graph_demo / "demo_artifacts" / "demo_data.js"),
    ]:
        shutil.copy2(source, destination)
    rows = []
    cards = []
    for index, event in enumerate(events, 1):
        image = event["image"].replace("\\", "/")
        result = by_name.get(image)
        if result is None:
            raise ValueError(f"Missing detection row: {image}")
        terms = tuple(dict.fromkeys(term for topic in topics(result) for term in TOPIC_TERMS[topic]))
        selected = select_relations(relations, page_by_unit, terms, limit=4,
                                    allowed_pages=REVIEWED_REPORT_PAGES)
        page_numbers = select_pages(selected, source_pages, terms, limit=4)
        if not selected or not page_numbers:
            raise ValueError(f"No source-backed GraphRAG relation for {image}")
        trace_name = f"image_{index:03d}.json"
        report_name = f"image_{index:03d}.md"
        if args.reuse_qwen:
            previous = json.loads((traces_dir / trace_name).read_text(encoding="utf-8"))
            if previous["image"] != image or previous["event_id"] != event["event_id"]:
                raise ValueError(f"Saved Qwen trace does not match image event: {trace_name}")
            answer = previous["generation"]["answer"]
            generation = previous["generation"]
            if not answer or generation.get("model") != "qwen-plus":
                raise ValueError(f"Saved trace has no Qwen answer: {trace_name}")
            saved_choice = json.loads(generation["attempts"][-1]["raw_answer"])
            fact_ids = saved_choice["fact_ids"]
            if any(name not in REPORT_FACTS or REPORT_FACTS[name][0] not in page_numbers for name in fact_ids):
                raise ValueError(f"Saved facts do not match reviewed source pages: {trace_name}")
            answer = format_answer(saved_choice["image_result"].strip(), fact_ids)
            generation["answer"] = answer
        elif args.use_qwen:
            answer, generation = qwen_answer(event, result, selected, source_pages, page_numbers)
        else:
            answer, generation = offline_answer(event, result, page_numbers), {"model": None, "answer": None}
        trace = {"image": image, "event_id": event["event_id"], "yolo_boxes": event["candidate_boxes"],
                 "vision_description": {key: result.get(key) for key in
                                        ("short_caption", "surface_condition", "contamination", "detection_difficulty")},
                 "graph_index": "document_graphrag/graph_index",
                 "selection_rule": "rank actual indexed relations by image-condition terms within manually reviewed report pages, then rank their source pages",
                 "reviewed_report_pages": sorted(REVIEWED_REPORT_PAGES),
                 "topics": topics(result), "relations": selected,
                 "source_pages": [{"page": page, "pdf": PDF_NAME, "text": source_pages[page]} for page in page_numbers],
                 "generation": generation}
        (traces_dir / trace_name).write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        marked_path = Path(result["result_image"])
        if not marked_path.is_relative_to(run):
            marked_path = run / "detection" / "images" / marked_path.name
        if not marked_path.is_file():
            raise FileNotFoundError(f"YOLO marked image missing: {marked_path}")
        marked = marked_path.relative_to(run).as_posix()
        cited = sorted({int(number) for number in re.findall(r"\[P(\d+)\]", answer)})
        lines = [f"# 图像 {index:03d}：{image}", "", f"![YOLO标框图](<../../{marked}>)", "",
                 f"- YOLO目标框：{event['candidate_count']}个", f"- 千问视觉描述：{result.get('short_caption') or '未调用'}",
                 f"- 图谱关系：{len(selected)}条", "", "## 检索增强分析", "", answer, "", "## 原文依据", ""]
        for page in cited:
            lines.append(f"- [P{page}] [手册原文第{page}页](<../graph_demo/source/{PDF_NAME}#page={page}>)")
        lines += ["", f"[图谱展示](<../graph_demo/demo_artifacts/index.html>) · [检索记录](<../traces/{trace_name}>)", ""]
        (reports_dir / report_name).write_text("\n".join(lines), encoding="utf-8")
        rows.append({"image": image, "marked": marked, "boxes": event["candidate_count"],
                     "relations": len(selected), "report": report_name, "trace": trace_name,
                     "generation": "qwen-plus" if args.use_qwen else "offline_template"})
        cards.append({"caption": result.get("short_caption") if result.get("qwen_used") else "本次未调用千问视觉模型",
                      "answer": answer, "relations": selected})
        print(f"[{index}/{len(events)}] {image}: {len(selected)} relations; {rows[-1]['generation']}", flush=True)
    summary = {"images": len(rows), "yolo_boxes": sum(row["boxes"] for row in rows),
               "graph_index": "document_graphrag/graph_index", "document": PDF_NAME,
               "qwen_document_generation": args.use_qwen, "rows": rows}
    (result_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 钢丝绳断丝检测与文档 GraphRAG 综合演示", "",
             "流程：输入图像 → YOLO标框 → 千问视觉描述 → 图像事件 → Microsoft GraphRAG文档索引中的实体关系 → PDF原文页 → 逐图分析。",
             "", f"本次处理 {len(rows)} 张图像，YOLO共检出 {summary['yolo_boxes']} 个目标框。",
             "", "[打开单页展示](<complete_demo.html>)",
             "", "| 图像 | 框数 | 标框图 | 图谱关系 | 逐图报告 | 检索记录 |", "|---|---:|---|---:|---|---|"]
    for row in rows:
        lines.append(f"| {row['image']} | {row['boxes']} | [查看](<{row['marked']}>) | {row['relations']} | "
                     f"[查看](<document_graphrag/reports/{row['report']}>) | "
                     f"[查看](<document_graphrag/traces/{row['trace']}>) |")
    lines += ["", f"[查看 GraphRAG 知识图谱](<document_graphrag/graph_demo/demo_artifacts/index.html>) · "
              f"[查看来源 PDF](<document_graphrag/graph_demo/source/{PDF_NAME}>)", "",
              "图谱关系是模型抽取的检索线索，原文页码是分析依据。YOLO目标框不等于人工确认的断丝数量。", ""]
    (run / "complete_demo.md").write_text("\n".join(lines), encoding="utf-8")
    readme = ["# 文档 GraphRAG 运行文件", "",
              "本目录把原始文档、Microsoft GraphRAG 建图索引、逐图检索记录和生成报告放在同一运行结果下。",
              f"", f"- `graph_index/input/`：手册中 {len(source_pages)} 页有实质内容的原文文本。",
              "- `graph_index/output/`：实际生成的文档、文本块、实体、关系、社区及向量索引。",
              "- `graph_index/settings.yaml`、`graph_index/prompts/`：建索引配置与提示词，不含 API Key。",
              "- `graph_index/source_manifest.json`：来源网址、原 PDF 哈希与页数。",
              "- `graph_demo/`：可浏览图谱及来源 PDF。",
              "- `traces/image_*.json`：每张图的 YOLO 框、视觉描述、检索关系、原文页及千问提示词和回答。",
              "- `reports/image_*.md`：逐图分析报告。",
              "- `summary.json`：图像数、框数、生成方式和文件对应关系。", "",
              "本次文档问答不是 GraphRAG 内置查询接口的原样输出，而是使用 GraphRAG 实际索引中的关系，",
              "沿关系取回 PDF 原文页，再由千问基于原文生成分析。关系抽取结果是检索线索，原文页才是事实依据。", ""]
    (result_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    write_dashboard(run, rows, cards, args.use_qwen)
    print(f"Complete demo: {run / 'complete_demo.html'}")


if __name__ == "__main__":
    main()
