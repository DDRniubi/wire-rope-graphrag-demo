"""Select source-backed relationships from a GraphRAG index."""

from __future__ import annotations

import re


TOPIC_TERMS = {
    "visible_break": ("broken wire", "wire break", "inspection", "visual", "strand"),
    "fatigue": ("fatigue", "bending", "cyclic", "sheave", "failure"),
    "lubrication": ("lubrication", "lubricant", "corrosion", "maintenance"),
    "internal_limit": ("internal", "inspection", "nondestructive", "magnetic"),
}
FIRST_CONTENT_PAGE = 13


def select_relations(relations, page_by_unit: dict[str, int | None],
                     terms: tuple[str, ...], limit: int = 4,
                     allowed_pages: set[int] | None = None) -> list[dict]:
    candidates = []
    for row in relations.itertuples():
        source = str(row.source)
        target = str(row.target)
        endpoints = f"{source} {target}".lower()
        description = str(row.description).lower()
        value = row.text_unit_ids
        unit_ids = [] if value is None or isinstance(value, float) else [str(item) for item in value]
        pages = sorted({page_by_unit[item] for item in unit_ids
                        if item in page_by_unit and page_by_unit[item] is not None
                        and page_by_unit[item] >= FIRST_CONTENT_PAGE})
        if allowed_pages is not None:
            pages = [page for page in pages if page in allowed_pages]
        if not pages:
            continue
        score = sum(5 if term in endpoints else 1 if term in description else 0 for term in terms)
        if score == 0:
            continue
        score += min(float(row.weight), 5) / 10
        candidates.append((score, {
            "id": str(row.id), "source": source, "target": target,
            "description": str(row.description), "pages": pages,
            "text_unit_ids": unit_ids,
        }))
    candidates.sort(key=lambda item: (-item[0], item[1]["source"], item[1]["target"]))
    chosen = []
    seen_pairs = set()
    for _, relation in candidates:
        pair = tuple(sorted((relation["source"], relation["target"])))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        chosen.append(relation)
        if len(chosen) >= limit:
            break
    return chosen


def indexed_pages(documents, text_units) -> tuple[dict[str, int], dict[str, int | None], dict[int, str]]:
    page_by_document = {}
    source_pages = {}
    for row in documents.itertuples():
        match = re.search(r"page_(\d+)", str(row.title))
        if match:
            page = int(match.group(1))
            page_by_document[str(row.id)] = page
            source_pages[page] = str(row.text)
    page_by_unit = {str(row.id): page_by_document.get(str(row.document_id))
                    for row in text_units.itertuples()}
    return page_by_document, page_by_unit, source_pages


def select_pages(evidence: list[dict], source_pages: dict[int, str],
                 terms: tuple[str, ...], limit: int = 4) -> list[int]:
    def score(page: int) -> tuple[int, int]:
        text = source_pages.get(page, "").lower()
        return (sum(text.count(term) for term in terms), -page)

    chosen = []
    for relation in evidence:
        candidates = [page for page in relation["pages"] if page in source_pages and page not in chosen]
        if candidates:
            chosen.append(max(candidates, key=score))
        if len(chosen) >= limit:
            return chosen
    remaining = {page for relation in evidence for page in relation["pages"]} - set(chosen)
    chosen.extend(sorted(remaining, key=score, reverse=True)[:limit - len(chosen)])
    return chosen
