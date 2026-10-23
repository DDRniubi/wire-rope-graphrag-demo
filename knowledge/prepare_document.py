"""Extract the USACE manual into page-addressable GraphRAG inputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
PDF = ROOT / "source" / "EM_1110-2-3200.pdf"
SOURCE_URL = "https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-3200.pdf"
TITLE = "Wire Rope for Civil Works Structures (EM 1110-2-3200, 2016)"


def main() -> None:
    pdf_bytes = PDF.read_bytes()
    reader = PdfReader(PDF)
    if len(reader.pages) < 100:
        raise ValueError(f"Unexpectedly short manual: {len(reader.pages)} pages")
    input_dir = ROOT / "input"
    input_dir.mkdir(exist_ok=True)
    unexpected = [path.name for path in input_dir.glob("*.txt")
                  if not re.fullmatch(r"usace_wire_rope_page_\d{3}\.txt", path.name)]
    if unexpected:
        raise ValueError(f"Unexpected GraphRAG input files: {unexpected}")
    extraction = []
    indexed = 0
    for number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        clean = re.sub(r"[ \t]+", " ", raw)
        clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
        # Covers and intentionally blank pages have no useful retrieval content.
        include = len(clean) >= 200
        entry = {"pdf_page": number, "characters": len(clean), "indexed": include}
        if include:
            name = f"usace_wire_rope_page_{number:03d}.txt"
            header = f"Document: {TITLE}\nPDF page: {number} of {len(reader.pages)}\n\n"
            (input_dir / name).write_text(header + clean + "\n", encoding="utf-8")
            entry["file"] = name
            indexed += 1
        extraction.append(entry)
    manifest = {
        "document_title": TITLE,
        "publisher": "U.S. Army Corps of Engineers",
        "source_url": SOURCE_URL,
        "pdf_file": PDF.name,
        "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "pages": len(reader.pages),
        "indexed_pages": indexed,
        "extraction": extraction,
    }
    (ROOT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"pages": len(reader.pages), "indexed_pages": indexed,
                      "characters": sum(row["characters"] for row in extraction)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
