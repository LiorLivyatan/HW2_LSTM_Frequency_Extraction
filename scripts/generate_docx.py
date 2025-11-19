#!/usr/bin/env python3
"""
Generate a minimal .docx file from a plain-text/markdown source.

This script creates a valid WordprocessingML document using only the standard
library (zipfile). It converts each input line into a separate paragraph.

Usage:
  python scripts/generate_docx.py evaluation/self_assessment.md evaluation/self_assessment.docx
"""

import sys
import zipfile
from datetime import datetime


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="/docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="/docProps/app.xml"/>
</Relationships>
"""

APP_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex CLI</Application>
</Properties>
"""

CORE_XML_TEMPLATE = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{title}</dc:title>
  <dc:creator>Codex Agent</dc:creator>
  <cp:lastModifiedBy>Codex Agent</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{ts}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{ts}</dcterms:modified>
  <dc:description>Auto-generated from Markdown</dc:description>
</cp:coreProperties>
"""


def xml_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def make_document_xml(lines):
    paragraphs = []
    for line in lines:
        # Preserve spaces and direction-agnostic text; Word handles RTL text.
        t = xml_escape(line.rstrip("\n"))
        paragraphs.append(
            f"<w:p><w:r><w:t xml:space=\"preserve\">{t}</w:t></w:r></w:p>"
        )

    body = "".join(paragraphs) + "<w:sectPr/>"
    return (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        f"<w:body>{body}</w:body></w:document>"
    )


def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/generate_docx.py <input_md> <output_docx>")
        sys.exit(1)

    in_md, out_docx = sys.argv[1], sys.argv[2]

    with open(in_md, "r", encoding="utf-8") as f:
        lines = f.readlines()

    doc_xml = make_document_xml(lines)
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    core_xml = CORE_XML_TEMPLATE.format(title="Self-Assessment", ts=now)

    with zipfile.ZipFile(out_docx, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("docProps/app.xml", APP_XML)
        z.writestr("docProps/core.xml", core_xml)
        z.writestr("word/document.xml", doc_xml)

    print(f"DOCX written to {out_docx}")


if __name__ == "__main__":
    main()

