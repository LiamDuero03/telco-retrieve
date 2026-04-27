"""GSMA specification source."""

from __future__ import annotations

import re
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean
from open_telco_rag.core.marker_loader import MarkerMixin


class GSMASource(MarkerMixin):
    """Loads, cleans and extracts metadata from GSMA documents."""

    SUPPORTED_EXTENSIONS = [".docx", ".pdf"]

    def __init__(self, use_marker: bool = False):
        super().__init__(use_marker=use_marker)

    def supported_extensions(self) -> list[str]:
        return self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> str:
        path = Path(path)
        if path.suffix.lower() == ".docx":
            return self._load_docx(path)
        elif path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def clean(self, text: str) -> str:
        """Apply GSMA specific cleaning on top of base cleaning."""
        text = base_clean(text)
        text = re.sub(r'GSM Association.*?\n', '', text)
        text = re.sub(r'GSMA.*?Confidential.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Official Document.*?\n', '', text)
        text = re.sub(r'V\d+\.\d+.*?Page \d+.*?\n', '', text)
        text = re.sub(r'^(Confidential|Non-Confidential|Public|Restricted).*?\n',
                      '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'Revision History.*?(?=\n[A-Z])', '', text, flags=re.DOTALL)
        text = re.sub(r'^Figure \d+.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Table \d+.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        """Extract GSMA document group, number and version from filename.

        Examples:
          IG.01 v1.0.docx          → group: IG, doc_num: 01, version: v1.0
          OPG.10 v2.0 Open Gateway → group: OPG, doc_num: 10, version: v2.0
          SGP.21 v2.5.docx         → group: SGP, doc_num: 21, version: v2.5
        """
        path = Path(path)
        meta = {
            "source":      path.name,
            "filename":    path.name,
            "source_type": "gsma",
        }

        meta["folder"] = path.parent.name

        name = path.stem

        doc_match = re.match(r'([A-Z]+)[\.\-](\d+)', name, re.IGNORECASE)
        if doc_match:
            meta["group"]   = doc_match.group(1).upper()
            meta["doc_num"] = doc_match.group(2)

        ver_match = re.search(r'v(\d+\.\d+(?:\.\d+)?)', name, re.IGNORECASE)
        if ver_match:
            meta["version"] = f"v{ver_match.group(1)}"

        title_match = re.search(r'v\d+\.\d+\s+(.+)$', name, re.IGNORECASE)
        if title_match:
            meta["title"] = title_match.group(1).strip()

        topic_map = {
            "IG":  "Intelligence Group",
            "OPG": "Operator Platform Group",
            "SGP": "eSIM / SIM specifications",
            "AID": "Artificial Intelligence and Data",
            "TSG": "Terminal Steering Group",
            "RCS": "Rich Communication Services",
            "SAS": "Security Accreditation Scheme",
            "WAS": "Wholesale Analytics and Settlement",
        }
        if "group" in meta:
            meta["topic"] = topic_map.get(meta["group"], "GSMA Working Group")

        return meta

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_docx(self, path: Path) -> str:
        # DOCX uses python-docx only — Marker requires WeasyPrint/GTK (not available on Windows)
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required. "
                "Install with: pip install open-telco-rag[docx]"
            )
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _load_pdf(self, path: Path) -> str:
        if self._use_marker:
            return self._marker_load(path)
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required. "
                "Install with: pip install open-telco-rag[pdf]"
            )
        reader = PdfReader(str(path))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)