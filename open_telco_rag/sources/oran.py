"""O-RAN Alliance specification source."""

from __future__ import annotations

import re
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean
from open_telco_rag.core.marker_loader import MarkerMixin


class ORANSource(MarkerMixin):
    """Loads, cleans and extracts metadata from O-RAN Alliance specifications."""

    SUPPORTED_EXTENSIONS = [".pdf", ".docx"]

    def __init__(self, use_marker: bool = False):
        super().__init__(use_marker=use_marker)

    def supported_extensions(self) -> list[str]:
        return self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> str:
        """Load text from PDF or DOCX."""
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        elif path.suffix.lower() in [".docx", ".doc"]:
            return self._load_docx(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def clean(self, text: str) -> str:
        """Apply O-RAN specific cleaning on top of base cleaning."""
        text = base_clean(text)
        text = re.sub(r"(O-RAN ALLIANCE[^\n]*\n){2,}", "O-RAN ALLIANCE\n", text)
        text = re.sub(r"^\s*v\d+\.\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*O-RAN\.[A-Z0-9.+-]+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        """Extract work group, release, version from O-RAN filename.

        Example: O-RAN.WG1.TR.Use-Cases-Analysis-Report-R005-v19.00.pdf
        """
        path = Path(path)
        name = path.stem
        meta = {
            "source":      path.name,
            "filename":    path.name,
            "source_type": "oran",
        }

        wg_match = re.search(r"\.(WG\d+|TIFG|TS|TR)\.", name, re.IGNORECASE)
        if wg_match:
            meta["work_group"] = wg_match.group(1).upper()

        type_match = re.search(r"\.(TS|TR)\.", name, re.IGNORECASE)
        if type_match:
            meta["doc_type"] = type_match.group(1).upper()

        release_match = re.search(r"-(R\d+)-", name, re.IGNORECASE)
        if release_match:
            meta["release"] = release_match.group(1).upper()

        version_match = re.search(r"-(v[\d.]+)$", name, re.IGNORECASE)
        if version_match:
            meta["version"] = version_match.group(1).lower()

        return meta

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> str:
        if self._use_marker:
            return self._marker_load(path)
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF support. "
                "Install with: pip install open-telco-rag[pdf]"
            )
        reader = PdfReader(str(path))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    def _load_docx(self, path: Path) -> str:
        # DOCX uses python-docx only — Marker requires WeasyPrint/GTK (not available on Windows)
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX support. "
                "Install with: pip install open-telco-rag[docx]"
            )
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())