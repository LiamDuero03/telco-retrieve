"""ITU-T specification source — all series."""

from __future__ import annotations

import re
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean
from open_telco_rag.core.marker_loader import MarkerMixin

# All active telecom series downloaded
TELECOM_SERIES = {
    "G": "Transmission systems and media, digital systems and networks",
    "E": "Overall network operation, telephone service, service operation and human factors",
    "Q": "Switching and signalling, and associated measurements and tests",
    "H": "Audiovisual and multimedia systems",
    "X": "Data networks, open system communications and security",
    "Y": "Global information infrastructure, Internet protocol aspects, next-generation networks",
    "M": "Telecommunication management, including TMN and network maintenance",
    "L": "Environment and ICTs, climate change, e-waste, energy efficiency",
    "P": "Telephone transmission quality, telephone installations, local line networks",
    "K": "Protection against interference",
    "J": "Cable networks and transmission of television, sound programme and multimedia signals",
    "F": "Non-telephone telecommunication services",
    "T": "Terminals for telematic services",
    "V": "Data communication over the telephone network",
}

# Status codes in ITU-T filenames
STATUS_MAP = {
    "I": "In force",
    "S": "Superseded",
    "P": "Provisional",
    "W": "Withdrawn",
}


class ITUSource(MarkerMixin):
    """Loads, cleans and extracts metadata from ITU-T recommendations.

    Handles all 14 active telecom series (G, E, Q, H, X, Y, M, L, P, K, J, F, T, V).

    Expected filename pattern:
        T-REC-Y.3172-201906-I!!PDF-E.pdf       (recommendation)
        T-REC-Y.Sup96-202511-I!!PDF-E.pdf      (supplement)
        T-REC-G.709-202006-I!!PDF-E.pdf        (G-series)

    Files are organised into series subfolders:
        data/itu/Y/T-REC-Y.*.pdf
        data/itu/G/T-REC-G.*.pdf
        ...
    """

    SUPPORTED_EXTENSIONS = [".pdf"]

    def __init__(self, use_marker: bool = False):
        super().__init__(use_marker=use_marker)

    def supported_extensions(self) -> list[str]:
        return self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> str:
        """Load PDF as text."""
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def clean(self, text: str) -> str:
        """Apply ITU-T specific cleaning on top of base cleaning."""
        text = base_clean(text)

        # Remove series-agnostic ITU header/footer boilerplate
        text = re.sub(r'ITU-T\s+Rec\.\s+[A-Z]\.\S+.*?\n', '', text)
        text = re.sub(
            r'Rec(?:ommendation)?\s+ITU-T\s+[A-Z]\.\S+.*?\n', '', text
        )
        text = re.sub(r'International Telecommunication Union.*?\n', '', text)
        text = re.sub(r'Telecommunication Standardization Sector.*?\n', '', text)

        # Remove edition/publication/IPR notices
        text = re.sub(r'Edition\s+\d+.*?\n', '', text)
        text = re.sub(r'Published.*?ITU.*?\n', '', text)
        text = re.sub(r'ITU[\s\u00a0]+\d{4}\s+All rights reserved.*?\n', '', text)
        text = re.sub(r'All rights reserved.*?ITU.*?\n', '', text)

        # Remove figure/table captions
        text = re.sub(r'^Figure\s+[\d\-–]+[^\n]*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Table\s+[\d\-–]+[^\n]*$', '', text, flags=re.MULTILINE)

        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        # Collapse whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        """Extract metadata from ITU-T filename.

        Handles all series and document types:
          T-REC-Y.3172-201906-I!!PDF-E.pdf  → series=Y, rec_number=Y.3172
          T-REC-Y.Sup96-202511-I!!PDF-E.pdf → series=Y, rec_number=Y.Sup96, doc_type=Supplement
          T-REC-G.709-202006-I!!PDF-E.pdf   → series=G, rec_number=G.709
          T-REC-Q.931-199805-I!!PDF-E.pdf   → series=Q, rec_number=Q.931
        """
        path = Path(path)
        meta = {
            "source":      str(path.name),
            "filename":    path.name,
            "source_type": "itu",
        }

        # Series subfolder (parent directory name e.g. "Y", "G")
        parent = path.parent.name
        if parent in TELECOM_SERIES:
            meta["subfolder"] = parent

        stem = path.stem  # e.g. T-REC-Y.3172-201906-I!!PDF-E

        # ── Series letter and recommendation number ───────────────────────────
        m = re.search(
            r'T-REC-([A-Z])\.(\S+?)(?=-\d{6})',
            stem,
            re.IGNORECASE,
        )
        if m:
            series      = m.group(1).upper()
            number_part = m.group(2)

            meta["series"]       = series
            meta["series_title"] = TELECOM_SERIES.get(series, "ITU-T Recommendation")

            if re.match(r'Sup\d+', number_part, re.IGNORECASE):
                meta["rec_number"] = f"{series}.{number_part}"
                meta["doc_type"]   = "Supplement"
            else:
                meta["rec_number"] = f"{series}.{number_part}"
                meta["doc_type"]   = "Recommendation"

        # ── Approval date ─────────────────────────────────────────────────────
        date_m = re.search(r'-(\d{4})(\d{2})-', stem)
        if date_m:
            meta["year"]  = date_m.group(1)
            meta["month"] = date_m.group(2)

        # ── Status ────────────────────────────────────────────────────────────
        status_m = re.search(r'-(\d{6})-([ISPW])!!', stem)
        if status_m:
            status_code    = status_m.group(2)
            meta["status"] = STATUS_MAP.get(status_code, status_code)

        return meta

    # ── Private loaders ───────────────────────────────────────────────────────

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