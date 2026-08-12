"""3GPP specification source — TSpec-LLM dataset."""

from __future__ import annotations

import re
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean


class TPPSource:
    """Loads, cleans and extracts metadata from 3GPP TSpec-LLM Markdown files."""

    # .md = TSpec-LLM marked/converted files (structured, has heading markup)
    # .docx = original 3GPP Word docs (unstructured baseline, flat text only)
    SUPPORTED_EXTENSIONS = [".md", ".docx"]

    def supported_extensions(self) -> list[str]:
        return self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> str:
        """Load a source file as plain text.

        .md files are read as-is (they carry markdown heading structure).
        .docx files are loaded as a flat, unstructured baseline via
        `_load_docx` — paragraph text only, no heading markup preserved —
        intentionally, to serve as the "no markdown structure" comparison arm.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".md":
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".docx":
            return self._load_docx(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def clean(self, text: str) -> str:
        """Apply 3GPP specific cleaning on top of base cleaning."""
        text = base_clean(text)
        # Remove markdown table header blocks (pipe table format)
        text = re.sub(r"(\+[-=+]+\+\n)(\|.*?\n)+(\+[-=+]+\+\n?)+", "", text)
        # Remove image references
        text = re.sub(r"!\[.*?\]\(.*?\)(\{.*?\})?", "", text)
        # Remove superscript notation
        text = re.sub(r"\^[^^]+\^", "", text)
        # Remove width/height attributes left over from image conversion
        text = re.sub(r'\{[^}]*width=[^}]*\}', '', text)
        # Remove standalone media references
        text = re.sub(r"media/image\d+\.\w+", "", text)
        # Remove 3GPP document cover table boilerplate
        text = re.sub(
            r"3rd Generation Partnership Project.*?The present document",
            "The present document",
            text, flags=re.DOTALL
        )
        # Collapse whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        """Extract metadata from TSpec-LLM filename.

        Example: 28312-i11.md
        → series: 28, spec_number: 28.312, version: i11
        """
        path = Path(path)
        meta = {
            "source":      path.name,
            "filename":    path.name,
            "source_type": "3gpp",
        }

        parts = path.parts

        # Extract release from directory — e.g. Rel-18
        for part in parts:
            if re.match(r"Rel-\d+", part, re.IGNORECASE):
                meta["release"] = part
                break

        # Extract series from directory — e.g. 28_series → 28
        for part in parts:
            series_match = re.match(r"(\d+)_series", part, re.IGNORECASE)
            if series_match:
                meta["series"] = series_match.group(1)
                break

        # Extract spec number and version from filename
        # e.g. 28312-i11.md → spec_number: 28.312, version: i11
        name = path.stem  # e.g. 28312-i11
        file_match = re.match(r"(\d{5})-(\w+)", name)
        if file_match:
            raw_num          = file_match.group(1)   # e.g. 28312
            meta["version"]  = file_match.group(2)   # e.g. i11
            # Convert 28312 → 28.312
            meta["spec_number"] = f"{raw_num[:2]}.{raw_num[2:]}"
            meta["series"]      = raw_num[:2]

        # 3GPP version letter maps to release
        # a=Rel-10, b=Rel-11, c=Rel-12 ... i=Rel-18, j=Rel-19
        version_to_release = {
            "a": "Rel-10", "b": "Rel-11", "c": "Rel-12",
            "d": "Rel-13", "e": "Rel-14", "f": "Rel-15",
            "g": "Rel-16", "h": "Rel-17", "i": "Rel-18", "j": "Rel-19"
        }
        if "version" in meta:
            version_letter = meta["version"][0].lower()
            if version_letter in version_to_release:
                meta["release"] = version_to_release[version_letter]

        return meta
    
    def _load_docx(self, path: Path) -> str:
        """Fallback DOCX loader if needed."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX support. "
                "Install with: pip install open-telco-rag[docx]"
            )
        doc = Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())