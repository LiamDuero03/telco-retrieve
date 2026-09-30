"""3GPP specification source - TSpec-LLM dataset."""

from __future__ import annotations

import re
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean

# Converter output filename reused across every spec's folder (e.g. 28312-i11/raw.md).
_GENERIC_FILENAME_STEMS = {"raw", "output", "converted"}


class TPPSource:
    """Loads, cleans and extracts metadata from 3GPP TSpec-LLM Markdown files."""

    # .md = TSpec-LLM marked/converted files (structured, has heading markup)
    # .docx = original 3GPP Word docs (unstructured baseline, flat text only)
    SUPPORTED_EXTENSIONS = [".md", ".docx"]

    def supported_extensions(self) -> list[str]:
        return self.SUPPORTED_EXTENSIONS

    def load(self, path: Path) -> str:
        """Load a source file as plain text.

        .md files keep markdown heading structure as-is. .docx loads flat/
        unstructured via `_load_docx` (no headings) - intentional "no
        markdown structure" comparison arm.
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
        # Remove pandoc superscripts (^x^): one line, no spaces, no $, so LaTeX exponents ($10^{-5}$) survive
        text = re.sub(r"\^[^^\s$]+\^", "", text)
        # Remove width/height attributes left over from image conversion
        text = re.sub(r'\{[^}\n]*width=[^}\n]*\}', '', text)
        # Remove standalone media references
        text = re.sub(r"media/image\d+\.\w+", "", text)
        # Remove 3GPP document cover table boilerplate, never past the Scope heading
        text = re.sub(
            r"3rd Generation Partnership Project(?:(?!^(?:#{1,6}\s*)?1\s+Scope\b).)*?The present document",
            "The present document",
            text, flags=re.DOTALL | re.MULTILINE
        )
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        """Extract metadata from a TSpec-LLM file.

        Handles both layouts: a directly-named spec (28312-i11.md) or a
        generic file in a spec-named folder (28312-i11/raw.md) - the latter
        reads the identifier from the parent folder. release comes from a
        "Rel-NN" ancestor directory (checked against the full path).
        canonical_id combines spec_number (+ doc_part) with release, e.g.
        "28.312 (Rel-18)".
        """
        path = Path(path)

        generic_filename = path.stem.lower() in _GENERIC_FILENAME_STEMS
        id_stem = path.parent.name if generic_filename else path.stem
        display_name = f"{id_stem}{path.suffix}" if generic_filename else path.name

        meta = {
            "source":      display_name,
            "filename":    display_name,
            "source_type": "3gpp",
        }

        parts = path.parts

        # Extract release from directory - e.g. Rel-18
        for part in parts:
            if re.match(r"Rel-\d+", part, re.IGNORECASE):
                meta["release"] = part
                break

        # Extract series from directory - e.g. 28_series -> 28
        for part in parts:
            series_match = re.match(r"(\d+)_series", part, re.IGNORECASE)
            if series_match:
                meta["series"] = series_match.group(1)
                break

        # e.g. 28312-i11 -> spec_number: 28.312, version: i11
        #      38508-2   -> spec_number: 38.508, doc_part: 2 (multi-part spec)
        #      25401     -> spec_number: 25.401 (no version/part in the name)
        spec_match = re.match(r"(\d{5})(?:-(\w+))?$", id_stem)
        if spec_match:
            raw_num = spec_match.group(1)   # e.g. 28312
            suffix  = spec_match.group(2)   # e.g. i11, 2, or None
            # Convert 28312 -> 28.312
            meta["spec_number"] = f"{raw_num[:2]}.{raw_num[2:]}"
            meta["series"]      = raw_num[:2]
            if suffix:
                if re.match(r"[a-j]\d{2}$", suffix, re.IGNORECASE):
                    meta["version"] = suffix
                else:
                    meta["doc_part"] = suffix   # e.g. "2" for TS 38.508-2

        # Version letter -> release (fallback; the directory scan above wins).
        version_to_release = {
            "a": "Rel-10", "b": "Rel-11", "c": "Rel-12",
            "d": "Rel-13", "e": "Rel-14", "f": "Rel-15",
            "g": "Rel-16", "h": "Rel-17", "i": "Rel-18", "j": "Rel-19"
        }
        if "version" in meta:
            version_letter = meta["version"][0].lower()
            if version_letter in version_to_release:
                meta.setdefault("release", version_to_release[version_letter])

        spec_id = meta.get("spec_number") or id_stem
        if "doc_part" in meta:
            spec_id = f"{spec_id}-{meta['doc_part']}"
        meta["canonical_id"] = f"{spec_id} ({meta['release']})" if "release" in meta else spec_id

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