"""ETSI specification source — full telecom corpus."""

from __future__ import annotations

import re
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean
from open_telco_rag.core.marker_loader import MarkerMixin

# Map folder name → (doc_type, group)
_FOLDER_MAP: dict[str, tuple[str, str]] = {
    "gs_ENI":      ("GS", "ENI"),
    "gr_ENI":      ("GR", "ENI"),
    "gs_ZSM":      ("GS", "ZSM"),
    "gr_ZSM":      ("GR", "ZSM"),
    "gs_MEC":      ("GS", "MEC"),
    "gr_MEC":      ("GR", "MEC"),
    "gs_MEC_DEC":  ("GS", "MEC-DEC"),
    "gr_MEC_DEC":  ("GR", "MEC-DEC"),
    "gs_NFV":      ("GS", "NFV"),
    "gr_NFV":      ("GR", "NFV"),
    "gs_NFV_EVE":  ("GS", "NFV-EVE"),
    "gr_NFV_EVE":  ("GR", "NFV-EVE"),
    "gs_NFV_IFA":  ("GS", "NFV-IFA"),
    "gr_NFV_IFA":  ("GR", "NFV-IFA"),
    "gs_NFV_INF":  ("GS", "NFV-INF"),
    "gs_NFV_MAN":  ("GS", "NFV-MAN"),
    "gr_NFV_MAN":  ("GR", "NFV-MAN"),
    "gs_NFV_REL":  ("GS", "NFV-REL"),
    "gr_NFV_REL":  ("GR", "NFV-REL"),
    "gs_NFV_SEC":  ("GS", "NFV-SEC"),
    "gr_NFV_SEC":  ("GR", "NFV-SEC"),
    "gs_NFV_SOL":  ("GS", "NFV-SOL"),
    "gr_NFV_SOL":  ("GR", "NFV-SOL"),
    "gs_NFV_TST":  ("GS", "NFV-TST"),
    "gr_NFV_TST":  ("GR", "NFV-TST"),
    "gs_F5G":      ("GS", "F5G"),
    "gr_F5G":      ("GR", "F5G"),
    "gs_RIS":      ("GS", "RIS"),
    "gr_SAI":      ("GR", "SAI"),
    "ENI_gs":      ("GS", "ENI"),
    "ENI_gr":      ("GR", "ENI"),
    "ZSM_gs":      ("GS", "ZSM"),
    "ZSM_gr":      ("GR", "ZSM"),
    "MEC_gs":      ("GS", "MEC"),
    "MEC_gr":      ("GR", "MEC"),
    "NFV_gs":      ("GS", "NFV"),
    "NFV_gr":      ("GR", "NFV"),
    "NFV_ts":      ("TS", "NFV"),
    "NFV_tr":      ("TR", "NFV"),
    "MEC-GS011-AppEnabler":        ("GS", "MEC"),
    "mec.gs010-2-app-pkg-lcm-api": ("GS", "MEC"),
    "mec.gs011-app-enablement-api":("GS", "MEC"),
    "mec.gs012-rnis-api":          ("GS", "MEC"),
    "mec.gs013-location-api":      ("GS", "MEC"),
    "mec.gs014-ue-identity-api":   ("GS", "MEC"),
    "mec.gs015-bandwith-mgmt-api": ("GS", "MEC"),
    "mec.gs016-ue-app-api":        ("GS", "MEC"),
    "mec.gs021-amsi-api":          ("GS", "MEC"),
    "mec.gs028-wai-api":           ("GS", "MEC"),
    "mec.gs030-vis-api":           ("GS", "MEC"),
    "NFV-SOL006":                  ("GS", "NFV-SOL"),
    "NFV-SOL009":                  ("GS", "NFV-SOL"),
    "NFV-SOL012":                  ("GS", "NFV-SOL"),
    "ts":          ("TS", "TC"),
    "tr":          ("TR", "TC"),
    "en":          ("EN", "TC"),
    "es":          ("ES", "TC"),
    "sr":          ("SR", "TC"),
    "etsi":        ("", ""),
}

_GROUP_TITLE = {
    "ENI":     "Experiential Networked Intelligence",
    "ZSM":     "Zero-touch Service Management",
    "MEC":     "Multi-access Edge Computing",
    "MEC-DEC": "Multi-access Edge Computing — Deployment",
    "NFV":     "Network Functions Virtualisation",
    "NFV-EVE": "NFV Evolution",
    "NFV-IFA": "NFV Interfaces and Architecture",
    "NFV-INF": "NFV Infrastructure",
    "NFV-MAN": "NFV Management",
    "NFV-REL": "NFV Reliability",
    "NFV-SEC": "NFV Security",
    "NFV-SOL": "NFV Solutions",
    "NFV-TST": "NFV Testing",
    "F5G":     "5th Generation Fixed Networks",
    "RIS":     "Reconfigurable Intelligent Surfaces",
    "SAI":     "Securing Artificial Intelligence",
    "TC":      "ETSI Technical Committee",
}


class ETSISource(MarkerMixin):
    """Loads, cleans and extracts metadata from ETSI specifications."""

    def __init__(self, use_marker: bool = False):
        super().__init__(use_marker=use_marker)

    def supported_extensions(self) -> list[str]:
        return [".pdf", ".yaml", ".yml"]

    def load(self, path: Path) -> str:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            return self._load_yaml(path)
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def clean(self, text: str) -> str:
        """Apply ETSI-specific cleaning on top of base cleaning."""
        text = base_clean(text)
        text = re.sub(r'ETSI\s+(?:GS|GR|TS|TR|EN|ES|SR)\s+[\w\-]+\s+\d+.*?\n', '', text)
        text = re.sub(r'ETSI\s+G[SR]\s+\S+\s+\d+.*?\n', '', text)
        text = re.sub(r'©\s*ETSI\s+\d{4}.*?\n', '', text)
        text = re.sub(r'650 Route des Lucioles.*?\n', '', text)
        text = re.sub(r'F-06921 Sophia Antipolis.*?\n', '', text)
        text = re.sub(r'Important notice.*?(?=\n#{1,3}\s)', '', text, flags=re.DOTALL)
        text = re.sub(r'^Figure\s+[\d\-–\s]+.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Table\s+[\d\-–\s]+.*?$',  '', text, flags=re.MULTILINE)
        text = re.sub(r'^\[[\w\.]+\].*?$',          '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        path   = Path(path)
        folder = path.parent.name
        meta   = {
            "source":      path.name,
            "filename":    path.name,
            "source_type": "etsi",
            "folder":      folder,
        }

        doc_type, group = _FOLDER_MAP.get(folder, ("", ""))

        if not doc_type:
            fl = folder.lower()
            if fl.startswith("gs_"):
                doc_type = "GS"; group = folder[3:].replace("_", "-")
            elif fl.startswith("gr_"):
                doc_type = "GR"; group = folder[3:].replace("_", "-")
            elif fl.endswith("_gs"):
                doc_type = "GS"; group = folder[:-3].replace("_", "-")
            elif fl.endswith("_gr"):
                doc_type = "GR"; group = folder[:-3].replace("_", "-")
            elif "mec" in fl:
                doc_type = "GS"; group = "MEC"
            elif "nfv" in fl or "sol" in fl:
                doc_type = "GS"; group = "NFV"

        if not doc_type and folder == "etsi":
            stem_low = path.stem.lower()
            if stem_low.startswith("gs_"):
                doc_type = "GS"
                m = re.match(r'gs_([a-z]+)', stem_low)
                if m: group = m.group(1).upper()
            elif stem_low.startswith("gr_"):
                doc_type = "GR"
                m = re.match(r'gr_([a-z]+)', stem_low)
                if m: group = m.group(1).upper()

        if doc_type: meta["doc_type"]    = doc_type
        if group:    meta["group"]       = group
        if group:    meta["group_title"] = _GROUP_TITLE.get(group, group)

        if path.suffix.lower() in (".yaml", ".yml"):
            meta["doc_type"] = "OpenAPI Specification"
            try:
                import yaml
                with open(path, encoding="utf-8") as f:
                    spec = yaml.safe_load(f)
                if isinstance(spec, dict) and "info" in spec:
                    info = spec["info"]
                    meta["api_title"]   = info.get("title", "")
                    meta["api_version"] = info.get("version", "")
            except Exception:
                pass
            return meta

        stem  = path.stem
        num_m = re.search(r'[a-z](\d{3,6})v', stem, re.IGNORECASE)
        if num_m:
            meta["doc_num"] = num_m.group(1)

        ver_m = re.search(r'v(\d{2})(\d{2})(\d{2})', stem, re.IGNORECASE)
        if ver_m:
            meta["version"] = (
                f"{int(ver_m.group(1))}."
                f"{int(ver_m.group(2))}."
                f"{int(ver_m.group(3))}"
            )

        return meta

    # ── Loaders ───────────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> str:
        if self._use_marker:
            return self._marker_load(path)
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required. Install with: pip install open-telco-rag[pdf]"
            )
        reader = PdfReader(str(path))
        pages  = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)

    def _load_yaml(self, path: Path) -> str:
        """Convert ETSI MEC/NFV YAML spec to readable prose."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required. Install with: pip install pyyaml")
        with open(path, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        return self._spec_to_prose(spec, path)

    def _spec_to_prose(self, spec, path: Path) -> str:
        """Convert an OpenAPI spec dict to readable prose."""
        if not isinstance(spec, dict):
            return str(spec) if spec else ""

        if "info" not in spec and "paths" not in spec and "definitions" not in spec:
            lines = [f"# {path.stem}\n"]
            for key, value in spec.items():
                lines.append(f"## {key}")
                lines.append(str(value))
                lines.append("")
            return "\n".join(lines)

        lines       = []
        info        = spec.get("info", {})
        title       = info.get("title", path.stem)
        version     = info.get("version", "")
        description = info.get("description", "")

        lines.append(f"# {title} (v{version})")
        if description:
            lines.append(f"\n{description}\n")

        host = spec.get("host", "")
        base = spec.get("basePath", "")
        if host or base:
            lines.append(f"Base URL: {host}{base}\n")

        paths = spec.get("paths", {})
        if paths:
            lines.append("## Endpoints\n")
            for path_str, path_item in paths.items():
                if not isinstance(path_item, dict):
                    continue
                for method, operation in path_item.items():
                    if method.startswith("x-") or method == "parameters":
                        continue
                    if not isinstance(operation, dict):
                        continue
                    summary = operation.get("summary", "")
                    desc    = operation.get("description", "")
                    lines.append(f"### {method.upper()} {path_str}")
                    if summary: lines.append(summary)
                    if desc:    lines.append(desc)
                    params = operation.get("parameters", [])
                    if params:
                        lines.append("Parameters:")
                        for p in params:
                            if not isinstance(p, dict): continue
                            lines.append(
                                f"  - {p.get('name','')} ({p.get('in','')}, "
                                f"{'required' if p.get('required') else 'optional'}): "
                                f"{p.get('description','')}"
                            )
                    responses = operation.get("responses", {})
                    if responses:
                        lines.append("Responses:")
                        for code, resp in responses.items():
                            if isinstance(resp, dict):
                                lines.append(f"  - {code}: {resp.get('description','')}")
                    lines.append("")

        definitions = spec.get(
            "definitions",
            spec.get("components", {}).get("schemas", {})
        )
        if definitions and isinstance(definitions, dict):
            lines.append("## Data Models\n")
            for def_name, definition in definitions.items():
                if not isinstance(definition, dict): continue
                def_desc  = definition.get("description", "")
                def_props = definition.get("properties", {})
                lines.append(f"### {def_name}")
                if def_desc: lines.append(def_desc)
                if def_props and isinstance(def_props, dict):
                    lines.append("Properties:")
                    for prop_name, prop in def_props.items():
                        if not isinstance(prop, dict): continue
                        lines.append(
                            f"  - {prop_name} ({prop.get('type', prop.get('$ref',''))}): "
                            f"{prop.get('description','')}"
                        )
                lines.append("")

        return "\n".join(lines)