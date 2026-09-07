"""TM Forum specification source - full corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean
from open_telco_rag.core.marker_loader import MarkerMixin


EXCLUDED_PATH_PARTS = {
    "alpaca.cpp", "text-generation-webui", "openapi-generator", "swagger2oas",
    "oas2puml", "OpenAPI-diagram-generator", "schemasVisualization",
    "nodejs-helm-chart-example", "oda-helm-charts", "tmf-rancher-catalog",
    "v5_java_apicore", "usekeeper", "yamldiff", "ipdr",
    "tmforum-rand.github.io", "MODA", "CanvasSpecjam2024", "RAND-LICENSE",
    "CSS-For-Catalyst-Submission", "TMF123_Template", "PM-PostmansR18-5",
    "Open_Api_And_Data_Model-latest", ".github", "charts",
    "ODA-Operator-Development",
}


def _classify_repo(path: Path) -> dict:
    result = {}
    for part in reversed(path.parts):
        m_ctk = re.match(r'CTK[_\-](TMF\d+)[_\-](.+)', part, re.IGNORECASE)
        if m_ctk:
            result["repo_type"]  = "ctk"
            result["tmf_number"] = m_ctk.group(1).upper()
            result["api_name"]   = m_ctk.group(2).replace("-", " ").replace("_", " ")
            return result
        m_ri = re.match(r'RI[_\-](TMF\d+)[_\-](.+)', part, re.IGNORECASE)
        if m_ri:
            result["repo_type"]  = "ri"
            result["tmf_number"] = m_ri.group(1).upper()
            result["api_name"]   = m_ri.group(2).replace("-", " ").replace("_", " ")
            return result
        m_pm = re.match(r'Postman[_\-](TMF\d+)[_\-](.+)', part, re.IGNORECASE)
        if m_pm:
            result["repo_type"]  = "postman"
            result["tmf_number"] = m_pm.group(1).upper()
            return result
        m = re.match(r'(TMF\d+)[_\-](.+)', part, re.IGNORECASE)
        if m:
            result["repo_type"]  = "api"
            result["tmf_number"] = m.group(1).upper()
            result["api_name"]   = m.group(2).replace("-", " ").replace("_", " ")
            return result
        m2 = re.match(r'(TMFC\d+)[_\-]?(.+)?', part, re.IGNORECASE)
        if m2 and part.upper().startswith("TMFC"):
            result["repo_type"]   = "oda_component"
            result["tmfc_number"] = m2.group(1).upper()
            result["component"]   = (m2.group(2) or "").replace("-", " ").replace("_", " ")
            return result
        m3 = re.match(r'(ig\d+)', part, re.IGNORECASE)
        if m3:
            result["repo_type"] = "ig"
            result["ig_number"] = m3.group(1).upper()
            return result
    for part in path.parts:
        if "ODA-Ready-for-publication" in part:
            result["repo_type"] = "oda"; return result
        if part == "schemas":
            result["repo_type"] = "schema"; return result
        if part == "Open_Api_And_Data_Model":
            result["repo_type"] = "monorepo"; return result
        if part == "gsma-gst":
            result["repo_type"] = "gsma_gst"; return result
        if part == "Intent_Repository":
            result["repo_type"] = "intent"; return result
        if part == "async-api-assets":
            result["repo_type"] = "async_api"; return result
    return result


class TMForumSource(MarkerMixin):
    """Loads, cleans and extracts metadata from TM Forum documents."""

    def __init__(self, use_marker: bool = False):
        super().__init__(use_marker=use_marker)

    def supported_extensions(self) -> list[str]:
        return [".docx", ".pdf", ".json", ".yaml", ".yml", ".md"]

    def load(self, path: Path) -> str:
        path = Path(path)
        for part in path.parts:
            if part in EXCLUDED_PATH_PARTS:
                return ""
        if path.name.startswith("~$"):
            return ""
        suffix = path.suffix.lower()
        if suffix == ".docx":
            return self._load_docx(path)
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".json":
            return self._load_swagger_json(path)
        if suffix in (".yaml", ".yml"):
            return self._load_yaml(path)
        if suffix == ".md":
            return path.read_text(encoding="utf-8", errors="ignore")
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def clean(self, text: str) -> str:
        text = base_clean(text)
        text = re.sub(r'TM Forum Confidential.*?\n', '', text)
        text = re.sub(r'©\s*TM Forum.*?\n', '', text)
        text = re.sub(r'TM Forum \d{4}.*?\n', '', text)
        text = re.sub(r'Revision History.*?(?=\n#{1,3}\s)', '', text, flags=re.DOTALL)
        text = re.sub(r'NOTICE.*?(?=\n#{1,3}\s)', '', text, flags=re.DOTALL)
        text = re.sub(r'^Figure \d+.*?$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^Table \d+.*?$',  '', text, flags=re.MULTILINE)
        text = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        path = Path(path)
        meta = {
            "source":      path.name,
            "filename":    path.name,
            "source_type": "tmforum",
            "folder":      path.parent.name,
        }
        repo_info = _classify_repo(path)
        meta.update(repo_info)

        suffix   = path.suffix.lower()
        name     = path.stem
        name_low = name.lower()

        if "tmf_number" not in meta:
            m = re.search(r'(TMF\d+)', name, re.IGNORECASE)
            if m: meta["tmf_number"] = m.group(1).upper()

        if "tmfc_number" not in meta:
            m = re.search(r'(TMFC\d+)', name, re.IGNORECASE)
            if m: meta["tmfc_number"] = m.group(1).upper()

        if "ig_number" not in meta:
            m = re.match(r'(IG\d+)([A-Z])?', name, re.IGNORECASE)
            if m:
                meta["ig_number"] = m.group(1).upper()
                if m.group(2): meta["ig_variant"] = m.group(2).upper()

        v_match = re.search(r'[_\-]v([\d.]+)', name, re.IGNORECASE)
        if v_match:
            meta["version"] = f"v{v_match.group(1)}"

        if suffix in (".json", ".yaml", ".yml"):
            meta["doc_type"] = "OpenAPI Specification"
            if "schema" in name_low or meta.get("repo_type") == "schema":
                meta["doc_type"] = "Data Schema"
                meta["entity"]   = re.sub(r'\.schema$', '', name, flags=re.IGNORECASE)
        elif suffix in (".docx", ".pdf"):
            meta["doc_type"] = "Document"
        elif suffix == ".md":
            meta["doc_type"] = "README"

        return meta

    # ── Loaders ───────────────────────────────────────────────────────────

    def _load_docx(self, path: Path) -> str:
        # DOCX uses python-docx only - Marker requires WeasyPrint/GTK (not available on Windows)
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

    def _load_yaml(self, path: Path) -> str:
        try:
            import yaml
        except ImportError:
            raise ImportError("pyyaml is required. Install with: pip install pyyaml")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return str(data) if data else ""
        if "openapi" in data or "swagger" in data or "paths" in data:
            return self._spec_to_prose(data, path)
        if "componentMetadata" in data or "coreFunction" in data:
            return self._oda_component_to_prose(data, path)
        lines = [f"# {path.stem}"]
        for k, v in data.items():
            lines.append(f"## {k}")
            lines.append(str(v))
        return "\n\n".join(lines)

    def _load_swagger_json(self, path: Path) -> str:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
        if not isinstance(spec, dict):
            return ""
        return self._spec_to_prose(spec, path)

    def _spec_to_prose(self, spec: dict, path: Path) -> str:
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
                if not isinstance(path_item, dict): continue
                for method, operation in path_item.items():
                    if method.startswith("x-") or method == "parameters": continue
                    if not isinstance(operation, dict): continue
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
        definitions = spec.get("definitions", spec.get("components", {}).get("schemas", {}))
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
                            f"  - {prop_name} "
                            f"({prop.get('type', prop.get('$ref',''))}): "
                            f"{prop.get('description','')}"
                        )
                lines.append("")
        return "\n".join(lines)

    def _oda_component_to_prose(self, spec: dict, path: Path) -> str:
        lines   = []
        meta    = spec.get("componentMetadata", {})
        name    = meta.get("name", path.stem)
        comp_id = meta.get("id", "")
        version = meta.get("version", "")
        desc    = meta.get("description", "")
        lines.append(f"# ODA Component: {name} ({comp_id} v{version})")
        if desc: lines.append(f"\n{desc}\n")
        core    = spec.get("coreFunction", {})
        exposed = core.get("exposedAPIs", [])
        if exposed:
            lines.append("## Exposed APIs\n")
            for api in exposed:
                if not isinstance(api, dict): continue
                lines.append(
                    f"- {api.get('name','')} "
                    f"(v{api.get('version','')}, {api.get('required','optional')}): "
                    f"{api.get('description','')}"
                )
        dependent = core.get("dependentAPIs", [])
        if dependent:
            lines.append("\n## Dependent APIs\n")
            for api in dependent:
                if not isinstance(api, dict): continue
                lines.append(
                    f"- {api.get('name','')} "
                    f"(v{api.get('version','')}, {api.get('required','optional')}): "
                    f"{api.get('description','')}"
                )
        mgmt      = spec.get("managementFunction", {})
        mgmt_apis = mgmt.get("exposedAPIs", [])
        if mgmt_apis:
            lines.append("\n## Management APIs\n")
            for api in mgmt_apis:
                if not isinstance(api, dict): continue
                lines.append(f"- {api.get('name','')} v{api.get('version','')}")
        return "\n".join(lines)