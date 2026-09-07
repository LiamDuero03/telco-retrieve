"""CAMARA API specification source - full Fall25 corpus."""

from __future__ import annotations

import re
import yaml
from pathlib import Path

from open_telco_rag.core.cleaning import base_clean


class CAMARASource:
    """Loads, cleans and extracts metadata from CAMARA API repositories.

    Now covers all 90 public repos from camaraproject organisation (Fall25).
    Exclusion rules are expanded to handle boilerplate across the full org.
    """

    SUPPORTED_EXTENSIONS = [".md", ".yaml", ".yml"]

    def supported_extensions(self) -> list[str]:
        return self.SUPPORTED_EXTENSIONS

    # ── File-level exclusions ─────────────────────────────────────────────────
    EXCLUDED_FILES = {
        # Governance / admin
        "CHANGELOG.md", "CODEOWNERS", "GOVERNANCE.MD", "GOVERNANCE.md",
        "MAINTAINERS.MD", "MAINTAINERS.md", "LICENSE", "LICENSE.md",
        "PARTICIPANTS.MD", "PARTICIPANTS.md",
        "CONTRIBUTING.md", "CONTRIBUTING.MD",
        "SECURITY.md", "SECURITY.MD",
        # Version / release tracking
        "VERSION.yaml", "release-plan.yaml",
        # Tooling config
        ".yamllint.yaml", "redocly.yaml", ".redocly.yaml",
        # Common schema fragments (not standalone API specs)
        "CAMARA_common.yaml", "NAM_Common.yaml",
        "commonalities-adoption.yaml",
        # Templates
        "event-subscription-template.yaml",
        "notification-as-cloud-event.yaml",
        "CHANGELOG-r1.md",  # variant changelog filename
    }

    # ── Pattern-based exclusions (matched against filename) ───────────────────
    EXCLUDED_PATTERNS = [
        r'\d{4}_Week\d+_syncCalls\.md',   # weekly sync meeting notes
        r'\d{4}-\d{2}-\d{2}.*\.md',       # date-prefixed meeting minutes
        r'^linting.*\.yaml$',             # linting configs
        r'^\..*',                          # hidden files
        r'MoM_\d+\.md',              # e.g. MoM_231023.md
        r'MOM-\d{4}-\d+.*\.md',      # e.g. MOM-2024-01-18.md
        r'^issue_.*\.md$',           # issue templates
        r'^pull_request_.*\.md$',    # PR templates
        r'^CHANGELOG.*\.md$',        # changelog variants
    ]

    # ── Path component exclusions (any path component matching -> skip) ─────────
    EXCLUDED_PATH_PARTS = {
        ".github",
        "test_definitions",
        "Test_definitions",
        "TEST_DEFINITIONS",
        "node_modules",
        ".git",
        "venv",
        "__pycache__",
        # Non-API repos cloned but not useful for ingestion
        "WorkingGroups",
        "EasyCLA",
        "test-repo-w-linting",
        "test-repo-wo-linting",
        "TestRepo",
        "camaraproject.github.io",
    }

    def load(self, path: Path) -> str:
        """Load Markdown or YAML file as text."""
        path = Path(path)

        if path.name in self.EXCLUDED_FILES:
            return ""
        if any(re.search(p, path.name, re.IGNORECASE)
               for p in self.EXCLUDED_PATTERNS):
            return ""

        for part in path.parts:
            if part in self.EXCLUDED_PATH_PARTS:
                return ""

        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return self._load_yaml(path)
        if suffix == ".md":
            return path.read_text(encoding="utf-8", errors="ignore")
        raise ValueError(f"Unsupported file type: {path.suffix}")

    def clean(self, text: str) -> str:
        """Apply CAMARA-specific cleaning on top of base cleaning."""
        text = base_clean(text)
        # HTML badge markup from READMEs
        text = re.sub(r'<a\b[^>]*>.*?</a>', '', text, flags=re.DOTALL)
        text = re.sub(r'<img\b[^>]*/?>',    '', text)
        text = re.sub(r'<[^>]+>',           '', text)
        # Markdown badge lines
        text = re.sub(
            r'!\[.*?\]\(https://img\.shields\.io[^\)]*\)', '', text
        )
        # Standalone URLs
        text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)
        # YAML front matter
        text = re.sub(r'^---\n.*?\n---\n', '', text, flags=re.DOTALL)
        # HTML-encoded characters
        text = (text.replace('â€œ', '"')
                    .replace('â€', '"')
                    .replace('â€™', "'"))
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def metadata(self, path: Path) -> dict:
        """Extract API name, version, repo name, and file role from path."""
        path = Path(path)
        meta = {
            "source":      path.name,
            "filename":    path.name,
            "source_type": "camara",
        }

        # ── Repo name (top-level CAMARA repo folder) ──────────────────────
        # First CamelCase or known-pattern dir after the data/camara root
        repo_name = None
        for part in path.parts:
            if (part.lower() not in {
                    "data", "camara", "code", "api_definitions",
                    "documentation", "artifacts", ".github",
                    "test_definitions", "swaggers"
                }
                and not part.startswith(".")
                and re.match(r'[A-Z]', part)):
                repo_name = part
                break

        if repo_name:
            meta["api_name"]  = repo_name
            meta["repo_name"] = repo_name

        # ── File role ─────────────────────────────────────────────────────
        suffix   = path.suffix.lower()
        name_low = path.name.lower()

        if suffix in (".yaml", ".yml"):
            meta["file_type"] = "openapi"
            meta["spec_name"] = path.stem
            try:
                with open(path, encoding="utf-8", errors="ignore") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    info = data.get("info", {})
                    if info.get("version"):
                        meta["api_version"] = info["version"]
                    if info.get("title"):
                        meta["api_title"] = info["title"]
            except Exception:
                pass

        elif suffix == ".md":
            if name_low == "readme.md":
                meta["file_type"] = "readme"
                meta["doc_role"]  = "api_overview"
            elif "user_story" in name_low or "userstory" in name_low:
                meta["file_type"] = "user_story"
                meta["doc_role"]  = "use_case"
            elif "mom" in name_low or "meeting" in name_low or "minutes" in name_low:
                meta["file_type"] = "meeting_minutes"
                meta["doc_role"]  = "design_history"
            elif "changelog" in name_low:
                meta["file_type"] = "changelog"
                meta["doc_role"]  = "version_history"
            else:
                meta["file_type"] = "documentation"
                meta["doc_role"]  = "reference"

        return meta

    # ── Private loaders ───────────────────────────────────────────────────────

    def _load_yaml(self, path: Path) -> str:
        """Convert OpenAPI YAML to readable text for chunking."""
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                data = yaml.safe_load(f)
        except Exception:
            return path.read_text(encoding="utf-8", errors="ignore")

        if not isinstance(data, dict):
            return path.read_text(encoding="utf-8", errors="ignore")

        sections = []

        info = data.get("info", {})
        if info.get("title"):
            sections.append(f"# {info['title']}")
        if info.get("version"):
            sections.append(f"Version: {info['version']}")
        if info.get("description"):
            sections.append(info["description"])

        servers = data.get("servers", [])
        if servers:
            sections.append("## Servers")
            for s in servers:
                desc = s.get("description", "")
                url  = s.get("url", "")
                sections.append(f"- {desc}: {url}" if desc else f"- {url}")

        paths = data.get("paths", {})
        for endpoint, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, details in methods.items():
                if method.startswith("x-") or not isinstance(details, dict):
                    continue
                summary     = details.get("summary", "")
                description = details.get("description", "")
                operation   = f"## {method.upper()} {endpoint}"
                if summary:
                    operation += f" — {summary}"
                sections.append(operation)
                if description:
                    sections.append(description)

                params = details.get("parameters", [])
                if params:
                    param_lines = ["**Parameters:**"]
                    for p in params:
                        if isinstance(p, dict):
                            name     = p.get("name", "")
                            loc      = p.get("in", "")
                            required = "required" if p.get("required") else "optional"
                            desc     = p.get("description", "")
                            param_lines.append(
                                f"- `{name}` ({loc}, {required}): {desc}"
                            )
                    sections.append("\n".join(param_lines))

                req_body = details.get("requestBody", {})
                if req_body and isinstance(req_body, dict):
                    rb_desc = req_body.get("description", "")
                    if rb_desc:
                        sections.append(f"**Request body:** {rb_desc}")

                responses = details.get("responses", {})
                if responses:
                    resp_lines = ["**Responses:**"]
                    for code, resp in responses.items():
                        if isinstance(resp, dict):
                            resp_desc = resp.get("description", "")
                            resp_lines.append(f"- {code}: {resp_desc}")
                    sections.append("\n".join(resp_lines))

        components = data.get("components", {})
        schemas    = components.get("schemas", {})
        if schemas:
            sections.append("## Data Models")
            for name, schema in schemas.items():
                if not isinstance(schema, dict):
                    continue
                desc = schema.get("description", "")
                if desc:
                    sections.append(f"### {name}\n{desc}")

        return "\n\n".join(sections)