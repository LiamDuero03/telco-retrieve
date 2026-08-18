"""
Shared Marker-based PDF loader using the Datalab hosted API (raw requests).

Requires:
    pip install requests
    Set DATALAB_API_KEY in your .env file

Usage in any source class:
    from open_telco_rag.core.marker_loader import MarkerMixin

    class ORANSource(MarkerMixin):
        def __init__(self, use_marker: bool = False):
            super().__init__(use_marker=use_marker)

        def _load_pdf(self, path):
            if self._use_marker:
                return self._marker_load(path)
            # existing pypdf fallback ...
"""

from __future__ import annotations
import os
import time
from pathlib import Path


class MarkerMixin:
    """
    Mixin that adds optional Marker-based PDF loading to any source class.
    Uses the Datalab hosted API via raw HTTP requests — no local GPU needed.
    """

    def __init__(self, use_marker: bool = False):
        self._use_marker = use_marker

    def _marker_load(self, path: Path) -> str:
        """Convert a PDF to Markdown using the Datalab Marker API."""
        import requests

        api_key = os.environ.get("DATALAB_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DATALAB_API_KEY not set. Add it to your .env file: "
                "DATALAB_API_KEY=your_key_here"
            )

        headers = {"X-API-Key": api_key}

        with open(path, "rb") as f:
            resp = requests.post(
                "https://www.datalab.to/api/v1/convert",
                files={"file": (path.name, f, "application/pdf")},
                data={"output_format": "markdown", "mode": "balanced"},
                headers=headers,
            )
        resp.raise_for_status()
        check_url = resp.json()["request_check_url"]

        # Poll until complete (max 10 minutes)
        for _ in range(300):
            result = requests.get(check_url, headers=headers).json()
            if result["status"] == "complete":
                return result.get("markdown", "")
            elif result["status"] == "failed":
                raise RuntimeError(
                    f"Datalab API failed on {path.name}: {result.get('error', 'unknown')}"
                )
            time.sleep(2)

        raise TimeoutError(f"Datalab API timed out after 10 minutes for {path.name}")