"""The staging and install pages of docs/reference: the ingest command it gives must be accepted by the real CLI and the real
bound validation, and the sections must say what a new user needs. No network."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

import cli.main as cli_main
from open_telco_rag.core.chunk_bound import parse_verify, validate_bound

REFERENCE = Path(__file__).resolve().parent.parent / "docs" / "reference"
STAGING_PAGE = (REFERENCE / "3gpp-data.md").read_text(encoding="utf-8")
INSTALL_PAGE = (REFERENCE / "installation.md").read_text(encoding="utf-8")


def _section(page: str, start: str, end: str) -> str:
    i = page.index(start)
    return page[i:page.index(end, i + len(start))]


STAGING = _section(STAGING_PAGE, "# Preparing the GSMA/3GPP Rel-19 data", "## Tips")
INSTALL = _section(INSTALL_PAGE, "# Installation", "## API keys")


def _ingest_command() -> list[str]:
    blocks = re.findall(r"```bash\n(.*?)```", STAGING, re.DOTALL)
    for block in blocks:
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            if line.startswith("open-telco-rag ingest 3gpp"):
                return shlex.split(line)
    raise AssertionError("no ingest command in the staging section")


def test_the_staging_ingest_command_is_the_bounded_one_and_the_cli_accepts_it(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_main, "_run_ingest", lambda args: captured.update(vars(args)))
    cli_main.main(_ingest_command()[1:])
    assert captured["source_type"] == "3gpp" and captured["model"] == "bge-m3"
    assert captured["strategy"] == "hierarchical_markdown" and captured["chunk_size"] == 2048
    assert captured["enrichment"] == "none" and captured["distance_metric"] == "cosine"
    assert captured["bound_tokens"] == 512 and captured["bound_outliers"] == "split"
    assert captured["bound_verify"] == "minilm=256,bge=512,bge-m3=512,qwen3-embedding-0.6b=512,embeddinggemma-300m=512"
    validate_bound(captured["bound_tokens"], captured["bound_tokenizer"], captured["bound_margin"],
                   captured["bound_verify"], captured["bound_outliers"])


def test_every_embedder_in_the_verify_list_is_a_registry_key():
    import json
    registry = json.loads((Path(__file__).resolve().parent.parent / "configs" / "embedding_models.json").read_text(encoding="utf-8"))
    names = [name for name, _ in parse_verify(_ingest_command()[_ingest_command().index("--bound-verify") + 1])]
    assert names and all(name in registry for name in names)


def test_the_section_says_why_and_that_the_verify_list_must_be_identical_for_every_embedder():
    text = " ".join(STAGING.split())
    assert "89,000 tokens" in text and "46.6%" in text and "minilm" in text
    assert "identical for every embedder" in text and "identical chunks" in text


def test_the_staging_section_names_the_install_requirement_and_the_real_output_layout():
    text = " ".join(STAGING.split())
    assert "ModuleNotFoundError" in text and "pip install -e" in text
    assert "<out>/Rel-19/<NN>_series/<spec>" in text


def test_the_staging_section_gets_the_numbers_and_the_run_time_right():
    text = " ".join(STAGING.split())
    assert "5,120 bytes" in text and "8 specs are under 5,000 bytes" in text and "11 under 5,120" in text
    assert "about 6 to 7 minutes" in text and "whole tree" in text
    assert "429" in text and "HF_TOKEN" in text and "staging_run.json" in text
    assert "byte-identical" in text and "--source" in text


def test_the_install_section_puts_torch_before_the_package_for_nvidia_amd_and_cpu():
    text = INSTALL
    for name in ("NVIDIA", "ROCm", "CPU"):
        assert name in text
    first_torch = text.index("pip install torch --index-url")
    package = text.index('pip install -e ".[local-embed,pdf,docx]"', first_torch)
    check = text.index("torch.cuda.is_available()", package)
    assert first_torch < package < check
    assert "torch.version.hip" in text and "torch.version.cuda" in text
    assert "whl/cpu" in text and "whl/rocm" in text and "whl/cu" in text
    assert "pulls" in text and "CUDA" in " ".join(text.split())


def test_the_tips_cover_the_open_points():
    section = _section(STAGING_PAGE, "## Tips", "## Next step")
    text = " ".join(section.split())
    assert "256" in text and "minilm" in text
    assert "`doc_part`" in text and "manifest" in text
    assert "float16" in text and "NaN" in text


@pytest.mark.parametrize("flag", ["--strategy", "--chunk-size", "--enrichment", "--distance-metric", "--bound-tokens",
                                  "--bound-verify", "--bound-outliers", "--model", "--data"])
def test_every_flag_of_the_ingest_command_exists_in_the_cli_help(flag, capsys):
    with pytest.raises(SystemExit):
        cli_main.main(["ingest", "--help"])
    assert flag in capsys.readouterr().out
