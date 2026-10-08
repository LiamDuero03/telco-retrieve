"""The real CLI (cli.main.main) with one thing replaced: the embedding model, which would need a
download. Used by the on-disk smoke test: python -m tests.cli_fake_embedder search "..." --model fake64 ..."""
from __future__ import annotations

import sys

import cli.main as cli_main
from tests.fake_collections import FakeEmbedder

if __name__ == "__main__":
    cli_main._get_embedder = lambda model_key, *args, **kwargs: FakeEmbedder(model_key)
    cli_main.main(sys.argv[1:])
