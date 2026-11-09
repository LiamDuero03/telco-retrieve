# Preparing the GSMA/3GPP Rel-19 data

`scripts/stage_3gpp.py` downloads the Markdown conversions of the Rel-19 specifications from the
[`GSMA/3GPP`](https://huggingface.co/datasets/GSMA/3GPP) dataset, checks them, and stages them under names that the 3GPP
loader reads: `<out>/Rel-19/<NN>_series/<spec>-<version>.md`. It downloads only `marked/Rel-19/**/raw.md` and
`conversion_log.jsonl`, never the ~32,760 figure images. It is opt-in: the loader and the default behaviour are unchanged.
The files are downloaded from the original source onto your machine; this repository does not contain or redistribute
them. Read the dataset's licence and the 3GPP terms before you use the data.

Install the package first (`pip install -e ".[local-embed,pdf,docx]"` from the repository root, see [Installation](installation.md)): the
script imports `open_telco_rag`, and without the install it stops with `ModuleNotFoundError`.

```bash
python scripts/stage_3gpp.py --revision a056f6018a7e8e67052aa68a702e272d0ae95d75 --out data/3gpp/staged --dry-run   # plan only
python scripts/stage_3gpp.py --revision a056f6018a7e8e67052aa68a702e272d0ae95d75 --out data/3gpp/staged
python scripts/stage_3gpp.py --revision a056f6018a7e8e67052aa68a702e272d0ae95d75 --out data/3gpp/staged --verify    # offline re-check
open-telco-rag ingest 3gpp --data data/3gpp/staged --model bge-m3 \
  --strategy hierarchical_markdown --chunk-size 2048 --enrichment none --distance-metric cosine \
  --bound-tokens 512 \
  --bound-verify minilm=256,bge=512,bge-m3=512,qwen3-embedding-0.6b=512,embeddinggemma-300m=512 \
  --bound-outliers split
```

**Why the bound.** Without `--bound-tokens`, chunks of the staged set reach about 89,000 tokens and 46.6% of them exceed
minilm's 256-token limit (measured on the Rel-19 set in a GPU test run). The options
above cut every chunk to at most 512 tokens and, with `--bound-verify ... --bound-outliers split`, cut further any chunk
that is still over another embedder's limit. Repeat the command for each embedder and change only `--model`: the
`--bound-*` options, with the same `--bound-verify` list, must be identical for every embedder, so that all collections
hold identical chunks and their results can be compared.

- `--revision` must be a 40-character commit hash (a branch name is refused), so that everyone stages the same files.
  The one above is the revision the counts below were measured on.
- Expected for that revision: 1,566 conversion-log rows (1,557 OK, 9 ERROR), 1,557 `raw.md` files and 1,557 staged files;
  67 multi-part specs; 20 specs with a `_cover` file name; 49 `.doc` log rows; 7 specs whose first conversion failed and
  a later one worked; one spec with only failed conversions and no file (TS 25.331).
- Stubs: files smaller than `--stub-bytes` (default 5,120 bytes) are flagged `is_stub` in the manifest and still staged.
  8 specs are under 5,000 bytes (38.460, 38.462, 38.461, 25.461, 37.481, 22.242, 25.431, 38.421) and 11 under 5,120 bytes
  (the same eight plus 38.471, 41.101 and 25.421). They hold little more than a cover and a contents list.
- Time and disk: `--dry-run` takes about 6 to 7 minutes, because it lists the whole tree (about 34,000 files, in pages of
  50) to read the size and hash of each spec file; it downloads no spec file. The real run then downloads about 0.73 GB in
  parallel (`--workers`) into `<out>_raw`, next to `--out`, not inside it, so that `ingest --data <out>` never reads it, and
  stages the same again: about 1.5 GB in total. A run that stops can be started again.
- The dataset is public. If you set `HF_TOKEN` in the environment it is used and never printed. A token does not
  prevent HTTP 429 (rate limit) answers: the script, and the Hugging Face library under it, retry with backoff, and the
  retries are counted in `staging_run.json`.
- For each spec the last conversion-log row with status OK decides the version (`38300-j20.docx` is V19.2.0). The
  version is compared with a `V<major>.<minor>.<patch>` string in the first 40 lines of the file; a mismatch is
  counted in the manifest, never an error.
- `manifest.json` records for every file its source path, staged name, size, sha256, version and part, and lists the excluded
  specs. It has no timestamps, no local paths and no run-dependent numbers: for the same revision it is byte-identical
  whether the files were downloaded or staged with `--source` from an already downloaded tree (with the same `raw.md` files
  and log). What differs from run to run (finish time, retry counts, the number of figure files in the listing) is in
  `staging_run.json`.
- `--source DIR` stages from an already downloaded tree with the dataset's layout, offline.

## Tips

- **Compare embedders with identical chunks.** Use the same `--bound-*` options and the same `--bound-verify` list for every embedder. minilm's 256-token limit is the strictest in the list above; if you do not use minilm, remove it from the list for every embedder.
- **Multi-part specs.** A spec with a part (for example `32111-1`) is staged as `32111-1.md`. Its chunk metadata has `doc_part`, and the version is in the manifest. For study reports (`23700-75`) the part is a study-item number, marked `doc_part_kind: study_item_number` in the manifest.
- **GPU and EmbeddingGemma.** Embedders load in float16 on a GPU. For `embeddinggemma-300m` use the key `embeddinggemma-300m-fp32` there, because float16 can give NaN vectors (see [embedding reference](embedding.md)). If a batch contains NaN or infinite values, `ingest` stops before writing it, names the chunk ids, and exits with code 4 when you pass `--fail-on-error`.
- **Check the result of a run** with the printed summary and the exit code. The per-file log `logs/ingest_*.csv` is written after the bulk write succeeds.
- **Paths.** The loader reads the release and the series from any `Rel-NN` or `NN_series` folder in the whole path, so do not put `--out` below such a folder (the script warns).

## Next step

Search and ask with the new collection: see the [CLI reference](cli.md).
