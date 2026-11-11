# TelcoRetrieve Challenge
 
In the TelcoRetrieve Challenge you answer telecom questions with a retrieval pipeline and a small language model. The `open-telco-rag challenge` commands turn a question file into a submission file, check it and score it.
 
Submit your file on [Zindi](https://zindi.world/competitions/telcoretrieve-challenge-unlocking-telco-knowledge).
 
Score = 0.7 × accuracy + 0.3 × token efficiency, where token efficiency = 100 × B / (B + T). B is 850 tokens per question and T is the total prompt tokens. Minimum accuracy is 80%. Every `input_prompt` must be at most 4,096 tokens.
 
## 1. Install
 
```bash
git clone https://github.com/Znbne/telco-retrieve
cd telco-retrieve
pip install -e ".[local-embed]"
```
 
For a specific GPU or a CPU-only PyTorch, see [Installation](../reference/installation.md).
 
## 2. Get the questions and the index
 
Download `Test.csv` from the [Zindi challenge page](https://zindi.world/competitions/telcoretrieve-challenge-unlocking-telco-knowledge) (Data tab). It has the columns `ID,question,options` and no answers.
 
The index for the default settings is in [`GSMA/telco-retrieve-chunks`](https://huggingface.co/datasets/GSMA/telco-retrieve-chunks), folder `telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b` (about 10 GB).
 
```bash
open-telco-rag pull-hf GSMA/telco-retrieve-chunks \
    telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b --local-dir ./runs
```
 
The database is in `./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db`. Its collection is `otel-0.6b_text_baseline_b512_v256_full`.
 
To build your own index, see [Build your own index](#build-your-own-index).
 
## 3. Start the generator
 
Serve `google/gemma-3n-E4B-it` with an OpenAI-compatible server such as vLLM. Then set its URL, which ends in `/v1`.
 
```bash
export LOCAL_LLM_BASE_URL=http://localhost:8000/v1
export LOCAL_LLM_API_KEY=...        # only if your server needs one
```
 
`--llm-base-url` does the same as the variable. `--generator-model` (or `LOCAL_LLM_MODEL`) sets the model name sent to the server.
 
## 4. Export, check and score
 
```bash
DB=./runs/telcoretrievechallenge/fixed-grid/text_baseline/otel-0.6b/db
open-telco-rag challenge export --questions Test.csv --out submission.csv \
    --db-path $DB --collection otel-0.6b_text_baseline_b512_v256_full
open-telco-rag challenge check --submission submission.csv --questions Test.csv
open-telco-rag challenge score --submission submission.csv --reference MyAnswers.csv
```
 
`export` writes one row per question in the file you give it.
 
- `export` retrieves passages, builds the prompt, calls the generator and writes the submission. It prints its settings, progress every 25 questions and a last line with the rows, prompt tokens and empty answers.
- `check` validates the file against the questions: header, one row per ID, valid answers, UTF-8 without BOM, LF line endings and the prompt length limit. It exits with status 1 on an error.
- `score` prints accuracy, token efficiency and the overall score against an answer file with the columns `ID,answer`. Use it with your own answers. The leaderboard score is computed by Zindi.
## The submission file
 
The columns are `ID,input_prompt,answer`.
 
- `input_prompt` is the system message, a blank line and the user message, as sent to the generator.
- `answer` is an option letter for multiple choice, or `TRUE` or `FALSE`.
## How tokens are counted
 
A prompt's length is its number of `cl100k_base` tokens (tiktoken). The count covers the whole `input_prompt`: system message, passages, question, options and answer instruction. If a prompt is over `--max-prompt-tokens`, `export` drops the lowest-ranked passages until it fits. Use `--top-k` and `--rerank-top-k` to change how many passages are used.
 
## Default settings
 
These come from [`configs/profiles/challenge.json`](../../configs/profiles/challenge.json). Flags override them.
 
| Setting | Default | Flag |
|---|---|---|
| Embedder | `otel-0.6b` | `--model` |
| Reranker | `otel-reranker-0.6b`, keeps 5 | `--rerank-top-k`, `--no-rerank` |
| Retrieval | hybrid, 10 passages | `--search-method`, `--top-k` |
| Generator | `google/gemma-3n-E4B-it` | `--generator-model` |
| Sampling | temperature 0, seed 1234 | `--temperature`, `--seed` |
 
`open-telco-rag challenge export --help` lists every flag. For your own embedder, prompt builder or answer parser, see [challenge options](../reference/challenge-options.md).
 
## Build your own index
 
```bash
open-telco-rag ingest 3gpp --data DATA --model otel-0.6b --distance-metric cosine \
    --db-path db --collection 3gpp_otel
```
 
`ingest` uses `minilm` and `l2` by default. The published index uses cosine distance, so `--distance-metric cosine` gives your index the same behaviour. `export` works with any metric.
 
## FAQ
 
**How do I resume a stopped export?** Run the same command with `--resume`. It continues from `<out>.partial`.
 
**`export` says the embedder does not match the index.** Pass the model that built the index, for example `--model otel-0.6b`.
 
**`check` reports an answer that is not an option.** Look at the empty-answer count on the last `export` line, or set `--answer-parser`.
 
**`check` reports a BOM or CR characters.** Write the CSV with LF line endings and no BOM. Do not re-save it in a spreadsheet program.
 
**Rows go missing after I opened the CSV.** IDs such as `123456789012` are text. Do not open or save the file in Excel, which turns them into numbers.
 
