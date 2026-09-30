"""Cloud embedder backend: batches text through OpenRouter's embeddings
API for models not run locally via sentence-transformers."""
import os
import json
import threading
import time
from pathlib import Path
from typing import List
import requests
from requests.adapters import HTTPAdapter

class OpenRouterEmbedder:
    """Embedder implementation (see types.Embedder) backed by OpenRouter's
    embeddings API, resolving model_key against configs/embedding_models.json."""

    def __init__(
        self,
        model_key: str = "minilm",
        models_file: str | Path = Path(__file__).parent.parent.parent / "configs" / "embedding_models.json",
        # kept low to stay under OpenAI's per-request payload limit
        batch_size: int = 200,
        # in-flight batch requests; exposed (not just a local default) so
        # calibration/throughput runs can compare concurrency levels using
        # this exact code path instead of a bespoke harness.
        max_workers: int = 10,
        # per-request timeout in seconds; a stalled connection used to block forever
        timeout: float = 120.0,
    ):
        models_file = Path(models_file)
        if not models_file.exists():
            raise FileNotFoundError(f"Embedding models file not found: {models_file}")

        with open(models_file) as f:
            available = json.load(f)

        if model_key not in available:
            raise ValueError(f"Model key '{model_key}' not in {models_file}.")

        self._model_key   = model_key
        self._model_name  = available[model_key]["model_name"]
        self._dimensions  = available[model_key]["dimensions"]
        self.batch_size   = batch_size
        self.max_workers  = max_workers
        self.timeout      = timeout
        # Plain string prefixes (e.g. qwen3-embedding-8b's instruction-style
        # query prefix) applied only when set -- see SentenceTransformerEmbedder
        # for the same mechanism used on the local-serving side.
        self._query_prefix    = available[model_key].get("query_prefix")
        self._document_prefix = available[model_key].get("document_prefix")
        
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
            
        self.url = "https://openrouter.ai/api/v1/embeddings"
        
        # shared session for thread-safe connection pooling
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount('https://', adapter)

        # Real usage accounting from OpenRouter's own response body (not a
        # local token/price estimate) -- accumulated across every embed()
        # call this instance makes; process_batch() updates these under
        # _usage_lock since batches run concurrently.
        self.total_prompt_tokens = 0
        self.total_cost_usd = 0.0
        self._usage_lock = threading.Lock()

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _apply_prefix(self, texts: List[str], is_query: bool) -> List[str]:
        prefix = self._query_prefix if is_query else self._document_prefix
        if not prefix:
            return texts
        return [f"{prefix}{t}" for t in texts]

    def embed(self, texts: list[str], is_query: bool = False,
              apply_prefix: bool = True) -> list[list[float]]:
        import concurrent.futures

        if apply_prefix:
            texts = self._apply_prefix(texts, is_query)
        all_embeddings_dict = {}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        def process_batch(start_idx, batch_texts):
            payload = {
                "model": self._model_name,
                "input": batch_texts
            }
            retries = 6
            backoff = 2
            last_error = "no response"

            for attempt in range(retries):
                try:
                    response = self.session.post(self.url, headers=headers, json=payload,
                                                 timeout=self.timeout)
                except requests.exceptions.RequestException as e:
                    # timeouts and dropped connections are transient: retry with backoff
                    last_error = f"{type(e).__name__}: {e}"
                    print(f"  [Network Error] {last_error}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                if response.status_code == 200:
                    try:
                        json_resp = response.json()
                    except ValueError:
                        last_error = "response was not valid JSON"
                        print(f"  [API Error] {last_error}. Retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue

                    # OpenRouter can return HTTP 200 with an error embedded in the body
                    if "error" in json_resp and json_resp["error"]:
                        err = json_resp["error"]
                        error_msg = err.get("message", "Unknown provider error") if isinstance(err, dict) else str(err)
                        last_error = error_msg
                        print(f"  [API Error] OpenRouter returned an error: {error_msg}. Retrying...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue

                    data = json_resp.get("data") or []
                    if len(data) != len(batch_texts):
                        # a short (or long) answer would silently misalign vectors and chunks
                        last_error = f"got {len(data)} embeddings for {len(batch_texts)} inputs"
                        print(f"  [API Error] {last_error}. Retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue

                    usage = json_resp.get("usage") or {}
                    if usage:
                        with self._usage_lock:
                            self.total_prompt_tokens += usage.get("prompt_tokens") or 0
                            self.total_cost_usd += usage.get("cost") or 0.0

                    data_sorted = sorted(data, key=lambda x: x["index"])
                    return start_idx, [item["embedding"] for item in data_sorted]

                elif response.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    label = "Rate Limited" if response.status_code == 429 else "Server Error"
                    print(f"  [{label}] OpenRouter returned {response.status_code}. Sleeping for {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    response.raise_for_status()   # other 4xx: not transient, do not retry

            raise Exception(f"Failed to fetch embeddings after {retries} attempts (last error: {last_error}).")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                futures[executor.submit(process_batch, i, batch)] = i
                
            for future in concurrent.futures.as_completed(futures):
                start_idx, embeds = future.result()
                all_embeddings_dict[start_idx] = embeds
                
        all_embeddings = []
        for i in sorted(all_embeddings_dict.keys()):
            all_embeddings.extend(all_embeddings_dict[i])
            
        return all_embeddings