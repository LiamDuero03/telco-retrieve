"""Cloud embedder backend: batches text through OpenRouter's embeddings
API for models not run locally via sentence-transformers."""
import os
import json
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
        
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")
            
        self.url = "https://openrouter.ai/api/v1/embeddings"
        
        # shared session for thread-safe connection pooling
        self.session = requests.Session()
        adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
        self.session.mount('https://', adapter)

    @property
    def model_key(self) -> str:
        return self._model_key

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: List[str]) -> List[List[float]]:
        import concurrent.futures
        
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
            
            for attempt in range(retries):
                response = self.session.post(self.url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    json_resp = response.json()
                    
                    # OpenRouter can return HTTP 200 with an error embedded in the body
                    if "error" in json_resp and json_resp["error"]:
                        error_msg = json_resp["error"].get("message", "Unknown provider error")
                        print(f"  [API Error] OpenRouter returned an error: {error_msg}. Retrying...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                        
                    data = json_resp.get("data", [])
                    data_sorted = sorted(data, key=lambda x: x["index"])
                    return start_idx, [item["embedding"] for item in data_sorted]
                    
                elif response.status_code == 429:
                    print(f"  [Rate Limited] OpenRouter returned 429. Sleeping for {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    response.raise_for_status()
                    
            raise Exception(f"Failed to fetch embeddings after {retries} attempts.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
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