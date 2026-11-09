# Python API


```python
from open_telco_rag.sources.oran import ORANSource
from open_telco_rag.embedders.openrouter_embedder import OpenRouterEmbedder
from open_telco_rag.stores.chroma import ChromaStore
from open_telco_rag.generators.openrouter import OpenRouterGenerator
from open_telco_rag.generators.local_llm import LocalLLMClient
from open_telco_rag.core.pipeline import ingest, search, generate

# Ingest (Using OpenRouter for High-Throughput Bulk Processing)
source   = ORANSource()
embedder = OpenRouterEmbedder(model_key="bge", batch_size=200) # Throttled safely for OpenAI limits
store    = ChromaStore(path="chroma_db", collection_name="oran_bge", metadata={"hnsw:space": "cosine"})

# llm_metadata needs an LLM client explicitly passed as enrichment_llm_client
# — LocalLLMClient talks to any OpenAI-compatible server (vLLM, Ollama, etc.);
# OpenRouterLLMClient (same module) is the cloud alternative.
llm_client = LocalLLMClient(base_url="http://localhost:8000/v1")

ingest(
    source=source, 
    store=store, 
    embedder=embedder, 
    data_dir="data/oran/",
    strategy="sliding_window_tokens",
    chunk_tokens=512,
    overlap_tokens=50,
    enrichment="llm_metadata",
    enrichment_model="Qwen/Qwen2.5-32B-Instruct",
    enrichment_llm_client=llm_client,
    batch_size=5000 # Feed 5,000 chunks concurrently to the embedder!
)

# Search
results = search("E2 interface service models", store=store, embedder=embedder, top_k=5, search_method="dense")

# Generate
generator = OpenRouterGenerator(model="meta-llama/llama-3.3-70b-instruct")
answer    = generate("What does WG3 define for the E2 interface?", results, generator=generator)
print(answer["answer"])
```
