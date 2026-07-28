import subprocess
import sys
import os

# --- PATH CONFIGURATION ---
TEST_DB_PATH = "test_chroma_db_sweep"
DATA_PATH = "data/telecom/" 

# --- CORRECTED MODEL MAPPINGS TO MATCH YOUR CONFIG ---
MODEL_MAPPING = {
    "minilm": "minilm",
    "bge-large": "bge",
    "e5-large": "e5",
    "cohere-v3": "or-bge-m3",
    "text-embed-3": "or-3-small",
    "gte-large": "mpnet"
}

MODELS = list(MODEL_MAPPING.keys())
ENRICHMENTS = ["none", "metadata_tagging", "llm_metadata", "acronym_expansion"]
SIZES_MAP = {"S": "200", "M": "400", "L": "800"}

ENRICH_LABELS = {
    "none": "none",
    "metadata_tagging": "metatag",
    "llm_metadata": "llmmmeta",
    "acronym_expansion": "acronym"
}

def run_cli_command(flags: dict):
    """Constructs and executes the entrypoint package binary against your flags."""
    cmd = ["open-telco-rag", "ingest", "3gpp"] 
    
    for flag, value in flags.items():
        cmd.append(flag)
        if value is not None:
            cmd.append(str(value))
            
    print(f"Executing: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error during execution:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

# --- STARTING MATRIX SWEEP GENERATOR ---
run_id = 1

# ==========================================
# PART 1: SLIDING WINDOW TOKENS
# ==========================================
sliding_window_configs = []
for size, overlap in [("256", "25"), ("512", "50"), ("1024", "100")]:
    for enrich in ENRICHMENTS:
        sliding_window_configs.append((size, overlap, enrich))

print("🚀 Initiating sweep matrix on Sliding Window Token sets via package binary...")
for size, overlap, enrich in sliding_window_configs:
    enrich_lbl = ENRICH_LABELS[enrich]
    for model_display_name in MODELS:
        # Get the actual valid key expected by embedding_models.json
        actual_model_key = MODEL_MAPPING[model_display_name]
        collection_name = f"3gpp_tokens_{size}_{enrich_lbl}_{model_display_name}"
        
        flags = {
            "--data": DATA_PATH,
            "--model": actual_model_key,  # Uses valid config string key
            "--db-path": TEST_DB_PATH,
            "--strategy": "sliding_window_tokens",
            "--chunk-size": size,
            "--overlap": overlap,
            "--enrichment": enrich,
            "--distance-metric": "cosine",
            "--collection": collection_name,
            "--embed-backend": "local",
            "--no-resume": None,
        }
        
        print(f"\n[Run {run_id:03d}] Building: {collection_name}")
        run_cli_command(flags)
        run_id += 1


# ==========================================
# PART 2: ADVANCED SEGMENTATION STRATEGIES
# ==========================================
advanced_strategies = [
    ("hierarchical_markdown", "hier"),
    ("parent_child", "parent"),
    ("text_baseline", "base"),
    ("lumber_chunker", "lumber")
]

print("\n🚀 Initiating sweep matrix on Advanced Content Segmentation strategies...")
for strategy, label in advanced_strategies:
    for size_profile, size_val in SIZES_MAP.items():
        for enrich in ENRICHMENTS:
            enrich_lbl = ENRICH_LABELS[enrich]
            for model_display_name in MODELS:
                actual_model_key = MODEL_MAPPING[model_display_name]
                
                if strategy == "lumber_chunker":
                    collection_name = f"3gpp_{label}_theta_{enrich_lbl}_{model_display_name}"
                else:
                    collection_name = f"3gpp_{label}_{size_profile.lower()}_{enrich_lbl}_{model_display_name}"
                
                flags = {
                    "--data": DATA_PATH,
                    "--model": actual_model_key,  # Uses valid config string key
                    "--db-path": TEST_DB_PATH,
                    "--strategy": strategy,
                    "--chunk-size": size_val,
                    "--overlap": str(int(size_val) // 4),
                    "--enrichment": enrich,
                    "--distance-metric": "cosine",
                    "--collection": collection_name,
                    "--embed-backend": "local",
                    "--no-resume": None
                }
                
                print(f"\n[Run {run_id:03d}] Building: {collection_name}")
                run_cli_command(flags)
                run_id += 1

print("\n🌟 Automated validation sweep completed cleanly.")