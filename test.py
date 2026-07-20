import json
import requests

# Your provided model payload
models_data = {
    "Claude Sonnet 4.5": {
        "model_name": "anthropic/claude-sonnet-4.5",
        "description": "State-of-the-art technical analysis and reasoning"
    },
    "Claude Sonnet 4": {
        "model_name": "anthropic/claude-sonnet-4",
        "description": "High-reliability long-context reasoning baseline"
    },
    "Claude Haiku 4.5": {
        "model_name": "anthropic/claude-haiku-4-5",
        "description": "Fast, high-efficiency lightweight analytical runner"
    },
    "GPT-4o": {
        "model_name": "openai/gpt-4o",
        "description": "Commercial frontier baseline generalist model"
    },
    "GPT-4o mini": {
        "model_name": "openai/gpt-4o-mini",
        "description": "Highly cost-effective fast processing engine"
    },
    "Llama 3.3 70B": {
        "model_name": "meta-llama/llama-3.3-70b-instruct",
        "description": "Premier open-weights technical reasoning model"
    },
    "Llama 3.1 8B": {
        "model_name": "meta-llama/llama-3.1-8b-instruct",
        "description": "Small-footprint local/edge emulation baseline"
    },
    "Qwen 2.5 7B": {
        "model_name": "qwen/qwen-2.5-7b-instruct",
        "description": "Highly performant compact code/spec assistant"
    },
    "Gemma 3n E4B": {
        "model_name": "google/gemma-3n-e4b-it",
        "description": "Next-gen experimental efficient instruction model"
    },
    "Mistral Large": {
        "model_name": "mistralai/mistral-large",
        "description": "Strong European multilingual frontier standard"
    }
}

def verify_openrouter_endpoints(models):
    print("Fetching active models from OpenRouter...")
    try:
        # OpenRouter's public endpoint for listing active models
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        
        # Extract all available model IDs into a set for O(1) lookups
        available_models = {model['id'] for model in response.json().get('data', [])}
        
    except Exception as e:
        print(f"Error fetching data from OpenRouter: {e}")
        return

    print("\n--- Endpoint Verification Results ---")
    valid_count = 0
    
    for friendly_name, info in models.items():
        slug = info.get("model_name")
        if slug in available_models:
            print(f"✅ [VALID]   {friendly_name} -> '{slug}'")
            valid_count += 1
        else:
            print(f"❌ [INVALID] {friendly_name} -> '{slug}'")
            
    print(f"\nVerification complete: {valid_count}/{len(models)} endpoints are active.")

if __name__ == "__main__":
    verify_openrouter_endpoints(models_data)