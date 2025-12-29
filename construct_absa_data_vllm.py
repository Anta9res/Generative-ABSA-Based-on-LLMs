from vllm import LLM, SamplingParams
from datasets import load_from_disk
import json
import os
from tqdm import tqdm

# Configuration
MODEL_PATH = "/root/autodl-tmp/data/models/Qwen3-8B"
INPUT_DATA_PATH = "/root/autodl-tmp/NLP论文复现/data/processed/test_dataset"
OUTPUT_DIR = "/root/autodl-tmp/NLP论文复现/data/absa_dataset"
MAX_SAMPLES = None # Process all samples

def create_prompt(text):
    return f"""You are an expert financial sentiment analyst specializing in cryptocurrencies.
Your task is to analyze the following Reddit post and identify if it contains opinions about specific aspects.

Target Aspects:
1. Price: Price movements, trading, investment returns.
2. Regulation: Government rules, bans, legal issues, SEC.
3. Technology: Network performance, security, upgrades, technical features.
4. Adoption: Usage by companies, countries, or general public.
5. Mining: Mining operations, energy consumption, hardware.

Instructions:
- Only output Valid JSON.
- For each aspect mentioned in the text, determine the sentiment (Positive, Negative, or Neutral).
- Extract a brief snippet as evidence.
- If no aspect is mentioned, output an empty list.

Input Text:
"{text}"

Output Format (JSON):
{{
    "aspects": [
        {{
            "aspect": "Price",
            "sentiment": "Negative",
            "evidence": "prices are crashing hard"
        }}
    ]
}}

JSON Response:"""

def main():
    print(f"Loading vLLM model from {MODEL_PATH}...")
    # Set gpu_memory_utilization to 0.4 to fit alongside training process on 80GB GPU
    # enforce_eager=True helps with compatibility issues in some environments
    llm = LLM(
        model=MODEL_PATH, 
        gpu_memory_utilization=0.4, 
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1
    )
    
    # Remove stop tokens that might trigger on valid JSON start (e.g. ```json)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    print(f"Loading dataset from {INPUT_DATA_PATH}...")
    dataset = load_from_disk(INPUT_DATA_PATH)
    
    # Process valid samples
    prompts = []
    raw_texts = []
    original_labels = []
    
    print("Preparing prompts...")
    # Using the first MAX_SAMPLES
    count = 0
    for item in tqdm(dataset):
        if MAX_SAMPLES is not None and count >= MAX_SAMPLES:
            break
        try:
            # Extract raw text from the training prompt format
            # Format: "...### Input:\n{text}\n\n### Response:..."
            if "### Input:\n" in item['text']:
                raw_text = item['text'].split("### Input:\n")[1].split("\n\n### Response:")[0].strip()
            else:
                continue
                
            prompts.append(create_prompt(raw_text))
            raw_texts.append(raw_text)
            original_labels.append(item.get("label"))
            count += 1
        except Exception as e:
            continue
            
    print(f"Generating for {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    print("Parsing results...")
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        
        # Debug: Print first few outputs
        if i < 5:
            print(f"\n--- Sample {i} ---")
            print(f"Prompt: {prompts[i][:100]}...")
            print(f"Generated: {generated_text}")
            print("------------------")

        try:
            json_str = generated_text.strip()
            # Clean up markdown code blocks if present
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[0]
            
            annotation = json.loads(json_str)
            
            # Store if valid aspects found
            if annotation.get("aspects") and isinstance(annotation["aspects"], list):
                results.append({
                    "original_text": raw_texts[i],
                    "absa_annotation": annotation["aspects"],
                    "original_sentiment": original_labels[i]
                })
        except json.JSONDecodeError:
            continue
        except Exception:
            continue
            
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_file = os.path.join(OUTPUT_DIR, "crypto_absa_silver.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} items to {output_file}")

if __name__ == "__main__":
    main()
