import json
import os
from datasets import Dataset
import random

# Configuration
INPUT_FILE = "/root/autodl-tmp/NLP论文复现/data/absa_dataset/crypto_absa_silver.json"
OUTPUT_DIR = "/root/autodl-tmp/NLP论文复现/data/processed/absa_dataset"

def format_absa_output(aspects):
    if not aspects:
        return "No specific financial aspects detected."
    
    # Sort for consistent ordering
    sorted_aspects = sorted(aspects, key=lambda x: x['aspect'])
    
    # Format: Aspect: Price, Sentiment: Negative; Aspect: Technology, Sentiment: Positive
    parts = []
    for item in sorted_aspects:
        parts.append(f"Aspect: {item['aspect']}, Sentiment: {item['sentiment']}")
    
    return "; ".join(parts)

def main():
    print(f"Reading {INPUT_FILE}...")
    
    # Check if file exists (it might still be writing)
    if not os.path.exists(INPUT_FILE):
        print(f"Warning: {INPUT_FILE} does not exist yet. Please wait for construction to finish.")
        return

    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. The file might be incomplete.")
        return

    print(f"Loaded {len(data)} annotated samples.")
    
    formatted_data = []
    
    # Statistics for filtering
    total_annotations = 0
    kept_annotations = 0
    
    for item in data:
        original_text = item.get('original_text', '')
        raw_aspects = item.get('absa_annotation', [])
        
        if not original_text:
            continue
            
        # Filter aspects
        valid_aspects = []
        for ann in raw_aspects:
            total_annotations += 1
            evidence = ann.get('evidence', '').lower()
            
            # Heuristic to filter out "not mentioned" hallucinations
            if "no direct mention" in evidence or "not mentioned" in evidence or "no mention" in evidence:
                continue
                
            valid_aspects.append(ann)
            kept_annotations += 1
            
        if not valid_aspects:
            continue
            
        # Create Instruction Tuning Format
        # Instruction: Joint Aspect-Sentiment Extraction
        instruction = "Analyze the following cryptocurrency text and extract financial aspects (Price, Regulation, Technology, Adoption, Mining) with their sentiments. Output format: 'Aspect: [Aspect], Sentiment: [Sentiment]'."
        
        output_text = format_absa_output(valid_aspects)
        
        # Unsloth/Alpaca format
        full_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{original_text}

### Response:
{output_text}"""

        formatted_data.append({
            "text": full_prompt,
            "original_text": original_text,
            "target": output_text
        })
    
    print(f"Formatted {len(formatted_data)} samples.")
    
    # Create Dataset
    full_dataset = Dataset.from_list(formatted_data)
    
    # Split 90/10
    dataset_split = full_dataset.train_test_split(test_size=0.1, seed=42)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    dataset_split.save_to_disk(OUTPUT_DIR)
    print(f"Saved processed ABSA dataset to {OUTPUT_DIR}")
    print(f"Train size: {len(dataset_split['train'])}")
    print(f"Test size: {len(dataset_split['test'])}")

if __name__ == "__main__":
    main()
