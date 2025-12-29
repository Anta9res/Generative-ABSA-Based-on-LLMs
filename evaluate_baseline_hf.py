import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import os
from tqdm import tqdm
import json

# Configuration
MODEL_PATH = "/root/autodl-tmp/data/models/Qwen3-8B"
TEST_DATA_PATH = "/root/autodl-tmp/NLP论文复现/data/processed/test_dataset"
OUTPUT_DIR = "/root/autodl-tmp/NLP论文复现/output/baseline_eval"
MAX_SAMPLES = None # None for all, or integer for testing

def get_prompt_without_label(text, label):
    if "### Response:\n" in text:
        prompt = text.split("### Response:\n")[0] + "### Response:\n"
        return prompt
    else:
        return text.replace(label, "").strip()

def map_output_to_label(output_text):
    output_text = output_text.strip().lower()
    if "positive" in output_text:
        return "Positive"
    elif "negative" in output_text:
        return "Negative"
    elif "neutral" in output_text:
        return "Neutral"
    else:
        return "Unknown"

def main():
    print(f"Loading HF model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Ensure left padding for generation
    tokenizer.padding_side = 'left' 
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    
    print(f"Loading test dataset from {TEST_DATA_PATH}...")
    dataset = load_from_disk(TEST_DATA_PATH)
    
    if MAX_SAMPLES:
        dataset = dataset.select(range(min(len(dataset), MAX_SAMPLES)))
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    prompts = []
    labels = []
    
    for item in dataset:
        prompt = get_prompt_without_label(item['text'], item['label'])
        prompts.append(prompt)
        labels.append(item['label'])
        
    print("Generating predictions...")
    
    batch_size = 8
    predictions = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=16, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        for text in generated_texts:
            predictions.append(map_output_to_label(text))
            
    # Calculate Metrics
    print("\n" + "="*50)
    print("Baseline (Qwen3-8B Zero-shot) Evaluation Results")
    print("="*50)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, predictions, zero_division=0))
    
    # Save results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions[:100], 
        "labels": labels[:100]
    }
    
    with open(os.path.join(OUTPUT_DIR, "baseline_metrics_hf.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
