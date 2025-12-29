from unsloth import FastLanguageModel
import torch
import os

# Configuration
MODEL_PATH = "/root/autodl-tmp/NLP论文复现/output/qwen3_8b_absa/lora_model"
OUTPUT_PATH = "/root/autodl-tmp/NLP论文复现/output/qwen3_8b_absa/merged_model"

def main():
    print(f"Loading LoRA model from {MODEL_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True, # Load in 4bit initially
    )
    
    print("Merging model to 16bit for vLLM...")
    # Merge to 16bit
    model.save_pretrained_merged(OUTPUT_PATH, tokenizer, save_method="merged_16bit")
    
    print(f"Merged model saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
