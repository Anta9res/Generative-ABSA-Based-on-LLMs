#!/usr/bin/env python3
import os
import torch
import json
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import TrainingArguments
from trl import SFTTrainer

# Configuration
CONFIG = {
    'model_path': "/root/autodl-tmp/data/models/Qwen3-8B",
    'dataset_path': "/root/autodl-tmp/NLP论文复现/data/processed/absa_dataset",
    'output_dir': "/root/autodl-tmp/NLP论文复现/output/qwen3_8b_absa",
    'max_seq_length': 2048,
    'epochs': 1,
    'batch_size': 4,
    'grad_accum': 4,
    'learning_rate': 2e-4,
}

def main():
    print("="*50)
    print("Generative ABSA Fine-tuning (Qwen3-8B)")
    print("="*50)
    
    print(f"Loading model from {CONFIG['model_path']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG['model_path'],
        max_seq_length=CONFIG['max_seq_length'],
        dtype=None,
        load_in_4bit=True,
    )

    print("Configuring LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print(f"Loading datasets from {CONFIG['dataset_path']}...")
    if not os.path.exists(CONFIG['dataset_path']):
        raise FileNotFoundError(f"Dataset path {CONFIG['dataset_path']} does not exist. Please run prepare_absa_dataset.py first.")
        
    dataset = load_from_disk(CONFIG['dataset_path'])
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Check data format
    print("Sample input:")
    print(train_dataset[0]['text'][:200] + "...")

    print("Starting Training...")
    training_args = TrainingArguments(
        output_dir=CONFIG['output_dir'],
        per_device_train_batch_size=CONFIG['batch_size'],
        gradient_accumulation_steps=CONFIG['grad_accum'],
        warmup_steps=100,
        num_train_epochs=CONFIG['epochs'],
        learning_rate=CONFIG['learning_rate'],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        save_strategy="steps",
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none", 
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=CONFIG['max_seq_length'],
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    trainer_stats = trainer.train()
    print("Training completed!")
    
    # Save model
    print("Saving model...")
    model.save_pretrained(os.path.join(CONFIG['output_dir'], "lora_model"))
    tokenizer.save_pretrained(os.path.join(CONFIG['output_dir'], "lora_model"))
    
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(eval_results)
    
    with open(os.path.join(CONFIG['output_dir'], "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)

if __name__ == "__main__":
    main()
