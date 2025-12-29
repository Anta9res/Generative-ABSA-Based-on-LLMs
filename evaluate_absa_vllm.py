from vllm import LLM, SamplingParams
from datasets import load_from_disk
from tqdm import tqdm
import re
import os
import json
from sklearn.metrics import accuracy_score

# Configuration
MODEL_PATH = "/root/autodl-tmp/NLP论文复现/output/qwen3_8b_absa/merged_model"
TEST_DATA_PATH = "/root/autodl-tmp/NLP论文复现/data/processed/absa_dataset"
OUTPUT_DIR = "/root/autodl-tmp/NLP论文复现/output/absa_eval_vllm"

def parse_output(text):
    """
    Parses the generated text: "Aspect: Price, Sentiment: Negative; Aspect: Technology, Sentiment: Positive"
    Returns a list of tuples: [('Price', 'Negative'), ('Technology', 'Positive')]
    """
    # Remove the prompt part if present (vLLM usually returns generated text only, but depends on config)
    if "### Response:\n" in text:
        text = text.split("### Response:\n")[1]
    
    text = text.strip()
    if not text or text == "No specific financial aspects detected.":
        return []
        
    pairs = []
    # Split by semicolon
    parts = text.split(';')
    for part in parts:
        # Expected format: Aspect: X, Sentiment: Y
        # Regex might be safer
        match = re.search(r"Aspect:\s*(.*?),\s*Sentiment:\s*(.*)", part.strip(), re.IGNORECASE)
        if match:
            aspect = match.group(1).strip()
            sentiment = match.group(2).strip()
            pairs.append((aspect, sentiment))
    
    return pairs

def calculate_metrics(gold_list, pred_list):
    """
    gold_list: list of list of (aspect, sentiment) tuples
    pred_list: list of list of (aspect, sentiment) tuples
    """
    # 1. Aspect Extraction Metrics
    tp_aspect = 0
    fp_aspect = 0
    fn_aspect = 0
    
    # 2. Sentiment Classification (Strict: Aspect must match)
    sentiment_gold = []
    sentiment_pred = []
    
    # 3. Joint Extraction
    tp_joint_total = 0
    fp_joint_total = 0
    fn_joint_total = 0
    
    for gold, pred in zip(gold_list, pred_list):
        gold_aspects = set([x[0].lower() for x in gold])
        pred_aspects = set([x[0].lower() for x in pred])
        
        # Aspect Extraction
        tp_aspect += len(gold_aspects.intersection(pred_aspects))
        fp_aspect += len(pred_aspects - gold_aspects)
        fn_aspect += len(gold_aspects - pred_aspects)
        
        # Joint & Sentiment
        # We need to map predicted sentiments to gold sentiments for matching aspects
        gold_dict = {x[0].lower(): x[1].lower() for x in gold}
        pred_dict = {x[0].lower(): x[1].lower() for x in pred}
        
        for aspect in pred_dict:
            if aspect in gold_dict:
                # Aspect Match found
                s_gold = gold_dict[aspect]
                s_pred = pred_dict[aspect]
                
                sentiment_gold.append(s_gold)
                sentiment_pred.append(s_pred)
            
        # Joint F1 Calculation Logic (Standard SemEval)
        # Gold set of (Aspect, Sentiment)
        gold_set = set([(x[0].lower(), x[1].lower()) for x in gold])
        pred_set = set([(x[0].lower(), x[1].lower()) for x in pred])
        
        tp_joint_total += len(gold_set.intersection(pred_set))
        fp_joint_total += len(pred_set - gold_set)
        fn_joint_total += len(gold_set - pred_set)
        
    # Aspect Metrics
    precision_aspect = tp_aspect / (tp_aspect + fp_aspect) if (tp_aspect + fp_aspect) > 0 else 0
    recall_aspect = tp_aspect / (tp_aspect + fn_aspect) if (tp_aspect + fn_aspect) > 0 else 0
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect) if (precision_aspect + recall_aspect) > 0 else 0
    
    # Sentiment Metrics (Accuracy on correctly extracted aspects)
    sentiment_acc = accuracy_score(sentiment_gold, sentiment_pred) if sentiment_gold else 0
    
    # Joint Metrics
    # Recalculate using set logic which is standard
    precision_joint = tp_joint_total / (tp_joint_total + fp_joint_total) if (tp_joint_total + fp_joint_total) > 0 else 0
    recall_joint = tp_joint_total / (tp_joint_total + fn_joint_total) if (tp_joint_total + fn_joint_total) > 0 else 0
    f1_joint = 2 * precision_joint * recall_joint / (precision_joint + recall_joint) if (precision_joint + recall_joint) > 0 else 0
    
    return {
        "aspect_f1": f1_aspect,
        "aspect_precision": precision_aspect,
        "aspect_recall": recall_aspect,
        "sentiment_accuracy": sentiment_acc,
        "joint_f1": f1_joint,
        "joint_precision": precision_joint,
        "joint_recall": recall_joint
    }

def main():
    print("="*50)
    print("Generative ABSA Evaluation (vLLM)")
    print("="*50)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model path {MODEL_PATH} not found. Ensure merging is complete.")
        return

    print(f"Loading vLLM model from {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.3, # Reduced to avoid OOM/crashes
        max_model_len=2048,         # Explicitly set context length
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        stop=["\n\n", "###"]
    )
    
    print(f"Loading test dataset from {TEST_DATA_PATH}...")
    dataset_dict = load_from_disk(TEST_DATA_PATH)
    test_dataset = dataset_dict['test']
    
    print(f"Evaluating on {len(test_dataset)} samples...")
    
    prompts = []
    ground_truths = []
    
    for item in test_dataset:
        text = item['text']
        target = item['target']
        
        # Extract prompt (up to "### Response:\n")
        if "### Response:\n" in text:
            prompt = text.split("### Response:\n")[0] + "### Response:\n"
        else:
            prompt = text 
        
        prompts.append(prompt)
        ground_truths.append(target)
        
    print(f"Generating for {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params)
    
    predictions = [output.outputs[0].text for output in outputs]
        
    # Parse and Evaluate
    print("Parsing outputs and calculating metrics...")
    
    parsed_preds = [parse_output(p) for p in predictions]
    parsed_golds = [parse_output(g) for g in ground_truths]
    
    metrics = calculate_metrics(parsed_golds, parsed_preds)
    
    print("\n" + "-"*30)
    print("Evaluation Results:")
    print("-"*30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save results
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    results = {
        "metrics": metrics,
        "details": [
            {
                "ground_truth": g,
                "prediction": p,
                "parsed_gold": pg,
                "parsed_pred": pp
            }
            for g, p, pg, pp in zip(ground_truths, predictions, parsed_golds, parsed_preds)
        ]
    }
    
    with open(os.path.join(OUTPUT_DIR, "eval_results_vllm.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nDetailed results saved to {OUTPUT_DIR}/eval_results_vllm.json")

if __name__ == "__main__":
    main()
