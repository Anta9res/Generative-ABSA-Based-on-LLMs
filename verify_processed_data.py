from datasets import load_from_disk
import os

DATASET_PATH = "/root/autodl-tmp/NLP论文复现/data/processed/absa_dataset"

def verify_processed():
    if not os.path.exists(DATASET_PATH):
        print("Dataset path not found.")
        return

    dataset = load_from_disk(DATASET_PATH)
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    # Check for "no direct mention" in the target (Response) field
    hallucination_count = 0
    total_samples = 0
    
    for split in ['train', 'test']:
        for item in dataset[split]:
            total_samples += 1
            target = item['target'].lower()
            if "no direct mention" in target or "not mentioned" in target:
                hallucination_count += 1
                print(f"Found hallucination in {split}: {target}")

    print(f"\nVerification Results:")
    print(f"Scanned {total_samples} samples.")
    print(f"Found {hallucination_count} samples with 'no direct mention' in target.")
    
    if hallucination_count == 0:
        print("✅ Data Quality Verification PASSED: No noise detected in targets.")
    else:
        print("❌ Data Quality Verification FAILED: Noise detected.")

    # Let's print a few samples to show the user the format is clean.
    print("\nSample Data (Train):")
    print(dataset['train'][0]['text'])
    print("-" * 20)
    print("Target:", dataset['train'][0]['target'])

if __name__ == "__main__":
    verify_processed()
