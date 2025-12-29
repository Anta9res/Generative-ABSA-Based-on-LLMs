import json
import random
from collections import Counter

FILE_PATH = "/root/autodl-tmp/NLP论文复现/data/absa_dataset/crypto_absa_silver.json"

def verify_data():
    print(f"Verifying {FILE_PATH}...")
    try:
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load JSON. {e}")
        return

    print(f"Total samples: {len(data)}")
    
    aspect_counts = Counter()
    sentiment_counts = Counter()
    
    valid_samples = 0
    for item in data:
        if 'absa_annotation' in item and isinstance(item['absa_annotation'], list):
            valid_samples += 1
            for aspect in item['absa_annotation']:
                aspect_counts[aspect['aspect']] += 1
                sentiment_counts[aspect['sentiment']] += 1
    
    print(f"Valid structured samples: {valid_samples}")
    print("\nAspect Distribution:")
    for k, v in aspect_counts.most_common():
        print(f"  {k}: {v}")
        
    print("\nSentiment Distribution:")
    for k, v in sentiment_counts.most_common():
        print(f"  {k}: {v}")
        
    print("\nRandom Samples:")
    samples = random.sample(data, min(3, len(data)))
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        print(f"Text: {sample['original_text'][:100]}...")
        print("Annotations:")
        for ann in sample['absa_annotation']:
            print(f"  - Aspect: {ann['aspect']}, Sentiment: {ann['sentiment']}, Evidence: {ann.get('evidence', '')[:50]}...")

if __name__ == "__main__":
    verify_data()
