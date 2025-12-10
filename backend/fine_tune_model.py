"""
Fine-tune embedding model on Islamic knowledge question-passage pairs
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import os
from datetime import datetime

# Configuration
BATCH_SIZE = 16
EPOCHS = 3
WARMUP_STEPS = 500
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OUTPUT_DIR = "fine_tuned_islamic_model"
PAIRS_FILES = ["hadith_pairs.jsonl", "quran_pairs.jsonl"]

def load_training_pairs(max_pairs=None):
    """Load question-passage pairs for training"""
    train_examples = []
    
    for pair_file in PAIRS_FILES:
        if not os.path.exists(pair_file):
            print(f"[!] File not found: {pair_file}")
            continue
            
        with open(pair_file, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                try:
                    item = json.loads(line)
                    train_examples.append(
                        InputExample(
                            texts=[item["query"], item["passage"]],
                            label=1.0  # Positive pair
                        )
                    )
                    count += 1
                    if max_pairs and count >= max_pairs:
                        break
                except json.JSONDecodeError:
                    continue
        
        print(f"[+] Loaded {count} pairs from {pair_file}")
    
    return train_examples

def fine_tune_model(train_examples, output_dir=OUTPUT_DIR):
    """Fine-tune the model on training pairs"""
    
    print(f"\n[*] Fine-tuning configuration:")
    print(f"   - Base model: {MODEL_NAME}")
    print(f"   - Training pairs: {len(train_examples)}")
    print(f"   - Batch size: {BATCH_SIZE}")
    print(f"   - Epochs: {EPOCHS}")
    print(f"   - Warmup steps: {WARMUP_STEPS}")
    print(f"   - Output directory: {output_dir}\n")
    
    # Load base model
    print("[*] Loading base model...")
    model = SentenceTransformer(MODEL_NAME)
    print("[+] Model loaded successfully")
    
    # Create data loader
    train_dataloader = DataLoader(
        train_examples, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    # Define loss function
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Fine-tune
    print("\n[*] Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        show_progress_bar=True,
        save_best_model=True
    )
    
    # Save model
    print(f"\n[*] Saving fine-tuned model to {output_dir}...")
    model.save(output_dir)
    print(f"[+] Model saved successfully!")
    
    return model, output_dir

def evaluate_model(model, test_examples, k=5):
    """Evaluate model on test set"""
    print(f"\n[*] Evaluating model on {len(test_examples)} examples...")
    
    correct = 0
    for example in test_examples[:min(100, len(test_examples))]:
        query = example.texts[0]
        passage = example.texts[1]
        
        # Encode query and passages
        query_embedding = model.encode(query)
        passage_embedding = model.encode(passage)
        
        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([query_embedding], [passage_embedding])[0][0]
        
        if similarity > 0.5:  # Threshold for correct match
            correct += 1
    
    accuracy = (correct / min(100, len(test_examples))) * 100
    print(f"[+] Evaluation accuracy: {accuracy:.2f}%")
    
    return accuracy

def main():
    """Main fine-tuning pipeline"""
    
    print("\n" + "="*60)
    print("Islamic Knowledge Embedding Model Fine-tuning")
    print("="*60)
    
    # Load training data
    print("\n[*] Loading training pairs...")
    train_examples = load_training_pairs()
    
    if not train_examples:
        print("[!] No training examples loaded. Exiting.")
        return
    
    print(f"[+] Total training examples: {len(train_examples)}")
    
    # Split into train/test (80/20)
    split_idx = int(len(train_examples) * 0.8)
    train_set = train_examples[:split_idx]
    test_set = train_examples[split_idx:]
    
    print(f"[+] Train set: {len(train_set)} examples")
    print(f"[+] Test set: {len(test_set)} examples")
    
    # Fine-tune model
    model, output_dir = fine_tune_model(train_set)
    
    # Evaluate
    try:
        evaluate_model(model, test_set)
    except Exception as e:
        print(f"[!] Evaluation failed: {e}")
    
    print("\n" + "="*60)
    print(f"[OK] Fine-tuning complete!")
    print(f"   - Model saved to: {os.path.abspath(output_dir)}")
    print(f"   - Use this model with: SentenceTransformer('{output_dir}')")
    print("="*60 + "\n")
    
    return model, output_dir

if __name__ == "__main__":
    model, output_dir = main()
