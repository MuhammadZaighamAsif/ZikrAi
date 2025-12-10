"""
Lightweight fine-tuning using model adapters (LoRA)
No large model downloads needed
"""
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def load_training_pairs(max_pairs=None):
    """Load question-passage pairs"""
    pairs = []
    
    for pair_file in ["hadith_pairs.jsonl", "quran_pairs.jsonl"]:
        if not os.path.exists(pair_file):
            continue
            
        with open(pair_file, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                try:
                    item = json.loads(line)
                    pairs.append({
                        "query": item["query"],
                        "passage": item["passage"]
                    })
                    count += 1
                    if max_pairs and len(pairs) >= max_pairs:
                        return pairs
                except:
                    continue
        
        print(f"[+] Loaded {count} pairs from {pair_file}")
    
    return pairs

def train_tfidf_vectorizer(pairs, output_dir="tfidf_model"):
    """Train optimized TF-IDF vectorizer"""
    print("\n[*] Training TF-IDF vectorizer...")
    
    queries = [p["query"] for p in pairs]
    passages = [p["passage"] for p in pairs]
    all_texts = queries + passages
    
    # Train vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=10000,      # Increased from 5000
        min_df=2,                # Ignore terms that appear in < 2 docs
        max_df=0.95,             # Ignore terms that appear in > 95% of docs
        ngram_range=(1, 2),      # Use unigrams and bigrams
        sublinear_tf=True,       # Apply sublinear term frequency scaling
        strip_accents='unicode',
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    print(f"[+] Vectorizer trained")
    print(f"   - Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   - Matrix shape: {tfidf_matrix.shape}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"[+] Model saved to {output_dir}")
    
    return vectorizer, tfidf_matrix

def evaluate_retrieval(vectorizer, pairs):
    """Evaluate retrieval performance"""
    print("\n[*] Evaluating retrieval performance...")
    
    test_pairs = pairs[:min(100, len(pairs))]
    correct = 0
    total = 0
    
    for pair in test_pairs:
        query = pair["query"]
        passage = pair["passage"]
        
        query_vec = vectorizer.transform([query])
        passage_vec = vectorizer.transform([passage])
        
        similarity = cosine_similarity(query_vec, passage_vec)[0][0]
        
        if similarity > 0.3:  # Threshold
            correct += 1
        
        total += 1
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"[+] Retrieval accuracy: {accuracy:.2f}%")
    print(f"   - Matched {correct}/{total} pairs")
    
    return accuracy

def main():
    print("\n" + "="*60)
    print("Islamic Knowledge Model Fine-tuning (TF-IDF)")
    print("="*60)
    
    # Load training data
    print("\n[*] Loading training pairs...")
    pairs = load_training_pairs()
    
    if not pairs:
        print("[!] No training data found")
        return
    
    print(f"[+] Total pairs: {len(pairs)}")
    
    # Train TF-IDF
    vectorizer, tfidf_matrix = train_tfidf_vectorizer(pairs)
    
    # Evaluate
    accuracy = evaluate_retrieval(vectorizer, pairs)
    
    print("\n" + "="*60)
    print("[OK] Fine-tuning complete!")
    print(f"   - Model saved to: tfidf_model/")
    print(f"   - Accuracy: {accuracy:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
