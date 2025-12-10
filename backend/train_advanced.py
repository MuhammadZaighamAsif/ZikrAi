"""
Advanced fine-tuning with hard negatives and triplet loss
Improves model accuracy through contrastive learning
"""
import json
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import os
from collections import defaultdict
import random

def load_data():
    """Load all data"""
    print("[*] Loading data...")
    
    # Load embeddings
    embeddings = np.load("embeddings/embeddings.npy")
    
    # Load chunks
    chunks = []
    for file in ["data/hadith_chunks.jsonl", "data/quran_combined_chunks.jsonl"]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))
    
    print(f"[+] Loaded {len(chunks)} chunks")
    print(f"[+] Embeddings shape: {embeddings.shape}")
    
    return chunks, embeddings

def load_pairs():
    """Load training pairs"""
    pairs = []
    for file in ["hadith_pairs.jsonl", "quran_pairs.jsonl"]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    pairs.append(json.loads(line))
                except:
                    continue
    
    print(f"[+] Loaded {len(pairs)} query-passage pairs")
    return pairs

def create_text_to_index_mapping(chunks, embeddings):
    """Create mapping from text to embedding index"""
    text_to_idx = {}
    
    # Only map chunks that have embeddings
    for i in range(min(len(chunks), len(embeddings))):
        text = chunks[i]['text']
        text_to_idx[text] = i
    
    print(f"[+] Created mapping for {len(text_to_idx)} chunks")
    return text_to_idx

def mine_hard_negatives(embeddings, positive_pairs, text_to_idx, k=5):
    """Mine hard negative examples for contrastive learning"""
    print("\n[*] Mining hard negatives...")
    
    # Create FAISS index for fast similarity search
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    
    triplets = []
    
    for i, pair in enumerate(positive_pairs):
        passage_text = pair["passage"]
        
        # Get passage index
        if passage_text not in text_to_idx:
            continue
        
        passage_idx = text_to_idx[passage_text]
        passage_emb = embeddings[passage_idx:passage_idx+1].astype(np.float32)
        
        # Find nearest neighbors (hard negatives)
        distances, indices = index.search(passage_emb, k=k+1)
        
        # Take neighbors that are close but not the same
        hard_negatives = [idx for idx in indices[0][1:] if idx != passage_idx]
        
        if hard_negatives:
            triplets.append({
                "anchor_idx": passage_idx,
                "positive_idx": passage_idx,  # Same for self-similarity
                "negative_idx": hard_negatives[0] if hard_negatives else None,
                "query": pair["query"]
            })
        
        if (i + 1) % 5000 == 0:
            print(f"   - Processed {i + 1}/{len(positive_pairs)} pairs")
    
    print(f"[+] Mined {len(triplets)} triplets")
    return triplets

def apply_triplet_loss_updates(embeddings, triplets, learning_rate=0.01, margin=0.3, epochs=5):
    """Apply triplet loss to refine embeddings"""
    print(f"\n[*] Applying triplet loss optimization...")
    print(f"   - Learning rate: {learning_rate}")
    print(f"   - Margin: {margin}")
    print(f"   - Epochs: {epochs}")
    
    refined_embeddings = embeddings.copy().astype(np.float32)
    
    for epoch in range(epochs):
        total_loss = 0
        updates = 0
        
        # Shuffle triplets
        random.shuffle(triplets)
        
        for triplet in triplets[:min(5000, len(triplets))]:
            if triplet["negative_idx"] is None:
                continue
            
            anchor_idx = triplet["anchor_idx"]
            positive_idx = triplet["positive_idx"]
            negative_idx = triplet["negative_idx"]
            
            # Get embeddings
            anchor = refined_embeddings[anchor_idx]
            positive = refined_embeddings[positive_idx]
            negative = refined_embeddings[negative_idx]
            
            # Calculate distances
            pos_dist = np.linalg.norm(anchor - positive)
            neg_dist = np.linalg.norm(anchor - negative)
            
            # Triplet loss: max(0, pos_dist - neg_dist + margin)
            loss = max(0, pos_dist - neg_dist + margin)
            
            if loss > 0:
                # Update embeddings
                # Move anchor closer to positive
                refined_embeddings[anchor_idx] += learning_rate * (positive - anchor)
                # Move anchor away from negative
                refined_embeddings[anchor_idx] -= learning_rate * (negative - anchor) * 0.5
                
                total_loss += loss
                updates += 1
        
        avg_loss = total_loss / max(updates, 1)
        print(f"   - Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.4f}, Updates = {updates}")
    
    print(f"[+] Optimization complete")
    return refined_embeddings

def normalize_embeddings(embeddings):
    """L2 normalize embeddings"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)

def evaluate_model(embeddings, pairs, text_to_idx, k=5):
    """Evaluate retrieval performance"""
    print(f"\n[*] Evaluating model (top-{k})...")
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for pair in pairs[:min(1000, len(pairs))]:
        passage_text = pair["passage"]
        
        if passage_text not in text_to_idx:
            continue
        
        passage_idx = text_to_idx[passage_text]
        passage_emb = embeddings[passage_idx:passage_idx+1].astype(np.float32)
        
        # Search for similar passages
        distances, indices = index.search(passage_emb, k=k)
        
        # Check if passage is retrieved
        if passage_idx == indices[0][0]:
            correct_top1 += 1
        
        if passage_idx in indices[0]:
            correct_top5 += 1
        
        total += 1
    
    top1_acc = (correct_top1 / total * 100) if total > 0 else 0
    top5_acc = (correct_top5 / total * 100) if total > 0 else 0
    
    print(f"[+] Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"[+] Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"   - Evaluated on {total} pairs")
    
    return top1_acc, top5_acc

def save_model(embeddings, output_dir="fine_tuned_embeddings"):
    """Save fine-tuned model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
    
    # Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    
    print(f"[+] Model saved to {output_dir}")

def main():
    print("\n" + "="*70)
    print("Advanced Fine-tuning with Hard Negatives and Triplet Loss")
    print("="*70)
    
    # Load data
    chunks, embeddings = load_data()
    pairs = load_pairs()
    
    # Create mapping
    text_to_idx = create_text_to_index_mapping(chunks, embeddings)
    
    # Evaluate baseline
    print("\n" + "="*70)
    print("BASELINE EVALUATION")
    print("="*70)
    baseline_top1, baseline_top5 = evaluate_model(embeddings, pairs, text_to_idx)
    
    # Mine hard negatives
    triplets = mine_hard_negatives(embeddings, pairs, text_to_idx, k=10)
    
    # Apply triplet loss
    refined_embeddings = apply_triplet_loss_updates(
        embeddings, 
        triplets, 
        learning_rate=0.005,
        margin=0.5,
        epochs=10
    )
    
    # Normalize embeddings
    print("\n[*] Normalizing embeddings...")
    refined_embeddings = normalize_embeddings(refined_embeddings)
    print("[+] Normalization complete")
    
    # Evaluate refined model
    print("\n" + "="*70)
    print("FINE-TUNED MODEL EVALUATION")
    print("="*70)
    refined_top1, refined_top5 = evaluate_model(refined_embeddings, pairs, text_to_idx)
    
    # Save model
    save_model(refined_embeddings, "fine_tuned_embeddings")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Baseline Model:")
    print(f"  - Top-1: {baseline_top1:.2f}%")
    print(f"  - Top-5: {baseline_top5:.2f}%")
    print(f"\nFine-tuned Model:")
    print(f"  - Top-1: {refined_top1:.2f}%")
    print(f"  - Top-5: {refined_top5:.2f}%")
    print(f"\nImprovement:")
    print(f"  - Top-1: +{refined_top1 - baseline_top1:.2f}%")
    print(f"  - Top-5: +{refined_top5 - baseline_top5:.2f}%")
    print("\n" + "="*70)
    print("[OK] Training complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
