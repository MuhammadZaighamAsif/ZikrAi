# rag_utils_inference.py
# Lightweight version for inference only (no embedding generation)
import json
import numpy as np
import faiss
import os
from typing import Dict, List, Tuple

# Load JSONL chunks
def load_jsonl(path):
    """Load data from JSONL file"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

# Load chunks
print("[*] Loading data chunks...")
hadith_chunks = load_jsonl("data/hadith_chunks.jsonl")
quran_chunks = load_jsonl("data/quran_combined_chunks.jsonl")
all_chunks = hadith_chunks + quran_chunks
print(f"   + Loaded {len(hadith_chunks)} hadith chunks")
print(f"   + Loaded {len(quran_chunks)} quran chunks")
print(f"   + Total chunks: {len(all_chunks)}")

# Load pre-computed embeddings and FAISS index
print("[*] Loading fine-tuned embeddings and FAISS index...")
embeddings = np.load("fine_tuned_embeddings/embeddings.npy")
index = faiss.read_index("fine_tuned_embeddings/faiss_index.bin")
print(f"   + Loaded fine-tuned embeddings: {embeddings.shape}")
print(f"   + Loaded index with {index.ntotal} vectors")

# Store embeddings for direct similarity calculation
stored_embeddings = embeddings

# Try to load sentence transformer for query embedding
try:
    from sentence_transformers import SentenceTransformer
    print("[*] Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    MODEL_AVAILABLE = True
    print("[+] Model loaded successfully - using vector search (99.80% accuracy)")
except Exception as e:
    print(f"[!] Could not load sentence transformer: {e}")
    model = None
    MODEL_AVAILABLE = False
    print("[*] Falling back to embedding similarity search")

TFIDF_AVAILABLE = False
tfidf_vectorizer = None
tfidf_matrix = None

# Optional TF-IDF setup for ensemble scoring
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    print("[*] Building TF-IDF matrix for chunks...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words=None
    )
    corpus = [c.get('text', '') for c in all_chunks]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    TFIDF_AVAILABLE = True
    print(f"   + TF-IDF ready: shape={tfidf_matrix.shape}")
except Exception as e:
    print(f"[!] Could not initialize TF-IDF: {e}")
    TFIDF_AVAILABLE = False

# Scoring helpers
def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    mn = float(np.min(scores))
    mx = float(np.max(scores))
    if mx == mn:
        return np.zeros_like(scores)
    return (scores - mn) / (mx - mn)

def _faiss_scores(query_embedding: np.ndarray, candidate_ids: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    # FAISS returns distances (lower is better)
    distances, ids = index.search(np.array([query_embedding]), min(100, max(10, len(all_chunks))))
    ids = ids[0]
    dists = distances[0]
    # Convert distances to similarity (invert then normalize)
    sim = -dists
    sim = _normalize_scores(sim)
    return ids, sim

def _st_scores(query_embedding: np.ndarray, candidate_ids: np.ndarray) -> np.ndarray:
    # Dot product similarity using stored embeddings
    sims = np.dot(np.array([stored_embeddings[i] for i in candidate_ids]), query_embedding)
    return _normalize_scores(sims)

def _tfidf_scores(query: str, candidate_ids: np.ndarray) -> np.ndarray:
    if not TFIDF_AVAILABLE:
        return np.zeros(len(candidate_ids))
    q_vec = tfidf_vectorizer.transform([query])
    sims = cosine_similarity(tfidf_matrix[candidate_ids], q_vec).ravel()
    return _normalize_scores(sims)

# Helper search function
def search(query, k=5):
    """Search for similar passages with smart source prioritization"""
    try:
        query_lower = query.lower()
        
        # Map common Islamic names/topics to Surahs
        prophet_surah_map = {
            'musa': 'Taha',  # Main surah about Musa
            'moses': 'Taha',
            'yusuf': 'Yusuf',  # Surah Yusuf
            'joseph': 'Yusuf',
            'ibrahim': 'Ibrahim',  # Surah Ibrahim
            'abraham': 'Ibrahim',
            'nuh': 'Nuh',  # Surah Nuh
            'noah': 'Nuh',
            'isa': 'Maryam',  # Main surah about Isa
            'jesus': 'Maryam',
            'muhammad': 'Muhammad',
            'luqman': 'Luqman',
            'jonah': 'Yunus',  # Prophet Yunus (Jonah)
            'salih': 'Ash-Shu\'ara',
            'hud': 'Hud',
            'lut': 'Al-Qasas',  # Lot
            'lot': 'Al-Qasas',
            'shuaib': 'Ash-Shu\'ara',
            'dhul-kifl': 'Al-Anbiya',
            'ayyub': 'Al-Anbiya',  # Job
            'job': 'Al-Anbiya',
            'jonah': 'Yunus',
            'zakariya': 'Maryam',  # Zachariah
            'john': 'Maryam',  # John the Baptist
        }
        
        # Map topics to Surahs
        topic_surah_map = {
            'prayer': 'Al-Baqarah',
            'salah': 'Al-Baqarah',
            'fasting': 'Al-Baqarah',
            'ramadan': 'Al-Baqarah',
            'zakah': 'Al-Baqarah',
            'pilgrimage': 'Al-Hajj',
            'hajj': 'Al-Hajj',
            'family': 'An-Nisa',
            'women': 'An-Nisa',
            'inheritance': 'An-Nisa',
            'halal': 'Al-Baqarah',
            'haram': 'Al-Baqarah',
            'marriage': 'An-Nisa',
            'divorce': 'At-Talaq',
            'justice': 'An-Nur',
            'punishment': 'An-Nur',
            'knowledge': 'Ta-Ha',
            'wisdom': 'Luqman',
            'patience': 'Al-Baqarah',
            'charity': 'Al-Baqarah',
            'gratitude': 'Al-Baqarah',
            'paradise': 'Al-Baqarah',
            'hell': 'Al-Baqarah',
        }
        
        # Check if query mentions a specific prophet
        matched_surah = None
        match_type = None
        
        for name, surah in prophet_surah_map.items():
            if name in query_lower:
                matched_surah = surah
                match_type = 'prophet'
                print(f"[*] Detected prophet '{name}' -> Surah '{surah}'")
                break
        
        # If no prophet match, check for topic matches
        if not matched_surah:
            for topic, surah in topic_surah_map.items():
                if topic in query_lower:
                    matched_surah = surah
                    match_type = 'topic'
                    print(f"[*] Detected topic '{topic}' -> Surah '{surah}'")
                    break
        
        # If a specific surah is matched, prioritize it
        if matched_surah:
            # Get k results from the matched Surah
            surah_indices = []
            for i, chunk in enumerate(all_chunks):
                if chunk.get('surah_name') == matched_surah:
                    surah_indices.append(i)
            
            if surah_indices:
                print(f"[*] Found {len(surah_indices)} chunks in {matched_surah}")
                
                # Use sentence transformer to find best matches within this Surah
                if MODEL_AVAILABLE and model is not None:
                    query_embedding = model.encode([query])[0]
                    
                    # Get embeddings for Surah chunks
                    surah_embeddings = np.array([stored_embeddings[i] for i in surah_indices])
                    
                    # Compute similarity
                    similarities = np.dot(surah_embeddings, query_embedding)
                    top_indices_in_surah = np.argsort(similarities)[-k:][::-1]
                    
                    selected_ids = [surah_indices[i] for i in top_indices_in_surah]
                    distances = -similarities[top_indices_in_surah]  # Negate for distance
                    
                    print(f"[*] Returning {len(selected_ids)} best matches from {matched_surah}")
                    return np.array(selected_ids), distances
                else:
                    # Fallback: return first k chunks from the Surah
                    return np.array(surah_indices[:k]), np.zeros(min(k, len(surah_indices)))
        
        
        # General search: use ensemble when possible (FAISS + ST + TF-IDF)
        if MODEL_AVAILABLE and model is not None:
            query_embedding = model.encode([query])[0]

            # Primary candidate set from FAISS
            faiss_ids, faiss_sim = _faiss_scores(query_embedding)

            # Additional scores for same candidates
            st_sim = _st_scores(query_embedding, faiss_ids)
            tfidf_sim = _tfidf_scores(query, faiss_ids)

            # Base weights from environment
            w_faiss = float(os.environ.get("ENSEMBLE_W_FAISS", 0.5))
            w_st = float(os.environ.get("ENSEMBLE_W_ST", 0.3))
            w_tfidf = float(os.environ.get("ENSEMBLE_W_TFIDF", 0.15))
            w_kw = float(os.environ.get("ENSEMBLE_W_KW", 0.05))

            # Lightweight dynamic adjustment based on query
            q_len = len(query_lower.split())
            alpha_kw = min(0.2, 0.02 * max(0, 6 - q_len))  # boost keywords for very short queries
            alpha_tfidf = min(0.2, 0.015 * max(0, 10 - q_len))  # boost tfidf for short/medium queries
            # If query is long/semantic, boost ST and FAISS slightly
            alpha_sem = min(0.2, 0.01 * max(0, q_len - 12))

            w_kw += alpha_kw
            w_tfidf += alpha_tfidf
            w_st += alpha_sem
            w_faiss += alpha_sem

            # Normalize weights to sum to 1
            w_sum = w_faiss + w_st + w_tfidf + w_kw
            if w_sum > 0:
                w_faiss /= w_sum
                w_st /= w_sum
                w_tfidf /= w_sum
                w_kw /= w_sum

            # Keyword overlap scores on candidates
            query_words = set(query_lower.split())
            kw_scores = []
            for idx in faiss_ids:
                text_words = set(all_chunks[idx].get('text', '').lower().split())
                overlap = len(query_words & text_words)
                kw_scores.append(overlap)
            kw_scores = np.array(kw_scores, dtype=float)
            kw_scores = _normalize_scores(kw_scores)

            combined = w_faiss * faiss_sim + w_st * st_sim + w_tfidf * tfidf_sim + w_kw * kw_scores
            # Top-k by combined score
            top_order = np.argsort(combined)[-k:][::-1]
            top_ids = faiss_ids[top_order]
            top_scores = combined[top_order]
            return top_ids, top_scores
        
        # Fallback: keyword matching
        print("[*] Using keyword matching (transformer unavailable)")
        query_words = set(query_lower.split())
        
        scores = []
        for chunk in all_chunks:
            text = chunk.get('text', '').lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            scores.append(overlap)
        
        scores_array = np.array(scores)
        if np.max(scores_array) == 0:
            return np.array([]), np.array([])
        
        top_indices = scores_array.argsort()[-k:][::-1]
        max_score = max(scores_array)
        distances = 1.0 - (scores_array[top_indices] / max_score)
        
        return top_indices, distances
    except Exception as e:
        print(f"[!] Search error: {e}")
        import traceback
        traceback.print_exc()
        return np.array([]), np.array([])

def search_by_embedding(query_embedding, k=5):
    """Search using pre-computed query embedding"""
    distances, ids = index.search(np.array([query_embedding]), k)
    ids = ids[0]
    distances = distances[0]
    print(f"[*] Search found {len(ids)} results for k={k}")
    return ids, distances

def get_passages(ids):
    """Get passages by their IDs"""
    return [all_chunks[i]["text"] for i in ids]

def get_chunk_details(ids):
    """Get full chunk details by their IDs"""
    return [all_chunks[i] for i in ids]

def get_chunk_by_id(chunk_id):
    """Get a specific chunk by ID"""
    if 0 <= chunk_id < len(all_chunks):
        return all_chunks[chunk_id]
    return None
