import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

# Check if embeddings already exist
if os.path.exists("embeddings/embeddings.npy") and os.path.exists("embeddings/faiss_index.bin"):
    print("✅ Embeddings and FAISS index already exist!")
    embeddings = np.load("embeddings/embeddings.npy")
    index = faiss.read_index("embeddings/faiss_index.bin")
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Index ntotal: {index.ntotal}")
else:
    print("Building FAISS index from scratch...")
    
    # Load trained model or use pre-trained
    try:
        model = SentenceTransformer("trained_islamic_embedding_model")
        print("Using trained Islamic embedding model")
    except:
        print("Trained model not found, using pre-trained all-mpnet-base-v2")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Load original chunks
    chunks = []
    for file in ["data/hadith_chunks.jsonl", "data/quran_combined_chunks.jsonl"]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

    texts = [c['text'] for c in chunks]

    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))

    # Save to embeddings directory
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/embeddings.npy", embeddings)
    faiss.write_index(index, "embeddings/faiss_index.bin")

    print("✅ Index saved!")
    print(f"Total embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {dimension}")
