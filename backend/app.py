# backend/app.py
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from dotenv import load_dotenv
import os
import sys

# Load environment variables from .env file
load_dotenv()

# Add backend directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import RAG utilities
from rag_utils_inference import search, get_passages, get_chunk_details, all_chunks, MODEL_AVAILABLE, TFIDF_AVAILABLE

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
    OPENAI_AVAILABLE = True
    print("[+] OpenAI API key loaded successfully")
else:
    print("[!] OPENAI_API_KEY not set - LLM responses will not be available")
    client = None
    OPENAI_AVAILABLE = False

app = Flask(__name__, static_folder='../frontend')

# Enable CORS manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

print("\n[OK] ZikrAi Server Ready!")
print(f"   [*] Total chunks loaded: {len(all_chunks)}")
print(f"   [*] Search method: {'Sentence Transformer' if MODEL_AVAILABLE else 'TF-IDF' if TFIDF_AVAILABLE else 'Not available'}")
print(f"   [*] Server starting on http://localhost:5000\n")

def rag_answer(query, k=5, strategy: str = 'ensemble'):
    """Generate answer using RAG"""
    if not MODEL_AVAILABLE and not TFIDF_AVAILABLE:
        error_msg = """I apologize, but no search method is available. 

To fix this, install scikit-learn:
pip install scikit-learn

Or install sentence-transformers (requires working PyTorch):
pip install sentence-transformers"""
        return error_msg, []
    
    try:
        # Configure ensemble weights based on strategy
        strategy = (strategy or 'ensemble').lower()
        if strategy == 'ensemble':
            os.environ['ENSEMBLE_W_FAISS'] = os.environ.get('ENSEMBLE_W_FAISS', '0.5')
            os.environ['ENSEMBLE_W_ST'] = os.environ.get('ENSEMBLE_W_ST', '0.3')
            os.environ['ENSEMBLE_W_TFIDF'] = os.environ.get('ENSEMBLE_W_TFIDF', '0.15')
            os.environ['ENSEMBLE_W_KW'] = os.environ.get('ENSEMBLE_W_KW', '0.05')
        elif strategy == 'faiss':
            os.environ['ENSEMBLE_W_FAISS'] = '1.0'
            os.environ['ENSEMBLE_W_ST'] = '0.0'
            os.environ['ENSEMBLE_W_TFIDF'] = '0.0'
            os.environ['ENSEMBLE_W_KW'] = '0.0'
        elif strategy == 'st':
            os.environ['ENSEMBLE_W_FAISS'] = '0.0'
            os.environ['ENSEMBLE_W_ST'] = '1.0'
            os.environ['ENSEMBLE_W_TFIDF'] = '0.0'
            os.environ['ENSEMBLE_W_KW'] = '0.0'
        elif strategy == 'tfidf':
            os.environ['ENSEMBLE_W_FAISS'] = '0.0'
            os.environ['ENSEMBLE_W_ST'] = '0.0'
            os.environ['ENSEMBLE_W_TFIDF'] = '1.0'
            os.environ['ENSEMBLE_W_KW'] = '0.0'
        elif strategy == 'keywords':
            os.environ['ENSEMBLE_W_FAISS'] = '0.0'
            os.environ['ENSEMBLE_W_ST'] = '0.0'
            os.environ['ENSEMBLE_W_TFIDF'] = '0.0'
            os.environ['ENSEMBLE_W_KW'] = '1.0'
        else:
            strategy = 'ensemble'
            os.environ['ENSEMBLE_W_FAISS'] = os.environ.get('ENSEMBLE_W_FAISS', '0.5')
            os.environ['ENSEMBLE_W_ST'] = os.environ.get('ENSEMBLE_W_ST', '0.3')
            os.environ['ENSEMBLE_W_TFIDF'] = os.environ.get('ENSEMBLE_W_TFIDF', '0.15')
            os.environ['ENSEMBLE_W_KW'] = os.environ.get('ENSEMBLE_W_KW', '0.05')

        # Search for relevant passages
        ids, distances = search(query, k=k)
        
        # Check if search returned valid results
        if ids is None or len(ids) == 0:
            return "I don't have enough information in the provided sources to answer this question.", []
        
        passages = get_passages(ids)
        details = get_chunk_details(ids)
        
        # Check if we got passages
        if not passages or not details:
            return "I don't have enough information in the provided sources to answer this question.", []
        
        # Build context with sources
        context_parts = []
        sources_info = []
        
        for i, (passage, detail) in enumerate(zip(passages, details), 1):
            if passage is None or detail is None:
                continue
                
            if 'surah_name' in detail:
                source = f"Quran - Surah {detail['surah_name']}, {detail['verse_number']}"
            elif 'source' in detail:
                source = detail.get('source', 'Unknown')
            else:
                source = "Islamic Text"
            
            context_parts.append(f"[{i}] {passage}\n(Source: {source})")
            
            # Safety check for distances
            try:
                relevance = float(1.0 / (1.0 + distances[i-1])) if i-1 < len(distances) else 0.5
            except:
                relevance = 0.5
            
            sources_info.append({
                "text": passage,
                "source": source,
                "relevance": relevance
            })
        
        if not context_parts:
            return "I don't have enough information in the provided sources to answer this question.", []
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are ZikrAi — an Islamic Knowledge Assistant.

Answer the user's question *ONLY* using the provided Islamic texts below.

Guidelines:
- Provide a clear, accurate answer based on the sources
- Quote relevant passages when appropriate
- If the answer is not in the sources, say: "I don't have enough information in the provided sources to answer this question."
- Be respectful and maintain Islamic etiquette

Question: {query}

Relevant Islamic Passages:
{context}

Please provide a comprehensive answer:"""
        
        # Get response from OpenAI
        if not OPENAI_AVAILABLE:
            answer = f"""Based on the search results, here are the most relevant passages:

{context}

Note: Full AI-generated answer unavailable. To enable, set OPENAI_API_KEY environment variable."""
            return answer, sources_info
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content
        return answer, sources_info
        
    except Exception as e:
        print(f"[!] RAG Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", []

# Serve frontend
@app.route("/")
def home():
    """Serve the main frontend page"""
    return send_from_directory('../frontend', 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    """Serve static files"""
    return send_from_directory('../frontend', path)

# API endpoints
@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "total_chunks": len(all_chunks),
        "model_available": MODEL_AVAILABLE or TFIDF_AVAILABLE,
        "search_method": "transformer" if MODEL_AVAILABLE else "tfidf" if TFIDF_AVAILABLE else "none"
    })

@app.route("/api/stats", methods=["GET"])
def stats():
    """Get system statistics"""
    hadith_count = sum(1 for c in all_chunks if c.get('source', '').startswith('Hadith'))
    quran_count = sum(1 for c in all_chunks if 'surah_name' in c)
    
    return jsonify({
        "total_chunks": len(all_chunks),
        "hadith_chunks": hadith_count,
        "quran_chunks": quran_count,
        "model_available": MODEL_AVAILABLE or TFIDF_AVAILABLE,
        "search_method": "transformer" if MODEL_AVAILABLE else "tfidf" if TFIDF_AVAILABLE else "none"
    })

@app.route("/api/ask", methods=["POST"])
def ask():
    """Ask a question endpoint"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    k = data.get("k", 5)
    strategy = data.get("strategy") or request.args.get("strategy") or 'ensemble'
    
    try:
        answer, sources = rag_answer(query, k=k, strategy=strategy)
        return jsonify({
            "success": True,
            "query": query,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "strategy": strategy
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    print("Starting Flask server...")
    print("Frontend: http://localhost:5000")
    print("Press CTRL+C to stop\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
