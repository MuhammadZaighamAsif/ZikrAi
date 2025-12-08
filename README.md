# ZikrAi

Quran/Hadith question answering with retrieval-augmented generation (RAG).

## Project Overview
- Backend: Flask API using FAISS + sentence-transformers, with optional TF‑IDF and keyword fusion.
- Frontend: HTML/CSS/JS chat UI with dark mode, font-size controls, copy-to-clipboard, and source expansion.

## Features
- Ensemble retrieval combining FAISS, sentence-transformers, TF‑IDF, and keyword overlap with dynamic per-query weighting.
- Transparent answers showing matched sources and chunks used.
- Clean UI with dark theme toggle, readability controls, toasts, and copy button.

## Repo Structure
- `backend/`: Flask app and retrieval logic (`app.py`, `rag_utils_inference.py`, `requirements.txt`).
- `frontend/`: UI (`index.html`, `static/css/style.css`, `static/js/app.js`).
- `scripts/`: Utility scripts (e.g., `process_quran_chunks.py`).
- `report.txt`: Project summary for submission.
- `.gitignore`: Excludes venv, caches, large generated artifacts, and secrets.

## Requirements
- Python 3.10+ recommended.
- Windows PowerShell 5.1 (commands below use PowerShell syntax).
- Packages listed in `backend/requirements.txt` (Flask, faiss-cpu, torch, sentence-transformers, scikit-learn, numpy, transformers).

## Setup
```powershell
# From repo root
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -r backend\requirements.txt
```

Optional environment variables:
- `OPENAI_API_KEY` (if enabling external LLMs in backend; not required for local retrieval).
- Create `backend\.env` if needed; do not commit it.

## Data & Models
- The repo does not include large data/embeddings.
- Expected local paths (ignored by `.gitignore` if generated):
	- `backend\data\` for chunked Quran/Hadith JSONL.
	- `backend\embeddings\` and `backend\fine_tuned_embeddings\` for `.npy` vectors.
	- FAISS index files (e.g., `.bin`) stored under `backend\`.
- You can regenerate embeddings/indices using scripts under `backend/` if required.

## Run
```powershell
# Start the backend
Set-Location backend
python app.py
```

Frontend:
- Open `frontend/index.html` directly or serve via a simple static server.
- Ensure `app.js` points to `http://127.0.0.1:5000` for the backend.

## API
- `GET /api/health`: Health check.
- `GET /api/stats`: Basic stats (counts, model info).
- `POST /api/ask`: Body `{ "query": "...", "strategy": "auto" }`
	- `strategy`: `faiss` | `st` | `tfidf` | `keywords` | `auto` (default).
	- Returns `{ answer, sources: [{text, meta, score}], debug }`.

## Notes
- Run backend from the `backend` directory to avoid relative path issues.
- Large artifacts (`.npy`, `.bin`, dataset folders) and secrets are ignored by `.gitignore`.
- For a quick demo, add small sample chunks under `backend\data\` locally.

## Acknowledgements
- Hadith: data sourced via public Hadith API projects such as `https://github.com/swmohammad/hadith-api` and similar community-maintained resources.
- Quran: text/metadata from public Quran JSON repositories such as `https://github.com/semarketir/quranjson`.

