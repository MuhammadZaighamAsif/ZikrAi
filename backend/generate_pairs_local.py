import json
import random

def load_chunks(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def generate_question_template(text, chunk_index):
    """Generate a question based on text content without calling API"""
    # Extract key phrases from text
    sentences = text.split('.')
    first_sentence = sentences[0].strip() if sentences else text[:100]
    
    # Template-based questions for Islamic texts
    templates = [
        f"What does the text say about {first_sentence[:30]}?",
        f"According to the passage, {first_sentence[:40]}?",
        f"Which aspect of Islamic teaching is discussed in this passage?",
        f"What is the main topic of this Islamic text?",
        f"What guidance does this passage provide?",
    ]
    
    return random.choice(templates)

def create_pairs_local(chunk_file, output_file, max_chunks=None):
    chunks = load_chunks(chunk_file)
    if max_chunks:
        chunks = chunks[:max_chunks]
    print(f"Generating pairs from {chunk_file}...")

    out = open(output_file, "w", encoding="utf-8")

    for i, c in enumerate(chunks):
        text = c["text"]
        
        # Generate question using template
        question = generate_question_template(text, i)

        pair = {
            "query": question,
            "passage": text
        }
        out.write(json.dumps(pair, ensure_ascii=False) + "\n")

    out.close()
    print(f"✅ Saved {len(chunks)} pairs to: {output_file}")


# ---- RUN FOR BOTH ----
# Generates all chunks without displaying progress

print("Generating local pairs (no API calls)...\n")
create_pairs_local("data/hadith_chunks.jsonl", "hadith_pairs.jsonl")
create_pairs_local("data/quran_combined_chunks.jsonl", "quran_pairs.jsonl")
print("\n✅ All pairs generated successfully.")
