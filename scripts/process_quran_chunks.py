"""
Script to process Quran data into chunks suitable for RAG
Converts quran_combined.jsonl into quran_combined_chunks.jsonl
"""

import json
from pathlib import Path


def load_jsonl(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    """Save data to a JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


def chunk_quran_verses(quran_data):
    """Process Quran data and create chunks suitable for RAG"""
    chunks = []
    
    for surah in quran_data:
        surah_index = surah['index']
        surah_name = surah['name']
        
        for verse_key, verse_content in surah['verses'].items():
            chunk = {
                'surah_index': surah_index,
                'surah_name': surah_name,
                'verse_number': verse_key,
                'arabic': verse_content['arabic'],
                'english': verse_content['english'],
                'text': f"{verse_content['english']} ({verse_content['arabic']})",
                'metadata': {
                    'source': 'quran',
                    'surah': surah_name,
                    'verse': verse_key
                }
            }
            chunks.append(chunk)
    
    return chunks


def main():
    """Process Quran data into chunks"""
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "quran_combined.jsonl"
    output_file = base_dir / "data" / "quran_combined_chunks.jsonl"
    
    print("Loading Quran data...")
    quran_data = load_jsonl(str(input_file))
    print(f"Loaded {len(quran_data)} surahs")
    
    print("\nChunking verses...")
    chunks = chunk_quran_verses(quran_data)
    print(f"Created {len(chunks)} verse chunks")
    
    print(f"\nSaving chunks to {output_file}...")
    save_jsonl(chunks, str(output_file))
    
    print("\n✓ Successfully processed Quran chunks!")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Total chunks: {len(chunks)}")
    
    # Show sample chunk
    if chunks:
        print("\nSample chunk:")
        import json
        print(json.dumps(chunks[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
