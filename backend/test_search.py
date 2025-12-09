"""Test search functionality and see what sources are returned"""
from rag_utils_inference import search, get_chunk_details, all_chunks
import json

# Test query
query = "What does Quran say about patience"
print(f"Query: {query}\n")

# Search
ids, distances = search(query, k=10)
details = get_chunk_details(ids)

# Display results
print("="*70)
print("SEARCH RESULTS")
print("="*70)

quran_count = 0
hadith_count = 0

for i, (detail, distance) in enumerate(zip(details, distances), 1):
    if 'surah_name' in detail:
        source = f"Quran - Surah {detail['surah_name']}, {detail['verse_number']}"
        quran_count += 1
    elif 'source' in detail:
        source = detail.get('source', 'Unknown')
        hadith_count += 1
    else:
        source = "Unknown"
    
    print(f"\n[{i}] Distance: {distance:.4f}")
    print(f"Source: {source}")
    print(f"Text: {detail['text'][:150]}...")

print("\n" + "="*70)
print(f"Quran verses: {quran_count}")
print(f"Hadith: {hadith_count}")
print("="*70)

# Check data distribution
print("\nDATA STATISTICS:")
quran_in_data = sum(1 for c in all_chunks if 'surah_name' in c)
hadith_in_data = sum(1 for c in all_chunks if 'source' in c and 'Hadith' in c.get('source', ''))
print(f"Total Quran verses available: {quran_in_data}")
print(f"Total Hadith available: {hadith_in_data}")
print(f"Total chunks: {len(all_chunks)}")
