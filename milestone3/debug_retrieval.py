from rag_pipeline import retrieve_chunks, build_context
import json

query = "في أي شهر وسنة جرت تجربة الطائرة المذكورة؟"
chunks = retrieve_chunks(query, top_k=5)

print(f"Query: {query}\n")
for i, c in enumerate(chunks, 1):
    print(f"Rank {i} | Episode: {c['episode']} | Score: {c['final_score']:.4f}")
    print(f"Text: {c['chunk_text'][:200]}...\n")
