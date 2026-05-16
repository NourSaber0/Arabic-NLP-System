from rag_pipeline import chunks_df
target_mask = chunks_df['chunk_text'].str.contains('2011') & chunks_df['chunk_text'].str.contains('جرب')
target_chunks = chunks_df[target_mask]
print(f"Found {len(target_chunks)} chunks with both 2011 and 'جرب'")
for idx, row in target_chunks.iterrows():
    print(f"--- Chunk {idx} (Episode: {row['episode']}) ---")
    print(row['chunk_text'])
