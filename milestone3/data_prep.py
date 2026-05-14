import os
import re
import faiss
import pickle
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. CONFIGURATION & DATA LIMIT FIX
# ==========================================
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "Transcripts"

# STRCIT LIMIT: Select exactly 3 episodes to satisfy MS1 requirements
SELECTED_EPISODES = [
    "أعظم طائرة حربية  الدحيح.txt",
    "الساموراي  الدحيح.txt",
    "تاج محل  الدحيح.txt"
]

CHUNK_SIZE = 12
CHUNK_OVERLAP = 3
MIN_WORDS = 20

AR_PUNCT_MAP = {",": "،", ";": "؛", "?": "؟", "\"": "«", "“": "«", "”": "»"}
PUNCT_RE = re.compile("|".join(re.escape(k) for k in AR_PUNCT_MAP.keys()))

# ==========================================
# 2. NORMALIZATION FUNCTIONS
# ==========================================
def strip_timestamp(line: str) -> str:
    timestamp_pattern = r"^(\d+[\.:]\d+([\.:]\d+)?[:]?)\s*"
    return re.sub(timestamp_pattern, "", line).strip()

def remove_noise_tags(text: str) -> str:
    return re.sub(r"\[.*?\]", "", text).strip()

def normalize_ms3_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = strip_timestamp(text)
    text = remove_noise_tags(text)
    text = PUNCT_RE.sub(lambda m: AR_PUNCT_MAP[m.group(0)], text)
    text = re.sub(r"([\u0600-\u06FF])([A-Za-z\d])", r"\1 \2", text)
    text = re.sub(r"([A-Za-z\d])([\u0600-\u06FF])", r"\1 \2", text)
    return re.sub(r"\s+", " ", text).strip()

# ==========================================
# 3. PROCESSING PIPELINE
# ==========================================
def process_data():
    print("Loading and filtering transcripts...")
    rows = []
    
    # Only load the 3 selected episodes
    for ep_name in SELECTED_EPISODES:
        file_path = TRANSCRIPTS_DIR / ep_name
        if not file_path.exists():
            print(f"Warning: {ep_name} not found!")
            continue
            
        episode_name = file_path.stem
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                match = re.match(r"^([\d.]+):\s*(.*)$", line)
                if match:
                    rows.append({
                        "episode": episode_name,
                        "timestamp": match.group(1),
                        "text": match.group(2).strip()
                    })

    df = pd.DataFrame(rows)
    df["normalized_text"] = df["text"].apply(normalize_ms3_text)
    print(f"Loaded {len(df)} rows across {df['episode'].nunique()} episodes.")

    print("Chunking texts...")
    chunks = []
    for episode in df["episode"].unique():
        ep_df = df[df["episode"] == episode].reset_index(drop=True)
        texts = ep_df["normalized_text"].tolist()
        timestamps = ep_df["timestamp"].tolist()
        
        start = 0
        while start < len(texts):
            end = min(start + CHUNK_SIZE, len(texts))
            chunk_text = " ".join(texts[start:end])
            
            # Count words to filter short chunks
            if len(chunk_text.split()) >= MIN_WORDS:
                chunks.append({
                    "episode": episode,
                    "start_timestamp": timestamps[start],
                    "end_timestamp": timestamps[end - 1],
                    "chunk_text": chunk_text
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP

    chunks_df = pd.DataFrame(chunks)
    print(f"Generated {len(chunks_df)} chunks.")

    print("Generating embeddings (this may take a minute)...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embeddings = embedder.encode(
        chunks_df["chunk_text"].tolist(), 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True
    )

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # ==========================================
    # 4. EXPORT ARTIFACTS
    # ==========================================
    os.makedirs("vector_store", exist_ok=True)
    
    faiss.write_index(index, "vector_store/ms3_index.bin")
    
    with open("vector_store/chunks_metadata.pkl", "wb") as f:
        pickle.dump(chunks_df.to_dict('records'), f)

    print("✅ Pipeline complete. Index and metadata saved to /vector_store")

if __name__ == "__main__":
    process_data()