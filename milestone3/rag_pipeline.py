# ==========================================
# IMPORTS
# ==========================================

from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


import pandas as pd
import numpy as np
import faiss
import json
import re
import os

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

TRANSCRIPTS_DIR = DATA_DIR / "Transcripts"
QA_DIR = DATA_DIR / "QA"

# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv(BASE_DIR.parent / ".env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

print("Google key loaded:", GOOGLE_API_KEY is not None)
print("Groq key loaded:", GROQ_API_KEY is not None)


# ==========================================
# LOAD FILE LISTS
# ==========================================

transcript_files = sorted(
    TRANSCRIPTS_DIR.glob("*.txt")
)

qa_files = sorted(
    QA_DIR.glob("*.json")
)

print("Transcripts folder exists:", TRANSCRIPTS_DIR.exists())
print("QA folder exists:", QA_DIR.exists())

print("Number of transcript files:", len(transcript_files))
print("Number of QA files:", len(qa_files))


# ==========================================
# LOAD TRANSCRIPT FILES
# ==========================================

def load_transcripts(transcript_files):
    """
    Loads transcript text files and converts them
    into a structured pandas DataFrame.

    Expected line format:
    timestamp: text
    """

    rows = []

    for file_path in transcript_files:

        episode_name = file_path.stem

        with open(file_path, "r", encoding="utf-8") as f:

            for line in f:

                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Match timestamp + text
                match = re.match(
                    r"^([\d.]+):\s*(.*)$",
                    line
                )

                if match:

                    timestamp = float(match.group(1))

                    text = match.group(2).strip()

                    # Skip empty transcript text
                    if not text:
                        continue

                    rows.append({
                        "episode": episode_name,
                        "timestamp": timestamp,
                        "text": text
                    })

    return pd.DataFrame(rows)

# ==========================================
# LOAD TRANSCRIPTS DATAFRAME
# ==========================================

transcripts_df = load_transcripts(
    transcript_files
)

print("Total transcript rows:", len(transcripts_df))

print(transcripts_df.head())

# ==========================================
# TRANSCRIPT STATISTICS
# ==========================================

print(
    transcripts_df.groupby("episode")
    .size()
    .reset_index(name="num_lines")
)


# ==========================================
# LIMIT DATASET TO SELECTED EPISODES
# ==========================================

SELECTED_EPISODES = [
    "أعظم طائرة حربية  الدحيح",
    "الساموراي  الدحيح",
    "هل Citizen Kane أفضل فيلم في التاريخ؟  الدحيح",
    "الأخطبوط  الدحيح"
]

transcripts_df = transcripts_df[
    transcripts_df["episode"].isin(SELECTED_EPISODES)
].reset_index(drop=True)

print("Selected episodes:", transcripts_df["episode"].nunique())
print("Rows after filtering:", len(transcripts_df))

print(
    transcripts_df.groupby("episode")
    .size()
    .reset_index(name="num_lines")
)

# ==========================================
# MS3-SAFE TEXT NORMALIZATION
# ==========================================

"""
Normalization constraints for MS3:
- No stemming
- No lemmatization
- No punctuation removal
- No English token removal
- Preserve dialectal Arabic and Arabic-English code-switching
"""

AR_PUNCT_MAP = {
    ",": "،",
    ";": "؛",
    "?": "؟",
    "\"": "«",
    "“": "«",
    "”": "»"
}

PUNCT_RE = re.compile(
    "|".join(re.escape(k) for k in AR_PUNCT_MAP.keys())
)

def remove_noise_tags(text: str) -> str:
    """
    Removes non-speech tags such as [موسيقى].
    This does not remove actual spoken content.
    """
    return re.sub(r"\[.*?\]", "", text).strip()


def normalize_ms3_text(text: str) -> str:
    """
    Applies light normalization suitable for MS3 RAG.

    The goal is to clean formatting while preserving
    the natural transcript language.
    """

    if not isinstance(text, str):
        return ""

    # Remove non-speech tags only
    text = remove_noise_tags(text)

    # Standardize selected punctuation without removing punctuation
    text = PUNCT_RE.sub(
        lambda m: AR_PUNCT_MAP[m.group(0)],
        text
    )

    # Add spaces between Arabic and English/numeric tokens
    text = re.sub(
        r"([\u0600-\u06FF])([A-Za-z\d])",
        r"\1 \2",
        text
    )

    text = re.sub(
        r"([A-Za-z\d])([\u0600-\u06FF])",
        r"\1 \2",
        text
    )

    # Normalize whitespace only
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply normalization
transcripts_df["normalized_text"] = transcripts_df["text"].apply(
    normalize_ms3_text
)

print(
    "normalized_text column created:",
    "normalized_text" in transcripts_df.columns
)

print(
    transcripts_df[
        ["episode", "timestamp", "text", "normalized_text"]
    ].head()
)

print("Original sample:")
print(transcripts_df["text"].iloc[0])

print("\nNormalized sample:")
print(transcripts_df["normalized_text"].iloc[0])


# ==========================================
# CHUNKING STRATEGY
# ==========================================

CHUNK_SIZE = 12
CHUNK_OVERLAP = 3
MIN_WORDS = 20

def create_chunks(transcripts_df, chunk_size=12, chunk_overlap=3, min_words=20):
    """
    Creates overlapping chunks from transcript lines.

    Each chunk remains traceable to:
    - source episode
    - start timestamp
    - end timestamp
    """

    chunks = []

    for episode in transcripts_df["episode"].unique():

        episode_df = transcripts_df[
            transcripts_df["episode"] == episode
        ].reset_index(drop=True)

        texts = episode_df["normalized_text"].tolist()
        timestamps = episode_df["timestamp"].tolist()

        start = 0

        while start < len(texts):

            end = min(start + chunk_size, len(texts))

            chunk_text = " ".join(texts[start:end])

            num_words = len(chunk_text.split())

            if num_words >= min_words:
                chunks.append({
                    "episode": episode,
                    "start_timestamp": timestamps[start],
                    "end_timestamp": timestamps[end - 1],
                    "chunk_text": chunk_text,
                    "num_words": num_words,
                    "num_chars": len(chunk_text)
                })

            start += chunk_size - chunk_overlap

    return pd.DataFrame(chunks)


chunks_df = create_chunks(
    transcripts_df,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    min_words=MIN_WORDS
)

print("Total chunks:", len(chunks_df))
print("Average words per chunk:", round(chunks_df["num_words"].mean(), 2))
print("Minimum words:", chunks_df["num_words"].min())
print("Maximum words:", chunks_df["num_words"].max())

print(chunks_df.head())

# ==========================================
# EMBEDDING MODEL
# ==========================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

texts = chunks_df["chunk_text"].tolist()

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embeddings shape:", embeddings.shape)

# ==========================================
# BUILD FAISS VECTOR STORE
# ==========================================

embedding_dim = embeddings.shape[1]

index = faiss.IndexFlatIP(embedding_dim)

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)


# ==========================================
# KEYWORD-AWARE RETRIEVAL FUNCTION
# ==========================================

def extract_query_terms(query):
    """
    Extracts simple Arabic/English terms from the query.
    Used only for reranking retrieved semantic candidates.
    """

    query = query.lower()

    tokens = re.findall(
        r"[\u0600-\u06FFA-Za-z0-9]+",
        query
    )

    tokens = [
        token for token in tokens
        if len(token) > 2
    ]

    return set(tokens)


def keyword_score(query_terms, text):
    """
    Computes how many query terms appear in the chunk text.
    """

    text = text.lower()

    if not query_terms:
        return 0

    matches = sum(
        1 for term in query_terms
        if term in text
    )

    return matches / len(query_terms)


def retrieve_chunks(
    query,
    top_k=5,
    fetch_k=60,
    min_score=0.25,
    episode_filter=None
):
    """
    Hybrid retrieval:
    1. FAISS retrieves semantic candidates.
    2. Keyword overlap reranks the candidates.
    3. Final top_k chunks are returned.
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        fetch_k
    )

    query_terms = extract_query_terms(query)

    results = []

    for semantic_score, idx in zip(scores[0], indices[0]):

        row = chunks_df.iloc[idx]

        if float(semantic_score) < min_score:
            continue

        if episode_filter is not None and row["episode"] != episode_filter:
            continue

        k_score = keyword_score(
            query_terms,
            row["chunk_text"]
        )

        final_score = (
            0.70 * float(semantic_score)
        ) + (
            0.30 * k_score
        )

        results.append({
            "semantic_score": float(semantic_score),
            "keyword_score": k_score,
            "final_score": final_score,
            "episode": row["episode"],
            "start_timestamp": row["start_timestamp"],
            "end_timestamp": row["end_timestamp"],
            "chunk_text": row["chunk_text"]
        })

    results = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return results[:top_k]


# ==========================================
# CONTEXT CONSTRUCTION
# ==========================================

def build_context(retrieved_chunks):
    """
    Builds a structured context string from retrieved chunks.
    Each chunk keeps its episode and timestamp for traceability.
    """

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        source = (
            f"[Source {i}] "
            f"Episode: {chunk['episode']} | "
            f"Time: {chunk['start_timestamp']} - "
            f"{chunk['end_timestamp']}"
        )

        text = chunk["chunk_text"]

        context_parts.append(
            source + "\n" + text
        )

    return "\n\n".join(context_parts)

# ==========================================
# OUT-OF-DOMAIN DETECTION
# ==========================================

OOD_THRESHOLD = 0.25

def is_out_of_domain(retrieved_chunks, threshold=OOD_THRESHOLD):
    """
    Detects whether the query is outside the transcript knowledge base.

    If no retrieved chunks are found, or the best retrieval score is too low,
    the query is considered out-of-domain.
    """

    if not retrieved_chunks:
        return True

    best_score = retrieved_chunks[0].get("final_score", 0)

    return best_score < threshold


# ==========================================
# SYSTEM PROMPT
# ==========================================

SYSTEM_PROMPT = """
أنت مساعد ذكي يعتمد فقط على المعلومات الموجودة في السياق المسترجع.

قواعد مهمة:
- أجب باستخدام المعلومات الموجودة في السياق فقط.
- يمكنك إعادة صياغة وشرح المعلومات الموجودة بوضوح.
- لا تضف أي معلومات غير موجودة في السياق.
- إذا كان السياق لا يحتوي على معلومات كافية فعلًا، قل:
"لا أملك معلومات كافية للإجابة من البيانات المتاحة."
- يمكنك الإجابة بالعربية أو الإنجليزية حسب لغة السؤال.
- حاول أن تكون الإجابة واضحة ومختصرة.
"""

# ==========================================
# FINAL PROMPT CONSTRUCTION
# ==========================================

def build_prompt(query, context):
    """
    Builds the final prompt passed to the language model.
    """

    prompt = f"""
{SYSTEM_PROMPT}

السياق:
{context}

السؤال:
{query}

الإجابة:
"""

    return prompt


# ==========================================
# INITIALIZE GEMINI MODEL
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

print("Gemini model initialized successfully.")

# ==========================================
# INITIALIZE GROQ FALLBACK MODEL
# ==========================================

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
)

print("Groq model initialized successfully.")





# ==========================================
# LLM CALL WITH RETRY
# ==========================================

def call_llm_with_retry(
    llm_model,
    prompt,
    max_retries=2
):
    """
    Calls an LLM with a retry mechanism.
    Retries API calls before failing.
    """

    last_error = None

    for attempt in range(max_retries + 1):

        try:
            response = llm_model.invoke(prompt)
            return response

        except Exception as e:
            last_error = e

    raise last_error


# ==========================================
# RAG ANSWER GENERATION PIPELINE
# ==========================================

def generate_answer(query, top_k=5):

    retrieved_chunks = retrieve_chunks(
        query,
        top_k=top_k
    )

    context = build_context(retrieved_chunks)

    # ==========================================
    # OUT-OF-DOMAIN CHECK
    # ==========================================

    if is_out_of_domain(retrieved_chunks):

        return {
            "query": query,
            "response": (
                "لا أملك معلومات كافية للإجابة من البيانات المتاحة.\n"
                "I do not have enough information to answer from the available data."
            ),
            "retrieved_chunks": retrieved_chunks,
            "context": "",
            "status": "out_of_domain",
            "model_used": None
        }

    prompt = build_prompt(query, context)

    # ==========================================
    # TRY GEMINI FIRST
    # ==========================================

    try:
        response = call_llm_with_retry(
            llm,
            prompt
        )

        return {
            "query": query,
            "response": response.content,
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "status": "success",
            "model_used": "gemini-2.0-flash"
        }

    except Exception as gemini_error:

        # ==========================================
        # FALLBACK TO GROQ
        # ==========================================

        try:
            response = call_llm_with_retry(
                groq_llm,
                prompt
            )

            return {
                "query": query,
                "response": response.content,
                "retrieved_chunks": retrieved_chunks,
                "context": context,
                "status": "success",
                "model_used": "llama-3.3-70b-versatile"
            }

        except Exception as groq_error:

                return {
                    "query": query,
                    "response": (
                        "حدث خطأ أثناء استدعاء نماذج اللغة. "
                        "يرجى المحاولة لاحقًا."
                    ),
                    "retrieved_chunks": retrieved_chunks,
                    "context": context,
                    "status": "error",
                    "model_used": None,
                    "error_message": (
                        f"Gemini error: {gemini_error} | "
                        f"Groq error: {groq_error} | "
                    
                    )
                }
            


# ==========================================
# LANGCHAIN RAG CHAIN WRAPPER
# ==========================================

from langchain_core.runnables import RunnableLambda

def langchain_rag_function(query):
    """
    LangChain-compatible wrapper around the custom RAG pipeline.
    This keeps our FAISS hybrid retriever while exposing the RAG flow
    as a LangChain Runnable chain.
    """

    return generate_answer(
        query=query,
        top_k=5
    )


rag_chain = RunnableLambda(langchain_rag_function)

print("LangChain RAG chain initialized successfully.")


# ==========================================
# CONTEXT WINDOW STRATEGIES
# ==========================================

def get_context_window(history, strategy="sliding_window", max_turns=3):
    """
    Returns conversation history using different context window strategies.

    Strategies:
    - full_history: use all previous turns
    - sliding_window: use last N turns
    - strict_truncation: keep first turn + last N-1 turns
    - summarized_history: use a simple summary placeholder + recent turns
    """

    if strategy == "full_history":
        selected_history = history

    elif strategy == "sliding_window":
        selected_history = history[-max_turns:]

    elif strategy == "strict_truncation":
        if len(history) <= max_turns:
            selected_history = history
        else:
            selected_history = [history[0]] + history[-(max_turns - 1):]

    elif strategy == "summarized_history":
        if len(history) <= max_turns:
            selected_history = history
        else:
            summary_turn = {
                "user": "Conversation summary",
                "assistant": "Previous conversation discussed earlier user questions and assistant answers."
            }
            selected_history = [summary_turn] + history[-(max_turns - 1):]

    else:
        selected_history = history[-max_turns:]

    return selected_history


# ==========================================
# MULTI-TURN CHAT MEMORY
# ==========================================

conversation_history = []

def format_chat_history(
    history,
    max_turns=3,
    strategy="sliding_window"
):
    """
    Formats chat history according to the selected context window strategy.
    """

    selected_history = get_context_window(
        history,
        strategy=strategy,
        max_turns=max_turns
    )

    formatted_history = []

    for turn in selected_history:
        formatted_history.append(
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        )

    return "\n\n".join(formatted_history)


def build_chat_prompt(query, context, chat_history):
    """
    Builds a multi-turn prompt using retrieved context
    plus recent conversation history.
    """

    prompt = f"""
{SYSTEM_PROMPT}

سجل المحادثة السابق:
{chat_history}

السياق المسترجع:
{context}

السؤال الحالي:
{query}

الإجابة:
"""

    return prompt


def chat_with_memory(query, top_k=5, max_turns=3, memory_strategy="sliding_window"):
    """
    Multi-turn RAG chatbot with configurable memory strategies.
    """

    # ==========================================
    # HISTORY-AWARE RETRIEVAL QUERY
    # ==========================================

    retrieval_query = query

    if conversation_history:
        last_user_query = conversation_history[-1]["user"]

        retrieval_query = (
            last_user_query + " " + query
        )

    retrieved_chunks = retrieve_chunks(
        retrieval_query,
        top_k=top_k
    )

    # ==========================================
    # OUT-OF-DOMAIN CHECK
    # ==========================================

    if is_out_of_domain(retrieved_chunks):

        return {
            "query": query,
            "response": (
                "لا أملك معلومات كافية للإجابة من البيانات المتاحة.\n"
                "I do not have enough information to answer from the available data."
            ),
            "retrieved_chunks": retrieved_chunks,
            "context": "",
            "chat_history": "",
            "status": "out_of_domain",
            "model_used": None
        }

    context = build_context(retrieved_chunks)

    chat_history = format_chat_history(
        conversation_history,
        max_turns=max_turns,
        strategy=memory_strategy
    )

    prompt = build_chat_prompt(
        query,
        context,
        chat_history
    )

    # ==========================================
    # TRY GEMINI → GROQ 
    # ==========================================

    try:
        response = call_llm_with_retry(
            llm,
            prompt
        )
        model_used = "gemini-2.0-flash"

    except Exception:

        try:
            response = call_llm_with_retry(
                groq_llm,
                prompt
            )
            model_used = "llama-3.3-70b-versatile"

        except Exception:
            return {
                "query": query,
                "response": (
                    "حدث خطأ أثناء استدعاء نماذج اللغة. "
                    "يرجى المحاولة لاحقًا."
                ),
                "retrieved_chunks": retrieved_chunks,
                "context": context,
                "chat_history": chat_history,
                "status": "error",
                "model_used": None
            }

    answer = response.content

    conversation_history.append({
        "user": query,
        "assistant": answer
    })

    return {
        "query": query,
        "response": answer,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "chat_history": chat_history,
        "status": "success",
        "model_used": model_used
    }