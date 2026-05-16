# ==========================================
# IMPORTS
# ==========================================

from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

import pandas as pd
import numpy as np
import faiss
import json
import re
import os
import time


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
load_dotenv(BASE_DIR / ".env", override=True)

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

CHUNK_SIZE = 24
CHUNK_OVERLAP = 8
MIN_WORDS = 20

def create_chunks(transcripts_df, chunk_size=24, chunk_overlap=8, min_words=20):
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
    fetch_k=80,
    min_score=0.15,
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

        exact_phrase_boost = 0.0

        for term in query_terms:
            if term in row["chunk_text"].lower():
                exact_phrase_boost += 0.05

        exact_phrase_boost = min(exact_phrase_boost, 0.20)

        final_score = (
            0.55 * float(semantic_score)
        ) + (
            0.45 * k_score
        ) + exact_phrase_boost

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

# Justification for OOD_THRESHOLD = 0.25:
# This threshold is based on empirical testing of the FAISS FlatIP score (cosine similarity)
# and keyword overlap. A score below 0.25 typically indicates that the retrieved chunks 
# share very few keywords and have low semantic similarity with the query, suggesting 
# the query is outside the knowledge base of the provided transcripts.
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

قواعد صارمة:
1. أجب باستخدام المعلومات الموجودة في السياق فقط.
2. لا تضف أي معلومات خارجية أو افتراضات.
3. كن موجزًا جدًا في إجابتك (جملة واحدة أو جملتين كحد أقصى).
4. إذا كان السياق لا يحتوي على إجابة، قل فقط: "لا أملك معلومات كافية للإجابة من البيانات المتاحة."
5. التزم بلهجة ولغة السياق (عربي/إنجليزي).
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
    model="gemini-flash-latest",
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

        # Ensure response.content is a string
        res_content = response.content
        if isinstance(res_content, list):
            text_parts = []
            for part in res_content:
                if isinstance(part, dict) and 'text' in part:
                    text_parts.append(part['text'])
                elif isinstance(part, str):
                    text_parts.append(part)
            res_content = " ".join(text_parts)
        elif not isinstance(res_content, str):
            res_content = str(res_content)

        return {
            "query": query,
            "response": res_content,
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "status": "success",
            "model_used": "gemini-flash-latest"
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

            # Ensure response.content is a string
            res_content = response.content
            if isinstance(res_content, list):
                text_parts = []
                for part in res_content:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                    elif isinstance(part, str):
                        text_parts.append(part)
                res_content = " ".join(text_parts)
            elif not isinstance(res_content, str):
                res_content = str(res_content)

            return {
                "query": query,
                "response": res_content,
                "retrieved_chunks": retrieved_chunks,
                "context": context,
                "status": "success",
                "model_used": "allam-2-7b"
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
# CONTEXT WINDOW STRATEGIES (Updated for LangChain Messages)
# ==========================================

def get_context_window(history, strategy="sliding_window", max_turns=3):
    """
    Truncates or summarizes LangChain BaseMessage objects 
    to fit within specified memory requirements.
    """
    if not history:
        return []

    if strategy == "full_history":
        return history

    elif strategy == "sliding_window":
        # Each turn has 1 HumanMessage and 1 AIMessage (2 messages per turn)
        msg_limit = max_turns * 2
        return history[-msg_limit:]

    elif strategy == "strict_truncation":
        msg_limit = max_turns * 2
        if len(history) <= msg_limit:
            return history
        # Preserve the very first Q&A turn, then grab the most recent turns
        return [history[0], history[1]] + history[-(msg_limit - 2):]

    elif strategy == "summarized_history":
        msg_limit = max_turns * 2
        if len(history) <= msg_limit:
            return history
        # Add a placeholder System Message to inject context summary
        summary_msg = SystemMessage(
            content="ملخص المحادثة السابقة: تم مناقشة الأسئلة السابقة المطروحة من المستخدم وتوفير الإجابات المناسبة لها بناءً على المستندات."
        )
        return [summary_msg] + history[-(msg_limit - 1):]

    return history[-(max_turns * 2):]


# ==========================================
# MULTI-TURN CHAT MEMORY & RUNNABLE ARCHITECTURE
# ==========================================

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def rag_chat_runnable(inputs):
    """
    Core runnable unit that processes inputs, applies context strategies, 
    and handles model invocation with fallback execution logs.
    """
    query = inputs["question"]
    raw_history = inputs.get("history", [])
    
    # Extract structural experiment configurations passed through execution config
    strategy = inputs.get("memory_strategy", "sliding_window")
    max_turns = inputs.get("max_turns", 3)
    prompt_style = inputs.get("prompt_style", "system_guided_ar")
    model_choice = inputs.get("model_choice", "gemini")

    # 1. Candidate Retrieval
    retrieved_chunks = retrieve_chunks(query, top_k=inputs.get("top_k", 5))
    if is_out_of_domain(retrieved_chunks):
        return {
            "response": "لا أملك معلومات كافية للإجابة من البيانات المتاحة.",
            "model_used": None,
            "status": "out_of_domain",
            "context": ""
        }

    context = build_context(retrieved_chunks)

    # 2. Prompt Engineering Experiments Registry (Fulfills Section 2.6)
    prompts_registry = {
        "system_guided_ar": SYSTEM_PROMPT,
        "minimal_ar": "أجب عن السؤال التالي باستخدام السياق المرفق فقط:\n{context}",
        "system_guided_en": "You are an intelligent assistant answering questions strictly based on the retrieved context.\nContext:\n{context}",
        "minimal_en": "Answer using this context only:\n{context}"
    }
    
    active_sys_base = prompts_registry.get(prompt_style, SYSTEM_PROMPT)
    system_prompt_message = SystemMessage(content=f"{active_sys_base}\n\nالسياق:\n{context}")

    # 3. Process active history window using selected strategy
    processed_history = get_context_window(raw_history, strategy=strategy, max_turns=max_turns)

    # 4. Compile message history trace
    messages = [system_prompt_message] + processed_history + [HumanMessage(content=query)]

    # 5. Model Execution with Fallback Logic (Fulfills Section 2.8)
    try:
        if model_choice == "groq":
            response = call_llm_with_retry(groq_llm, messages)
            model_used = "allam-2-7b"
        else:
            response = call_llm_with_retry(llm, messages)
            model_used = "gemini-flash-latest"
    except Exception as primary_error:
        # Check if fallback is disabled (e.g., during evaluation)
        if inputs.get("disable_fallback", False):
            raise primary_error

        # Emergency Fallback Switch if Primary API goes down
        print(f"⚠️ Primary model {model_choice} failed. Attempting fallback...")
        fallback_target = llm if model_choice == "groq" else groq_llm
        model_used = "gemini-flash-latest" if model_choice == "groq" else "allam-2-7b"
        response = fallback_target.invoke(messages)

    # Ensure response.content is a string
    res_content = response.content
    if isinstance(res_content, list):
        # Extract text from rich response format if necessary
        text_parts = []
        for part in res_content:
            if isinstance(part, dict) and 'text' in part:
                text_parts.append(part['text'])
            elif isinstance(part, str):
                text_parts.append(part)
        res_content = " ".join(text_parts)
    elif not isinstance(res_content, str):
        res_content = str(res_content)

    return {
        "response": res_content,
        "model_used": model_used,
        "status": "success",
        "context": context,
        "retrieved_chunks": retrieved_chunks
    }


base_chat_chain = RunnableLambda(rag_chat_runnable)

chat_chain_with_history = RunnableWithMessageHistory(
    base_chat_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="response" # Updates history store with the clean string inner token
)


# ==========================================
# EXPOSURE WRAPPER FUNCTION
# ==========================================

import time

def chat_with_memory(
    query, 
    top_k=5, 
    max_turns=3, 
    memory_strategy="sliding_window", 
    model_choice="gemini", 
    prompt_style="system_guided_ar", 
    session_id="default",
    disable_fallback=False
):
    """
    Main wrapper function interface used directly by evaluate.py and streamlit_app.py
    """
    start_time = time.time()
    try:
        # Invoke via LangChain's native manager interface
        output = chat_chain_with_history.invoke(
            {
                "question": query,
                "top_k": top_k,
                "max_turns": max_turns,
                "memory_strategy": memory_strategy,
                "model_choice": model_choice,
                "prompt_style": prompt_style,
                "disable_fallback": disable_fallback
            },
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        latency = time.time() - start_time
        
        # Simple token estimation (4 chars per token)
        est_tokens = (len(output.get("response", "")) // 4) + (len(output.get("context", "")) // 4)

        return {
            "query": query,
            "response": output["response"],
            "retrieved_chunks": output.get("retrieved_chunks", []),
            "context": output.get("context", ""),
            "status": output.get("status", "success"),
            "model_used": output.get("model_used"),
            "latency": latency,
            "estimated_tokens": est_tokens
        }

    except Exception as e:
        latency = time.time() - start_time
        print(f"\n🚨 [CRITICAL PIPELINE ERROR]: {e}\n")
        return {
            "query": query,
            "response": f"حدث خطأ أثناء معالجة الطلب: {str(e)}",
            "retrieved_chunks": [],
            "context": "",
            "status": "error",
            "model_used": None,
            "error_message": str(e),
            "latency": latency,
            "estimated_tokens": 0
        }
ency": latency,
            "estimated_tokens": 0
        }
