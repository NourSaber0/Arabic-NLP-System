# System Instructions: Milestone 3 - Retrieval-Augmented Generation (RAG) System

This document serves as the absolute specification, architectural blueprint, and runtime guidance file (`gemini.md`) for building, testing, and evaluating the Milestone 3 multi-turn Arabic/English RAG conversational chatbot.

---

## 1. System Objective & Strict Constraints

The system must function as a multi-turn conversational chatbot that answers user queries **strictly based on retrieved context** from Arabic podcast/audio transcripts.

### Critical Constraints:
* **No Model Training:** You are strictly prohibited from fine-tuning or training any model. All intelligence must come from retrieval, prompt engineering, and context management.
* **Text Preservation:** Use the normalized natural text from MS1. 
  * Do **NOT** remove punctuation.
  * Do **NOT** perform stemming or lemmatization.
  * Do **NOT** remove English tokens.
  * **Must** preserve dialectal variation and Arabic-English code-switching.
* **Grounded Generation:** The system must refuse to answer if the information is not present in the retrieved context. No hallucinations allowed.

---

## 2. Architectural Blueprint & Requirements

### 2.1 Data Architecture & Chunking
* **Scope:** 3–5 episodes from the original MS1 transcripts.
* **Chunking Strategy:** Define a fixed chunk size and overlap optimized for semantic coherence in dialectal Arabic.
* **Metadata Tracking:** Each chunk must explicitly store metadata tracing it back to its source episode (e.g., `episode_id`, `timestamp_range`).

### 2.2 Embedding & Vector Store Pipeline
* **Embedding Model:** Must use a multilingual embedding model capable of capturing Arabic semantics while simultaneously preserving the meaning of embedded English tokens (e.g., `text-embedding-3-small`, `multilingual-e5`, or Cohere Multilingual).
* **Vector Database:** Choose and configure a vector database (e.g., Chroma, FAISS, Pinecone) to index the chunked transcripts.

### 2.3 Multi-Turn Chatbot (LangChain Architecture)
* **Framework:** Powered completely via LangChain.
* **Memory Strategy:** Must implement and evaluate a stateful memory strategy to handle continuous user dialogue.

---

## 3. Comparative Experiments Matrix

The system must implement and allow toggleable comparison between the following components:

### 3.1 Prompting Strategies
| Strategy Type | Configuration A | Configuration B |
| :--- | :--- | :--- |
| **Directive Style** | **System-Guided Prompts:** Explicit rules, guardrails, and strict negative constraints. | **Minimal Prompts:** Simple instructions relying purely on LLM baseline compliance. |
| **Language Profile** | **Arabic Prompts:** System prompts written completely in Modern Standard Arabic (MSA). | **English Prompts:** System prompts written completely in English. |

### 3.2 Context Window & History Management
You must explicitly implement and compare these 4 history configurations:
1. **Full-History Context:** Appends every previous turn directly into the prompt context window until limit exhaustion.
2. **Sliding Window:** Keeps only the last $N$ turns of conversation to manage token usage efficiently.
3. **Strict Truncation:** Hard-cuts history at a specific token threshold without summarizing.
4. **Summarized-History:** Uses a secondary LLM call to compress older turns into a rolling summary paragraph.

---

## 4. Guardrails & Robustness

### 4.1 Out-of-Domain (OOD) Detection
* **Mechanism:** Implement a classification or distance-threshold layer to intercept queries unrelated to the transcript datasets.
* **Behavior:** If a query is determined to be OOD, bypass retrieval and return a predefined, polite rejection message.
* **Evaluation:** Must justify the OOD detection threshold/method in the final technical logs.

### 4.2 Error Handling & Resilience Layer
The system must be engineered to **never crash** during runtime.
* **Retry Mechanism:** Implement exponential backoff for API timeouts or rate limits (HTTP 429/5xx).
* **Fallback Strategy:** If the primary free-tier API fails repeatedly, gracefully swap to an alternative backup model provider seamlessly.

---

## 5. Model & Evaluation Framework

### 5.1 LLM Requirements
* Use **API-based** LLMs only.
* Restrict selections to **free-tier** access models (e.g., Gemini 1.5 Flash, Groq/Llama-3, Cohere).
* System must fully support processing and generating mixed Arabic and English responses.

### 5.2 Metrics & Validation
Evaluate system configurations using the QA pairs generated during MS2 across at least **two different LLMs** using these 3 mandatory categories:

> ### RAG Metric Categories
> * **Text Generation Quality:** Evaluates fluency and syntactic coherence.
> * **Semantic Correctness:** Measures how accurately the answer addresses the user's intent.
> * **Grounding to Retrieved Context (Faithfulness):** Validates that no facts outside the retrieved chunks were introduced.

---

## 6. Target Interface Layout (Streamlit)

The UI must expose both the consumer application and the engineering visibility logs simultaneously:

* **Chat Interface:** Multi-turn conversation window supporting mixed-language queries.
* **Sidebar Controls:** Dropdowns to dynamically switch between:
    * LLM Provider Engine
    * Prompt Type (System-Guided vs Minimal | Arabic vs English)
    * Context Window Strategy (Full, Sliding, Truncated, Summarized)
* **Developer Log Panel:** Real-time expander modules displaying:
    * Retrieved Chunks & Similarity Scores
    * Latency Metrics & Token Counts
    * OOD Confidence Scores
    * Active Model Fallback status

---

## 7. Operational Prompt Blueprint

Use this system prompt template within your LangChain configuration to enforce strict constraints:

```text
You are a strict, faithful RAG assistant specializing in analyzing specific Arabic transcripts. 

CRITICAL INSTRUCTIONS:
1. Answer the user's query USING ONLY the provided text blocks in the context section below.
2. Do not use any outside knowledge, assumptions, or extrapolations. 
3. If the context does not contain the answer to the query, respond exactly with the following rejection message: "عذرًا، هذا السؤال خارج نطاق الحلقات المتوفرة." / "Sorry, this question is outside the scope of the available episodes."
4. Maintain the exact code-switching behavior (Arabic/English mix) and dialectal tone present within the provided context.
5. Do not summarize or alter technical English tokens or terms found within the Arabic text.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

USER QUERY:
{question}

FAITHFUL ANSWER:
```
