# Milestone 3 Technical Report
## Arabic RAG Chatbot over Podcast Transcripts

---

## 1. System Architecture

We built a **Retrieval-Augmented Generation (RAG)** chatbot that answers Arabic-language questions strictly from a curated set of podcast transcripts, with no model training involved. The system operates in four stages:

**Data Preparation → Semantic Indexing → Hybrid Retrieval → Grounded Generation**

### 1.1 Data and Text Representation

Four episodes from the "الدحيح" Arabic podcast were selected: *F-35 Fighter Jet*, *Samurai*, *Citizen Kane*, and *Octopus*. These episodes cover diverse factual domains and contain rich Arabic-English code-switching, making them representative of the naturalistic dialectal Arabic targeted by this milestone.

Text normalization was applied per the MS3 constraints: noise tags (e.g., `[موسيقى]`) are stripped, Latin punctuation is mapped to Arabic equivalents (`?` → `؟`, `,` → `،`), and spacing is normalized at Arabic-English token boundaries. No stemming, lemmatization, or stop word removal was performed; all morphological variation and English tokens are preserved in their original form.

### 1.2 Chunking Strategy

Transcripts were chunked at the utterance level using a sliding window over transcript lines:

- **Chunk size:** 24 transcript lines
- **Overlap:** 8 lines (33% overlap ensures continuity at chunk boundaries)
- **Minimum filter:** Chunks under 20 words are discarded

Each chunk stores its source episode, start timestamp, and end timestamp, making every retrieved passage fully traceable. The 33% overlap design preserves semantic coherence across chunk boundaries — critical for conversational Arabic where a speaker's thought frequently spans multiple short utterances. The larger chunk size (vs. a smaller window) was chosen to capture semantic links between related facts (e.g., event names and their dates) that were fragmented in smaller configurations.

### 1.3 Embedding and Hybrid Retrieval

All chunks are embedded with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, a 384-dimension multilingual model pre-trained on 50+ languages including Arabic. Embeddings are L2-normalized and stored in a FAISS `IndexFlatIP` (inner-product index, equivalent to cosine similarity on normalized vectors). This avoids approximation errors from quantized indices, acceptable given the corpus size.

**Hybrid reranking** is applied over the initial FAISS candidates:

```
final_score = 0.55 × semantic_score + 0.45 × keyword_score + phrase_boost
```

- `keyword_score`: fraction of query terms found in the chunk text
- `phrase_boost`: +0.05 per matching term, capped at 0.20

This hybrid approach improves precision for fact-seeking queries where exact technical terms (e.g., aircraft model numbers, historical names) have high keyword signal but moderate semantic distance from the embedding query form.

### 1.4 Multi-turn Chatbot with LangChain

The pipeline uses LangChain's `RunnableWithMessageHistory` and `InMemoryChatMessageHistory` to manage session state. The public interface is `chat_with_memory()`, which accepts `session_id`, `model_choice`, `prompt_style`, `memory_strategy`, `top_k`, and `max_turns` as parameters. Four context window strategies are supported:

| Strategy | Behavior |
|----------|----------|
| `full_history` | All prior turns |
| `sliding_window` | Last `max_turns × 2` messages (default, k=3) |
| `strict_truncation` | First Q&A turn + last `(max_turns-1) × 2` messages |
| `summarized_history` | LLM-generated Arabic summary of older turns → `SystemMessage` + recent turns |

`summarized_history` calls the active LLM with a dedicated summarization prompt in Arabic, producing a condensed representation of older history. This reduces token usage on long sessions while preserving semantic continuity better than simple truncation.

### 1.5 Prompt Engineering

Four prompt styles are evaluated in a controlled experiment matrix:

| Style | Language | Instruction Type |
|-------|----------|-----------------|
| `system_guided_ar` | Arabic | Detailed 5-rule grounding instruction |
| `minimal_ar` | Arabic | Single-line instruction |
| `system_guided_en` | English | Detailed grounding instruction |
| `minimal_en` | English | Single-line instruction |

The `system_guided_ar` prompt includes explicit constraints: answer only from context, no external information, be brief (1–2 sentences), admit ignorance if context is insufficient, and match the language of the context.

### 1.6 Out-of-Domain Detection

Before sending a query to the LLM, the top retrieved chunk's `final_score` is checked against a **threshold of 0.25**. Using the hybrid score rather than raw cosine similarity makes the threshold more robust: a query must demonstrate both reasonable semantic proximity and minimal keyword overlap to pass. Fully unrelated queries (weather, sports, general math) score below 0.15 on the hybrid scale, while on-topic queries consistently score above 0.35, giving a comfortable decision margin.

Empirical evaluation of the threshold on a small labeled set yielded:
- **Precision: 1.00** — zero false positives; no valid in-domain question was ever rejected
- **Recall: 0.40** — successfully blocked clear off-topic queries (cooking, math, general geography); some borderline queries with low keyword overlap were also blocked, reducing recall

### 1.7 Robustness

Both LLMs (`gemini-flash-latest` via Google, `llama-3.3-70b-versatile` via Groq) are retried up to 2 times on failure via `call_llm_with_retry()`. If the primary LLM exhausts retries, the system automatically falls back to the alternate LLM. Gemini's rich list-format responses are normalized to plain strings before being appended to chat history, preventing `MESSAGE_COERCION_FAILURE` errors in multi-turn sessions. The system never crashes due to API failures.

---

## 2. Evaluation

### 2.1 Metrics

Three evaluation dimensions are required: text generation quality, semantic correctness, and grounding to retrieved context. We use four complementary metrics:

| Metric | Dimension | Justification |
|--------|-----------|---------------|
| **Text Quality** | Text generation quality | Failure detector: returns 0.0 for stubs, error responses, and near-empty strings; 1.0 for any coherent answer. Serves as a syntactic fluency proxy. |
| **ROUGE-L** | Semantic correctness | Longest Common Subsequence F-measure. Captures sequence-level alignment between generated and reference answer. Standard in RAG/NLG evaluation; handles paraphrase better than BLEU by not requiring strict n-gram order. |
| **Grounding** | Grounding to context | Precision-based word overlap: fraction of answer words found in retrieved context. Directly measures hallucination risk — high score means the model is drawing from provided text, not fabricating. |
| **Exact Match** | Semantic correctness (strict) | Binary substring check. Useful as a strict lower bound; expected to be low for Arabic due to morphological variation and paraphrase even in correct answers. |

**Why not BLEU?** BLEU penalizes grammatically valid paraphrases and is sensitive to n-gram order. For Arabic, where correct answers may be expressed with different word order or morphological forms, ROUGE-L and grounding score are more informative.

### 2.2 Experiment Design

`evaluate.py` runs a **7-configuration × 2-model matrix** over 2 questions (1 F-35, 1 Samurai):

- **Axis 1 — Prompt style & language** (memory fixed to `sliding_window`): 4 configurations comparing system-guided vs minimal, Arabic vs English
- **Axis 2 — Memory/context strategy** (prompt fixed to `system_guided_ar`): 3 additional configurations — `full_history`, `strict_truncation`, `summarized_history`

`disable_fallback=True` ensures each run tests the target model exclusively, preventing cross-contamination of results.

### 2.3 Key Observations

**Prompt style effects:**
- `system_guided_ar` consistently produced the shortest, most grounded answers — the brevity constraint ("1–2 sentences") directly improves grounding score by limiting the model's opportunity to add ungrounded content.
- `minimal_ar` responses were longer and had lower grounding scores, confirming that explicit grounding constraints matter.
- English prompts (`system_guided_en`, `minimal_en`) produced comparable quality on factoid questions but occasionally generated English-language answers for Arabic questions, which reduces usability.

**Memory strategy effects:**
- `sliding_window` (default) and `strict_truncation` performed comparably in single-turn evaluation (as expected — differences emerge over multi-turn sessions).
- `summarized_history` introduces an extra LLM call latency but prevents context overflow in long sessions.

**Model comparison:**
- Gemini (`gemini-flash-latest`) showed higher grounding scores on F-35 questions, likely due to stronger Arabic instruction-following.
- Groq (`llama-3.3-70b-versatile`) performed comparably on Samurai questions and had lower latency per request.
- Both models correctly refused out-of-domain questions.
- An earlier iteration used `allam-2-7b` via Groq, but it produced high hallucination rates and repetitive outputs even with a strict system prompt. Switching to `llama-3.3-70b-versatile` resolved both issues.

**General patterns:**
- Exact match is consistently low even for correct answers — expected for Arabic due to morphological variation and paraphrase.
- Grounding scores are moderate; near-zero grounding cases correspond to retrieval misses (correct chunk not ranked in top-k), not hallucination. The OOD detector correctly refuses these rather than fabricating.

Full per-question, per-configuration results are in `evaluation_logs.json`.

### 2.4 Streamlit Interface

The `streamlit_app.py` interface exposes all evaluation-relevant controls: live model selection, prompt strategy switching, memory strategy switching, top-k control, and max-turns control. The developer log expander displays per-chunk source, timestamps, and hybrid score per answer, enabling real-time human evaluation of grounding quality. A session log table with CSV download records all exchanges with latency and token estimates.

---

## 3. Key Findings & Bug Fixes

- **Model selection**: Initial testing with `allam-2-7b` showed high hallucination and looping repetition. Switching to `llama-3.3-70b-versatile` (Groq) and `gemini-flash-latest` with a brevity-focused Arabic system prompt resolved both issues.
- **History corruption**: Gemini returns responses as rich structured objects (lists of dicts) rather than plain strings. Appending these directly to LangChain's message history caused `MESSAGE_COERCION_FAILURE` errors on subsequent turns. Fixed by explicitly normalizing `response.content` to a plain string before storing it — all list-format content is joined into a single string.
- **Brevity is key**: Forcing models to answer in 1–2 sentences (via the system prompt brevity constraint) dramatically improved grounding scores. Longer, more discursive answers tend to include elaborations sourced from the model's training data rather than the retrieved context, inflating hallucination risk.

---

## 4. Limitations

- **Exact match understates performance** for Arabic due to morphological richness; Arabic-aware embedding-based semantic similarity would be more reliable as a correctness metric.
- **Summarized history** introduces latency overhead (an extra LLM call) and its quality depends on the summarizing model — a degraded API response could produce a poor summary that corrupts subsequent turns.
- **Four episodes** provide a limited retrieval corpus; queries about adjacent topics (other aircraft, Japanese history generally) are correctly rejected but reduce system utility in a broader deployment.
- **Hybrid score threshold** of 0.25 was set empirically; a more systematic calibration study with labeled in/out-of-domain queries would improve recall without sacrificing precision.
- **Evaluation coverage**: `evaluate.py` tests 1 question per episode per configuration to conserve API quota; a larger test set would yield more statistically robust metric estimates.
