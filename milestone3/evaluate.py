import json
import numpy as np
import time
import os
from pathlib import Path
from rag_pipeline import chat_with_memory, store

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
QA_DIR = BASE_DIR / "data" / "QA"

# ==========================================
# LOAD DATASET TEST SUITE
# ==========================================

def load_test_data(file_path, start=0, limit=3):
    """Load consecutive QA pairs from MS2 JSON dataset."""
    full_path = QA_DIR / file_path
    with open(full_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    qa_pairs = []
    for data in dataset["data"]:
        for paragraph in data["paragraphs"]:
            for qa in paragraph["qas"]:
                qa_pairs.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "expected": qa["answers"][0]["text"],
                    "context": paragraph["context"]
                })

    return qa_pairs[start:start + limit]


# ==========================================
# EVALUATION METRICS SCRIPTS
# ==========================================

import re
from rouge_score import rouge_scorer

def normalize_for_metric(text):
    """Normalize text for better metric matching."""
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_exact_match(predicted, expected):
    """Simple binary check for exact string match."""
    predicted = normalize_for_metric(predicted)
    expected = normalize_for_metric(expected)
    return 1.0 if expected in predicted else 0.0


def calculate_text_quality(answer):
    """
    CATEGORY: Text Generation Quality
    JUSTIFICATION: Measures fluency by detecting rejection stubs and error messages.
    While a simple failure detector, it serves as a proxy for 'Syntactic Coherence'
    by ensuring the model doesn't return non-sentences or known failure patterns.
    """
    if not answer or len(answer.strip()) < 5:
        return 0.0

    failure_patterns = [
        "حدث خطأ",
        "لا أملك معلومات كافية",
        "I do not have enough information",
        "error",
        "عذرًا، هذا السؤال خارج نطاق",
        "sorry, this question is outside the scope"
    ]

    for pattern in failure_patterns:
        if pattern.lower() in answer.lower():
            return 0.0

    if len(answer.split()) < 2:
        return 0.0

    return 1.0


def calculate_grounding(answer, context):
    """
    CATEGORY: Grounding to Retrieved Context (Faithfulness)
    JUSTIFICATION: Precision-based word overlap. We check what percentage of 
    the model's claims (words) can be found in the provided context. 
    A higher score indicates lower hallucination risk.
    """
    answer = normalize_for_metric(answer)
    context = normalize_for_metric(context)
    
    answer_words = set(answer.split())
    context_words = set(context.split())

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)


def calculate_semantic_correctness(predicted, expected):
    """
    CATEGORY: Semantic Correctness
    JUSTIFICATION: Uses ROUGE-L (Longest Common Subsequence) to measure 
    how well the generated answer captures the sequence and content of 
    the ground truth answer. ROUGE is a standard NLP metric for RAG evaluation.
    """
    predicted = normalize_for_metric(predicted)
    expected = normalize_for_metric(expected)
    
    if not expected:
        return 1.0
        
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = scorer.score(expected, predicted)
    return scores['rougeL'].fmeasure


# ==========================================
# EXECUTION EVALUATION LOOP
# ==========================================

def run_evaluation():
    print("🚀 Starting Milestone 3 Multi-LLM Evaluation Pipeline...")

    # Load questions per file
    try:
        # Load only 2 questions total (1 from each of 2 files)
        f35_qas = load_test_data("f35_qa_dataset.json", start=0, limit=1)
        samurai_qas = load_test_data("samurai_qa_dataset.json", start=0, limit=1)
    except FileNotFoundError as e:
        print(f"❌ Error: QA dataset files not found. {e}")
        return

    test_suite = f35_qas + samurai_qas
    
    # Focused Comparative Configurations (Fixing variables to isolate effects)
    # This reduces 16 combinations to 7 unique ones to save your API resources.
    configurations = [
        # Comparison 1 & 2: Prompt Style & Language (Memory fixed to sliding_window)
        {"prompt_style": "system_guided_ar", "memory_strategy": "sliding_window", "comp": "Prompt/Lang"},
        {"prompt_style": "minimal_ar",       "memory_strategy": "sliding_window", "comp": "Prompt/Lang"},
        {"prompt_style": "system_guided_en", "memory_strategy": "sliding_window", "comp": "Prompt/Lang"},
        {"prompt_style": "minimal_en",       "memory_strategy": "sliding_window", "comp": "Prompt/Lang"},
        
        # Comparison 3: Context Strategies (Prompt fixed to system_guided_ar)
        {"prompt_style": "system_guided_ar", "memory_strategy": "full_history",      "comp": "Context/Mem"},
        {"prompt_style": "system_guided_ar", "memory_strategy": "strict_truncation",  "comp": "Context/Mem"},
        {"prompt_style": "system_guided_ar", "memory_strategy": "summarized_history", "comp": "Context/Mem"},
    ]

    models_to_evaluate = ["gemini", "groq"]
    flat_results_list = []

    for model in models_to_evaluate:
        for config in configurations:
            p_style = config["prompt_style"]
            m_strat = config["memory_strategy"]
            
            print("\n" + "=" * 60)
            print(f"📡 EVALUATING: {model.upper()} | Prompt: {p_style} | Memory: {m_strat}")
            print("=" * 60)
            
            for i, test in enumerate(test_suite, 1):
                question = test["question"]
                expected = test["expected"]
                context = test["context"]

                print(f"\nTest {i}/{len(test_suite)}: {question}")

                # Clear history for clean independent turn evaluation
                store.clear()

                try:
                    # disable_fallback=True ensures we test the TARGET model.
                    output = chat_with_memory(
                        query=question,
                        top_k=5,
                        max_turns=3,
                        memory_strategy=m_strat,
                        model_choice=model,
                        prompt_style=p_style,
                        session_id=f"eval_{model}_{p_style}_{m_strat}_{i}",
                        disable_fallback=True 
                    )

                    answer = output["response"]
                    status = output["status"]
                    model_actually_used = output.get("model_used")

                    # Calculate the 3 Mandatory Metrics
                    quality = calculate_text_quality(answer)
                    semantic = calculate_semantic_correctness(answer, expected)
                    grounding = calculate_grounding(answer, context)
                    
                    # Also keep Exact Match for reference
                    exact = calculate_exact_match(answer, expected)

                    result = {
                        "test_id": test["id"],
                        "question": question,
                        "expected": expected,
                        "model_target": model,
                        "model_used": model_actually_used,
                        "prompt_style": p_style,           
                        "memory_strategy": m_strat,        
                        "status": status,
                        "answer": answer,
                        "metrics": {
                            "text_quality": quality,
                            "semantic_correctness": semantic,
                            "grounding": grounding,
                            "exact_match": exact
                        },
                        "latency": output.get("latency", 0)
                    }

                    flat_results_list.append(result)
                    print(f"Answer: {answer}")
                    print(f"Result: {status} | Quality: {quality:.2f} | Semantic: {semantic:.2f} | Grounding: {grounding:.2f} | Exact: {exact:.2f}")

                    # Incremental save
                    save_results(flat_results_list)

                except Exception as e:
                    print(f"❌ {model.upper()} FAILED: {str(e)}")
                    flat_results_list.append({
                        "test_id": test.get("id", "N/A"),
                        "question": question,
                        "model_target": model,
                        "status": "error",
                        "error_message": str(e)
                    })
                
                # Small sleep to respect rate limits
                time.sleep(2)

    save_results(flat_results_list)
    print_summary(flat_results_list)


def save_results(results):
    with open("evaluation_logs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n✅ Evaluation logs saved to evaluation_logs.json")


def print_summary(results):
    print("\n📊 EVALUATION SUMMARY (By Category)")
    print("=" * 80)
    models = sorted(list(set(r["model_target"] for r in results)))
    for model in models:
        m_results = [r for r in results if r["model_target"] == model and r["status"] == "success"]
        errors = len([r for r in results if r["model_target"] == model and r["status"] == "error"])
        
        print(f"\n📈 {model.upper()}: {len(m_results)} successes, {errors} errors")
        if m_results:
            avg_quality = np.mean([r["metrics"]["text_quality"] for r in m_results])
            avg_semantic = np.mean([r["metrics"]["semantic_correctness"] for r in m_results])
            avg_grounding = np.mean([r["metrics"]["grounding"] for r in m_results])
            avg_exact = np.mean([r["metrics"]["exact_match"] for r in m_results])
            
            print(f"  🔹 Text Generation Quality: {avg_quality:.2f}")
            print(f"  🔹 Semantic Correctness:    {avg_semantic:.2f}")
            print(f"  🔹 Grounding (Faithfulness): {avg_grounding:.2f}")
            print(f"  🔹 (Ref) Avg Exact Match:   {avg_exact:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
