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
    Evaluates fluency and syntactic coherence by checking for 
    error patterns, minimum length, and presence of placeholders.
    """
    if not answer or len(answer.strip()) < 5:
        return 0.0

    # Common error or refusal patterns
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

    # Basic fluency check: ensure we don't just have one or two words
    if len(answer.split()) < 2:
        return 0.0

    return 1.0


def calculate_grounding(answer, context):
    """
    CATEGORY: Grounding to Retrieved Context (Faithfulness)
    Validates that the generated answer is supported by the 
    provided context using word overlap as a proxy for faithfulness.
    """
    answer = normalize_for_metric(answer)
    context = normalize_for_metric(context)
    
    answer_words = set(answer.split())
    context_words = set(context.split())

    if not answer_words:
        return 0.0

    # Proportion of answer words found in context
    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)


def calculate_semantic_correctness(predicted, expected):
    """
    CATEGORY: Semantic Correctness
    Measures how accurately the answer addresses the user's intent 
    compared to the ground-truth expected answer.
    """
    predicted = normalize_for_metric(predicted)
    expected = normalize_for_metric(expected)
    
    predicted_words = set(predicted.split())
    expected_words = set(expected.split())

    if not expected_words:
        return 1.0 # Empty expected is vacuously correct

    # Recall-oriented overlap (how much of 'expected' did we capture?)
    overlap = predicted_words.intersection(expected_words)
    return len(overlap) / len(expected_words)


# ==========================================
# EXECUTION EVALUATION LOOP
# ==========================================

def run_evaluation():
    print("🚀 Starting Milestone 3 Multi-LLM Evaluation Pipeline...")

    # Load questions per file
    try:
        f35_qas = load_test_data("f35_qa_dataset.json", start=0, limit=3)
        samurai_qas = load_test_data("samurai_qa_dataset.json", start=0, limit=3)
        octopus_qas = load_test_data("octopus_qa_dataset.json", start=0, limit=3)
        citizen_kane_qas = load_test_data("citizen_kane_qa_dataset.json", start=0, limit=3)
    except FileNotFoundError as e:
        print(f"❌ Error: QA dataset files not found. {e}")
        return

    test_suite = f35_qas + samurai_qas + octopus_qas + citizen_kane_qas
    
    configurations = [
        {"prompt_style": "system_guided_ar", "memory_strategy": "sliding_window"},
        {"prompt_style": "minimal_ar", "memory_strategy": "sliding_window"},
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
