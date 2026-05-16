import json
import numpy as np
from rag_pipeline import chat_with_memory, store

# ==========================================
# LOAD DATASET TEST SUITE
# ==========================================

def load_test_data(file_path, start=0, limit=5):
    """Load consecutive QA pairs from MS2 JSON dataset."""
    with open(file_path, "r", encoding="utf-8") as f:
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

def exact_match_score(predicted, expected):
    predicted = predicted.strip().lower()
    expected = expected.strip().lower()
    return 1.0 if expected in predicted else 0.0


def quality_score(answer):
    if not answer or len(answer.strip()) < 3:
        return 0.0

    bad_phrases = [
        "حدث خطأ",
        "لا أملك معلومات كافية",
        "I do not have enough information",
        "error"
    ]

    if any(phrase.lower() in answer.lower() for phrase in bad_phrases):
        return 0.0

    return 1.0


def grounding_score(answer, context):
    answer_words = set(answer.lower().split())
    context_words = set(context.lower().split())

    if not answer_words:
        return 0.0

    overlap = answer_words.intersection(context_words)
    return len(overlap) / len(answer_words)


def semantic_correctness_score(predicted, expected):
    predicted_words = set(predicted.lower().split())
    expected_words = set(expected.lower().split())

    if not expected_words:
        return 0.0

    overlap = predicted_words.intersection(expected_words)
    return len(overlap) / len(expected_words)


# ==========================================
# EXECUTION EVALUATION LOOP
# ==========================================

def run_evaluation():
    print("🚀 Starting Milestone 3 Multi-LLM Evaluation Pipeline...")

    # Load 5 questions per file (Total = 20 Test Scenarios)
    f35_qas = load_test_data("data/QA/f35_qa_dataset.json", start=0, limit=5)
    samurai_qas = load_test_data("data/QA/samurai_qa_dataset.json", start=0, limit=5)
    octopus_qas = load_test_data("data/QA/octopus_qa_dataset.json", start=0, limit=5)
    citizen_kane_qas = load_test_data("data/QA/citizen_kane_qa_dataset.json", start=0, limit=5)

    test_suite = f35_qas + samurai_qas + octopus_qas + citizen_kane_qas
    
    # Run evaluation sweeps over both mandatory model targets
    models_to_evaluate = ["gemini", "groq"]
    all_evaluation_results = {}

    for model in models_to_evaluate:
        print("\n" + "=" * 60)
        print(f"📡 RUNNING SWEEP FOR MODEL ENGINE: {model.upper()}")
        print("=" * 60)
        
        results = []
        
        for i, test in enumerate(test_suite, 1):
            question = test["question"]
            expected = test["expected"]
            context = test["context"]

            print(f"\nTest {i}/{len(test_suite)}: {question}")
            print(f"Expected Answer: {expected}")
            print("-" * 60)

            # Clear native LangChain memory store between individual isolated turns
            store.clear()

            try:
                # Call pipeline wrapper passing model flag explicitly
                output = chat_with_memory(
                    query=question,
                    top_k=8,
                    max_turns=3,
                    memory_strategy="sliding_window",
                    model_choice=model,
                    prompt_style="system_guided_ar",
                    session_id=f"eval_{model}_{i}"
                )

                answer = output["response"]
                model_used = output["model_used"]
                retrieved_context = output["context"]
                status = output["status"]

                exact = exact_match_score(answer, expected)
                semantic = semantic_correctness_score(answer, expected)
                grounding = grounding_score(answer, context)
                quality = quality_score(answer)

                result = {
                    "test_id": test["id"],
                    "question": question,
                    "expected": expected,
                    "model_target": model,
                    "model_used": model_used,
                    "status": status,
                    "answer": answer,
                    "retrieved_context": retrieved_context,
                    "exact_match": exact,
                    "semantic_correctness": semantic,
                    "grounding_score": grounding,
                    "quality_score": quality
                }

                results.append(result)

                print(f"Actual Model Used: {model_used}")
                print(f"Status: {status}")
                print(f"Answer: {answer}")
                print(f"Exact Match: {exact:.2f} | Semantic: {semantic:.2f} | Grounding: {grounding:.2f}")

            except Exception as e:
                print(f"❌ Error during execution: {e}")

            print("-" * 40)
            
        all_evaluation_results[model] = results

    # Consolidate log payload files
    flat_results_list = all_evaluation_results["gemini"] + all_evaluation_results["groq"]
    save_results(flat_results_list)
    
    # Print unified report performance summary
    print_summary(all_evaluation_results)


def save_results(results):
    with open("evaluation_logs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n✅ Evaluation log configurations saved to evaluation_logs.json")


def print_summary(all_results):
    print("\n📊 MULTI-LLM EVALUATION METRICS REPORT")
    print("=" * 60)

    for model, results in all_results.items():
        print(f"\n📈 Performance Summary for: {model.upper()} ({len(results)} queries evaluated)")
        print("-" * 60)
        if not results:
            print("No data collected.")
            continue

        avg_exact = np.mean([r["exact_match"] for r in results])
        avg_semantic = np.mean([r["semantic_correctness"] for r in results])
        avg_grounding = np.mean([r["grounding_score"] for r in results])
        avg_quality = np.mean([r["quality_score"] for r in results])

        print(f"  🔹 Average Exact Match:         {avg_exact:.2f}")
        print(f"  🔹 Average Semantic Correctness: {avg_semantic:.2f}")
        print(f"  🔹 Average Grounding Score:     {avg_grounding:.2f}")
        print(f"  🔹 Average Quality Score:        {avg_quality:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()