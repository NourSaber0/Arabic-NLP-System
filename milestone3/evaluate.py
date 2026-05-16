import json
import numpy as np
from rag_pipeline import chat_with_memory, conversation_history


def load_test_data(file_path, start=0, limit=2):
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


def exact_match_score(predicted, expected):
    predicted = predicted.strip().lower()
    expected = expected.strip().lower()
    return 1.0 if expected in predicted else 0.0


def quality_score(answer):
    if not answer or len(answer.strip()) < 10:
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


def run_evaluation():
    print("🚀 Starting Milestone 3 Evaluation Pipeline...")

    f35_qas = load_test_data("data/QA/f35_qa_dataset.json", start=0, limit=10)
    samurai_qas = load_test_data("data/QA/samurai_qa_dataset.json", start=0, limit=10)

    test_suite = f35_qas + samurai_qas
    results = []

    print(f"\nLoaded {len(test_suite)} consecutive questions for testing.")
    print("=" * 60)

    for i, test in enumerate(test_suite, 1):
        question = test["question"]
        expected = test["expected"]
        context = test["context"]

        print(f"\nTest {i}: {question}")
        print(f"Expected Answer: {expected}")
        print("-" * 60)

        conversation_history.clear()

        try:
            output = chat_with_memory(
                question,
                top_k=5,
                max_turns=3,
                memory_strategy="sliding_window"
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

            print(f"Model Used: {model_used}")
            print(f"Status: {status}")
            print(f"Answer: {answer}")
            print(f"Exact Match: {exact:.2f}")
            print(f"Semantic Correctness: {semantic:.2f}")
            print(f"Grounding Score: {grounding:.2f}")
            print(f"Quality Score: {quality:.2f}")

        except Exception as e:
            print(f"Error: {e}")

        print("=" * 60)

    save_results(results)
    print_summary(results)


def save_results(results):
    with open("evaluation_logs.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nEvaluation logs saved to evaluation_logs.json")


def print_summary(results):
    print("\n📊 Evaluation Summary")
    print("=" * 60)

    if not results:
        print("No successful evaluation results.")
        return

    avg_exact = np.mean([r["exact_match"] for r in results])
    avg_semantic = np.mean([r["semantic_correctness"] for r in results])
    avg_grounding = np.mean([r["grounding_score"] for r in results])
    avg_quality = np.mean([r["quality_score"] for r in results])

    print(f"Average Exact Match: {avg_exact:.2f}")
    print(f"Average Semantic Correctness: {avg_semantic:.2f}")
    print(f"Average Grounding Score: {avg_grounding:.2f}")
    print(f"Average Quality Score: {avg_quality:.2f}")


if __name__ == "__main__":
    run_evaluation()