import json
from rag_pipeline import RAGChatbot

def load_test_data(file_path):
    """Parses the MS2 JSON files to extract questions and expected answers."""
    with open(file_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
        
    qa_pairs = []
    for data in dataset['data']:
        for paragraph in data['paragraphs']:
            for qa in paragraph['qas']:
                qa_pairs.append({
                    "id": qa['id'],
                    "question": qa['question'],
                    "expected": qa['answers'][0]['text']
                })
    return qa_pairs

def run_evaluation():
    print("🚀 Starting Evaluation Pipeline...")
    bot = RAGChatbot()
    
    # Load 3 questions from each dataset just for a quick test
    taj_mahal_qas = load_test_data("data/QA/taj_mahal_qa_dataset.json")[:3]
    samurai_qas = load_test_data("data/QA/samurai_qa_dataset.json")[:3]
    test_suite = taj_mahal_qas + samurai_qas

    print(f"\nLoaded {len(test_suite)} questions for testing.\n" + "="*50)

    for i, test in enumerate(test_suite, 1):
        question = test['question']
        expected = test['expected']
        
        print(f"\nTest {i}: {question}")
        print(f"Expected Answer: {expected}")
        print("-" * 30)
        
        # Test Gemini
        bot.clear_memory()
        gemini_ans, _ = bot.generate_response(question, model_choice="gemini")
        print(f"🤖 Gemini: {gemini_ans}")
        
        # Test Groq
        bot.clear_memory()
        groq_ans, _ = bot.generate_response(question, model_choice="groq")
        print(f"⚡ Groq: {groq_ans}")
        print("=" * 50)

if __name__ == "__main__":
    # Make sure the paths match where you saved the files
    run_evaluation()