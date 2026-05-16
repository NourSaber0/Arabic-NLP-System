from rag_pipeline import chat_with_memory, store
import json

questions = [
    "ما اسم الطائرة التي تدور حولها الحلقة؟", # F-35 (Easy)
    "ما الحدث البيولوجي الذي يحاول البحث تفسيره؟", # Cambrian (Medium)
    "كيف يهرب الأخطبوط؟", # Octopus (Medium)
    "من هو أورسون ويلز؟", # Citizen Kane (Harder/Generic)
    "في أي سنة حدث الانفجار الكامبري؟" # Fact-based
]

print("🚀 Running Groq Responsiveness Test...")
for i, q in enumerate(questions, 1):
    store.clear()
    print(f"\nQuestion {i}: {q}")
    output = chat_with_memory(query=q, model_choice="groq", disable_fallback=True)
    print(f"  Status: {output['status']}")
    print(f"  Model:  {output.get('model_used')}")
    print(f"  Answer: {output['response']}")
