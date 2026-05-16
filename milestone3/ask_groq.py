from rag_pipeline import chat_with_memory
import sys

print("🤖 Groq RAG Interactive CLI")
print("--------------------------")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    try:
        user_input = input("❓ Question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        if not user_input.strip():
            continue

        print("🔍 Searching and generating...")
        # explicitly force model_choice="groq"
        output = chat_with_memory(
            query=user_input, 
            model_choice="groq",
            disable_fallback=True
        )
        
        print(f"\n📝 [Groq Answer]:\n{output['response']}")
        print(f"\n(Status: {output['status']} | Model: {output.get('model_used')})")
        print("-" * 30)
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Error: {e}")

print("\nGoodbye!")
