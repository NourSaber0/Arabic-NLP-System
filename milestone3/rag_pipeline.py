import os
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import both Google and Groq integrations
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables (API Keys)
load_dotenv()

class RAGChatbot:
    def __init__(self, index_path="vector_store/ms3_index.bin", metadata_path="vector_store/chunks_metadata.pkl"):
        print("Initializing RAG Chatbot with Dual LLMs...")
        
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.chunks_data = pickle.load(f)
            
        self.chat_history = []
        
# Initialize LLM 1: Gemini (Google)
        # Using 'gemini-1.5-flash-latest' or 'gemini-pro' usually resolves the v1beta routing error
        self.llm_gemini = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.3,
            max_retries=3
        )
        
        # Initialize LLM 2: Llama (Groq)
        # Using their newest, fully supported Versatile model
        self.llm_groq = ChatGroq(
            model_name="llama-3.3-70b-versatile", 
            temperature=0.3,
            max_retries=3
        )

    def retrieve_context(self, query, top_k=5):
        query_embedding = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks_data[idx]
            results.append({
                "score": float(score),
                "episode": chunk["episode"],
                "chunk_text": chunk["chunk_text"]
            })
        return results

    def get_context_window(self, strategy="sliding_window", k=4):
        if strategy == "full_history":
            return self.chat_history
        elif strategy == "sliding_window":
            return self.chat_history[-k:] if len(self.chat_history) > k else self.chat_history
        elif strategy == "strict_truncation":
            if len(self.chat_history) <= k: return self.chat_history
            return [self.chat_history[0]] + self.chat_history[-(k-1):]
        elif strategy == "summarized_history":
            return [SystemMessage(content="Previous chat was summarized: User asked about X.")] + self.chat_history[-2:]
        return self.chat_history

    def generate_response(self, query, window_strategy="sliding_window", model_choice="gemini"):
        retrieved_chunks = self.retrieve_context(query)
        
        # OOD Detection Threshold
        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0
        if top_score < 0.35: 
            ood_message = "عذراً، هذا السؤال خارج نطاق الحلقات المحددة. يرجى طرح سؤال متعلق بمحتوى الدحيح."
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=ood_message))
            return ood_message, []

        context_text = "\n\n".join([f"[{c['episode']}]\n{c['chunk_text']}" for c in retrieved_chunks])
        
        system_prompt = SystemMessage(content=f"""
        أنت مساعد ذكي يجيب على الأسئلة بناءً على السياق المسترجع فقط. 
        يجب أن تكون إجاباتك دقيقة ولا تخترع أي معلومات.
        
        السياق المتاح:
        {context_text}
        """)
        
        messages = [system_prompt] + self.get_context_window(window_strategy) + [HumanMessage(content=query)]
        
        # Select the LLM based on user choice
        try:
            if model_choice == "groq":
                response = self.llm_groq.invoke(messages).content
            else:
                response = self.llm_gemini.invoke(messages).content
        except Exception as e:
            # Fallback strategy if one API fails (Fulfills Section 2.8)
            print(f"⚠️ Primary LLM failed ({e}). Falling back to alternate...")
            fallback_llm = self.llm_gemini if model_choice == "groq" else self.llm_groq
            response = fallback_llm.invoke(messages).content

        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response))
        
        return response, retrieved_chunks
        
    def clear_memory(self):
        """Clears the conversation history for fresh testing."""
        self.chat_history = []

# Quick Test

if __name__ == "__main__":
    bot = RAGChatbot()
    print("\n--- Testing Gemini ---")
    answer, _ = bot.generate_response("لماذا تم بناء تاج محل؟", model_choice="gemini") # "Why was the Taj Mahal built?"
    print(f"Gemini: {answer}")
    
    bot.chat_history = [] # Clear memory
    
    print("\n--- Testing Groq ---")
    answer, _ = bot.generate_response("ما هي أسلحة الساموراي؟", model_choice="groq") # "What are the weapons of the Samurai?"
    print(f"Groq: {answer}")