# Arabic NLP System: RAG Chatbot 🗣️

Welcome to the **Arabic NLP System** repository! This project implements a **Multi-Turn Retrieval-Augmented Generation (RAG) Chatbot** designed to answer questions strictly based on retrieved context from Arabic podcast and audio transcripts. 

The system leverages advanced NLP techniques to preserve dialectal variation and Arabic-English code-switching, ensuring grounded generation without hallucination.

---

## ✨ Features

- **Multi-Turn Conversational Memory:** Supports Full-History, Sliding Window, Strict Truncation, and Summarized-History memory strategies.
- **Dialect & Code-Switching Support:** Handles Modern Standard Arabic (MSA), Egyptian dialect (e.g., *El Daheeh* episodes), and mixed Arabic-English queries.
- **Grounded Generation:** Enforces strict adherence to retrieved text; gracefully refuses to answer out-of-domain (OOD) queries.
- **Model Fallback Mechanism:** Automatically switches LLM providers (e.g., from Groq to Google Gemini) if rate limits or API outages occur.
- **Interactive UI:** A complete Streamlit web application allowing users to tweak configurations and view backend metrics in real-time.

---

## 🛠️ Architecture & Tech Stack

- **Framework:** LangChain 🦜🔗
- **Embeddings:** HuggingFace `SentenceTransformers`
- **Vector Store:** FAISS
- **LLMs:** Google Gemini (`gemini-flash-latest`), Groq (Llama 3)
- **Frontend:** Streamlit

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/NourSaber0/Arabic-NLP-System.git
cd Arabic-NLP-System
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the `milestone3` directory (or root) and add your API keys:
```bash
# milestone3/.env
GOOGLE_API_KEY="your-google-api-key"
GROQ_API_KEY="your-groq-api-key"
```

### 5. Run the Application
Navigate to the `milestone3` directory and start the Streamlit app:
```bash
cd milestone3
streamlit run streamlit_app.py
```
You can also run the CLI interface:
```bash
python3 ask_groq.py
```

---

## 📂 Repository Structure

- `milestone1.ipynb`: Data collection, normalization, and exploratory data analysis.
- `MS1_Report.md`: Milestone 1 detailed report.
- `milestone3/`: Core RAG pipeline implementation.
  - `rag_pipeline.py`: The backbone LangChain RAG logic.
  - `streamlit_app.py`: The frontend UI application.
  - `ask_groq.py`: A simple CLI conversational interface.
  - `evaluate.py`: Evaluation scripts for RAG metrics (ROUGE, correctness, faithfulness).
- `Transcripts/` & `QA/`: Raw data and QA pairs used for evaluating the system.
- `GEMINI.md`: Full architectural blueprint and system specification.

---

## 📊 Evaluation & Metrics
The system is rigorously evaluated on:
1. **Text Generation Quality** (Fluency and coherence)
2. **Semantic Correctness** (Accuracy of the answer)
3. **Faithfulness** (Grounding to retrieved context)

Results are logged in `evaluation_logs.json`.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.
