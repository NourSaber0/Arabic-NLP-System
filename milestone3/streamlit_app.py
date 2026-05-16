
import streamlit as st
import pandas as pd
from rag_pipeline import chat_with_memory

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Arabic RAG Chatbot - MS3",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Arabic RAG Chatbot")
st.caption("Milestone 3: Retrieval-Augmented Generation over Arabic transcripts")

# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "evaluation_logs" not in st.session_state:
    st.session_state.evaluation_logs = []

# ==========================================
# SIDEBAR SETTINGS
# ==========================================

st.sidebar.header("Configuration Experiments")

model_choice = st.sidebar.selectbox(
    "LLM Provider Engine",
    ["gemini", "groq"],
    index=0,
    help="Select the primary LLM to generate answers."
)

prompt_style = st.sidebar.selectbox(
    "Prompt Strategy",
    [
        "system_guided_ar",
        "minimal_ar",
        "system_guided_en",
        "minimal_en"
    ],
    index=0,
    help="Choose the instruction style and language for the system prompt."
)

memory_strategy = st.sidebar.selectbox(
    "Memory Strategy",
    [
        "sliding_window",
        "full_history",
        "strict_truncation",
        "summarized_history"
    ],
    index=0,
    help="Define how the conversation history is managed in the context window."
)

st.sidebar.divider()
st.sidebar.header("Hyperparameters")

top_k = st.sidebar.slider(
    "Top-K Retrieved Chunks",
    min_value=1,
    max_value=8,
    value=5
)

max_turns = st.sidebar.slider(
    "Max Memory Turns",
    min_value=1,
    max_value=10,
    value=3
)

show_logs = st.sidebar.checkbox(
    "Show Developer Visibility Logs",
    value=True
)

# ==========================================
# DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# CHAT INPUT
# ==========================================

user_query = st.chat_input("Ask a question about the selected Arabic transcripts...")

if user_query:

    st.session_state.messages.append({
        "role": "user",
        "content": user_query
    })

    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):

        with st.spinner("Retrieving context and generating answer..."):

            result = chat_with_memory(
                user_query,
                top_k=top_k,
                max_turns=max_turns,
                memory_strategy=memory_strategy,
                model_choice=model_choice,
                prompt_style=prompt_style
            )

            answer = result["response"]

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # Track metrics for the session log
            st.session_state.evaluation_logs.append({
                "query": result["query"],
                "response": result["response"],
                "model_used": result.get("model_used"),
                "status": result.get("status", "success"),
                "prompt_style": prompt_style,
                "memory_strategy": memory_strategy,
                "latency": f"{result.get('latency', 0):.2f}s",
                "est_tokens": result.get("estimated_tokens", 0)
            })

            if show_logs:

                with st.expander("Developer Visibility Logs"):
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Used", result.get("model_used", "N/A"))
                    with col2:
                        st.metric("Latency", f"{result.get('latency', 0):.2f}s")
                    with col3:
                        st.metric("Est. Tokens", result.get("estimated_tokens", 0))

                    st.write("---")
                    
                    status = result.get("status", "success")
                    if status == "out_of_domain":
                        st.warning("⚠️ Query detected as Out-of-Domain (OOD)")
                    
                    retrieved_chunks = result.get("retrieved_chunks", [])
                    if retrieved_chunks:
                        best_score = retrieved_chunks[0].get("final_score", 0)
                        st.write(f"**Top Retrieval Score (OOD Confidence):** {best_score:.4f}")

                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"#### Source {i} (Score: {chunk['final_score']:.4f})")
                        st.caption(f"Episode: {chunk['episode']} | Time: {chunk['start_timestamp']} - {chunk['end_timestamp']}")
                        st.write(chunk["chunk_text"])

# ==========================================
# LOGS PANEL
# ==========================================

st.divider()

st.subheader("Engineering Performance Logs")

if st.session_state.evaluation_logs:
    logs_df = pd.DataFrame(st.session_state.evaluation_logs)
    st.dataframe(logs_df, use_container_width=True)

    csv = logs_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download engineering logs as CSV",
        data=csv,
        file_name="rag_performance_logs.csv",
        mime="text/csv"
    )
else:
    st.info("No logs yet. Start chatting to generate performance data.")
