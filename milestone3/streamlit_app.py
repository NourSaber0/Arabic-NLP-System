
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

st.sidebar.header("Settings")

top_k = st.sidebar.slider(
    "Top-K Retrieved Chunks",
    min_value=1,
    max_value=8,
    value=5
)

memory_strategy = st.sidebar.selectbox(
    "Memory Strategy",
    [
        "sliding_window",
        "full_history",
        "strict_truncation",
        "summarized_history"
    ]
)

max_turns = st.sidebar.slider(
    "Max Memory Turns",
    min_value=1,
    max_value=10,
    value=3
)

show_logs = st.sidebar.checkbox(
    "Show Retrieval Logs",
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
                memory_strategy=memory_strategy
            )

            answer = result["response"]

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.session_state.evaluation_logs.append({
                "query": result["query"],
                "response": result["response"],
                "model_used": result.get("model_used"),
                "status": result.get("status", "success"),
                "memory_strategy": memory_strategy,
                "top_k": top_k
            })

            if show_logs:

                with st.expander("Retrieved Context / Logs"):

                    st.write("Model used:", result.get("model_used"))
                    st.write("Status:", result.get("status", "success"))
                    st.write("Memory strategy:", memory_strategy)

                    retrieved_chunks = result.get("retrieved_chunks", [])

                    for i, chunk in enumerate(retrieved_chunks, start=1):
                        st.markdown(f"### Source {i}")
                        st.write("Episode:", chunk["episode"])
                        st.write(
                            "Time:",
                            chunk["start_timestamp"],
                            "→",
                            chunk["end_timestamp"]
                        )
                        st.write("Final score:", round(chunk["final_score"], 4))
                        st.write(chunk["chunk_text"])

# ==========================================
# LOGS PANEL
# ==========================================

st.divider()

st.subheader("Session Logs")

if st.session_state.evaluation_logs:
    logs_df = pd.DataFrame(st.session_state.evaluation_logs)
    st.dataframe(logs_df, use_container_width=True)

    csv = logs_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download session logs as CSV",
        data=csv,
        file_name="streamlit_session_logs.csv",
        mime="text/csv"
    )
else:
    st.info("No logs yet. Start chatting to generate logs.")
