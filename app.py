import streamlit as st
from rag_pipeline import RAGAssistant

st.set_page_config(page_title="Document-Aware AI Assistant (RAG)")

st.title("📄 Document-Aware AI Assistant (RAG)")

# -------------------------------
# Persist RAG object across reruns
# -------------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGAssistant()

rag = st.session_state.rag

# -------------------------------
# Document ingestion
# -------------------------------
doc_text = st.text_area(
    "Paste document text:",
    height=200
)

if st.button("Index Document"):
    if doc_text.strip():
        rag.ingest_text(doc_text)
        st.success("Document indexed successfully!")
    else:
        st.warning("Please paste some document text.")

# -------------------------------
# Question answering
# -------------------------------
question = st.text_input("Ask a question:")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        answer = rag.ask(question)
        st.subheader("Answer")
        st.write(answer)