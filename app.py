import streamlit as st
import os
from rag_pipeline import run_rag_bot

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("📄 RAG-based Knowledge Retrieval Chatbot")

# Load default PDF from /sample_docs if no upload
DEFAULT_FILE = "sample_docs/insurance_policy.pdf"

uploaded_file = st.file_uploader("Upload a PDF document (optional)", type="pdf")

if uploaded_file:
    save_path = os.path.join("sample_docs", uploaded_file.name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File uploaded successfully! Using: {uploaded_file.name}")
else:
    save_path = DEFAULT_FILE
    st.info(f"📂 No file uploaded — using default document: `{os.path.basename(DEFAULT_FILE)}`")

query = st.text_input("Ask a question based on the document:")

if query:
    with st.spinner("🤖 Gemini is thinking..."):
        try:
            answer = run_rag_bot(save_path, query)
            st.markdown("### 🤖 Answer:")
            st.success(answer)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
