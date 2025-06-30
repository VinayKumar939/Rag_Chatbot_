import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# ✅ Configure Gemini API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Load PDF Documents
def load_documents(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()

# ✅ Split & Embed Documents
def create_vector_store(documents):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# ✅ Main RAG Bot
def run_rag_bot(pdf_path, query):
    documents = load_documents(pdf_path)
    vector_store = create_vector_store(documents)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.get_relevant_documents(query)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the user's question.

        Context:
        {context}

        Question: {question}
        Answer:"""
    )

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",  # ✅ Use exact model from `check_models.py`
        temperature=0.2
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"context": context, "question": query})
    return response
