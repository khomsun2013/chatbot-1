import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.llms import OpenAI, Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
import requests
import openai
import ollama
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

load_dotenv()
# --- Config ---
CHROMA_DB_DIR = "./chroma_db"
#DATA_FILE = "class_regulation_qa_dataset.json"
DATA_FILE = "cs5600_qa_dataset_final.json"


gemini_api_key = os.getenv("GOOGLE_API_KEY")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

os.environ["STREAMLIT_WATCHER_SKIP_PACKAGES"] = "torch"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Load JSON data and initialize Chroma DB ---
def init_chroma():
    if not os.path.exists(CHROMA_DB_DIR):
        os.makedirs(CHROMA_DB_DIR)

    loader = JSONLoader(
        file_path=DATA_FILE,
        jq_schema=".[] | {page_content: .answer, metadata: {question: .question}}",
        text_content=False,
    )
    documents = loader.load()

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #EMBEDDING_MODEL = "nomic-embed-text"
    #ollama.pull(EMBEDDING_MODEL)
    #embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    #vectordb.persist()
    return vectordb

# --- Streamlit App ---
st.set_page_config(page_title="Class Q&A Assistant", layout="wide")
st.title("Class Q&A Assistant")

llm_choice = st.selectbox("Choose LLM", ["OpenAI", "Ollama", "Gemini"])
user_query = st.text_input("Ask a question about the class:", placeholder="e.g., What is the instructor's office hours?")

if "vectordb" not in st.session_state:
    with st.spinner("Setting up vector database..."):
        st.session_state.vectordb = init_chroma()

if user_query:
    with st.spinner("Generating answer..."):
        if llm_choice == "OpenAI":
            llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.7,api_key=api_key,base_url=base_url)  # Create a client instance
        elif llm_choice == "Ollama":
            llm = OllamaLLM(model="mistral")
            #llm = OllamaLLM(model="gpt-oss:20b")
        else:  # Gemini
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectordb.as_retriever()
        )
        response = qa.invoke(user_query)
        st.success(response['result'])
        

# Optional: show retrieved documents
with st.expander("🔍 Show retrieved questions"):
    docs = st.session_state.vectordb.similarity_search(user_query, k=2) if user_query else []
    for i, doc in enumerate(docs):
        #st.write(f"**Q{i+1}:** {doc.metadata.question}")
        st.write(f"> {doc.page_content}")
