import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import (
    MetadataFilters, 
    ExactMatchFilter, 
    FilterCondition
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
import chromadb

# --- 1. Load Secrets and Initialize Models ---
load_dotenv()
# (The OPENAI_API_KEY is now loaded from .env)

@st.cache_resource
def load_models():
    """Load and cache LLM, embed model, and index."""
    llm = OpenAI(model="gpt-4o", temperature=0)
    
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    
    # Load the index from the same persistent directory
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_collection("ticket_knowledge")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index, llm

index, llm = load_models()

# --- 2. Define the Chatbot Logic ---

# Define the prompt
qa_prompt_template = (
    "You are a helpful IT support assistant. Your goal is to answer the user's question "
    "using the provided context from past support tickets. Synthesize a helpful, direct "
    "answer. If the context provides a solution, provide that solution.\n\n"
    "--- Context from Past Solutions ---\n"
    "{context_str}\n"
    "-----------------------------------\n\n"
    "--- User's Question ---\n"
    "{query_str}\n\n"
    "--- Your Answer ---\n"
)
qa_prompt = PromptTemplate(qa_prompt_template)

def get_query_engine(language, type):
    """Build a query engine with the correct filters."""
    
    # Define filters
    working_filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="language", value=language),
            ExactMatchFilter(key="type", value=type)
        ],
        condition=FilterCondition.AND
    )
    
    # Define retriever
    retriever_working = VectorIndexRetriever(
        index=index,
        similarity_top_k=3,
        filters=working_filters,
        similarity_cutoff=0.35
    )
    
    # Define query engine
    query_engine = RetrieverQueryEngine.from_args(
        retriever_working,
        llm=llm,
        text_qa_template=qa_prompt
    )
    return query_engine

# --- 3. Build the Streamlit UI ---
st.title("Hello I am Vega 🤖")
st.write("What can I assist you with today?")

# Add sidebar filters
st.sidebar.header("Query Filters")
lang = st.sidebar.selectbox("Language", ("en", "de"))
ttype = st.sidebar.selectbox("Ticket Type", ("Problem", "Request", "Change", "Question"))

# Create a unique query engine based on the user's filter selection
query_engine = get_query_engine(lang, ttype)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            st.markdown(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})