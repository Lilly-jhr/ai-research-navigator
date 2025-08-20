
import os
import streamlit as st
from dotenv import load_dotenv

# LangChain Core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Retrievers & Loaders
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever

# Reranker, Embeddings, LLM
from langchain_cohere import CohereRerank
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

COLLECTIONS = {
    "Large Language Models": "llm_db",
    "Computer Vision": "cv_db"
}

# --- Caching Functions for RAW DATA ONLY ---

@st.cache_resource
def load_embeddings():
    """Load and cache the HuggingFace embeddings model. This is safe."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_documents(_collection_name):
    """Load and cache the raw documents for a specific collection."""
    data_path = f"data/{_collection_name.replace('_db', '_papers')}"
    loader = PyPDFDirectoryLoader(data_path)
    return loader.load()

# --- Main Application ---

st.set_page_config(page_title="AI Research Navigator", page_icon="🔬")
st.title("🔬 AI Research Navigator")

# --- Sidebar and Settings ---
st.sidebar.title("Settings")
selected_collection_name = st.sidebar.selectbox(
    "Choose a Research Collection:",
    options=list(COLLECTIONS.keys())
)
selected_collection_id = COLLECTIONS[selected_collection_name]

selected_mode = st.sidebar.radio(
    "Choose a mode:",
    options=["Question Answering", "Summarization"]
)



embeddings = load_embeddings()
docs = load_documents(selected_collection_id)

vectorstore_path = f"vectorstores/{selected_collection_id}"
faiss_index = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 5

ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
reranker = CohereRerank(cohere_api_key=COHERE_API_KEY, model="rerank-english-v3.0", top_n=3)
retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=ensemble_retriever)


llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

if selected_mode == "Question Answering":
    system_prompt = (
        "You are an expert AI Research Assistant. Your task is to answer user questions based ONLY on the provided context."
        "Be concise and precise. If the context does not contain the answer, state that you cannot find the answer."
        "Consider the chat history for follow-up questions."
        "\n\n"
        "Context:\n{context}"
    )
    input_placeholder = "Ask a question about {selected_collection_name}..."
else: # Summarization Mode
    system_prompt = (
        "You are an expert AI Research Summarizer. Your task is to provide a concise summary of the key points from the provided context."
        "Focus on the main contributions, methods, and results. Do not add information not present in the text."
        "\n\n"
        "Context:\n{context}"
    )
    input_placeholder = "Enter a topic or paper title to summarize..."


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


chat_history_key = f"chat_history_{selected_collection_id}_{selected_mode}"
if chat_history_key not in st.session_state:
    st.session_state[chat_history_key] = []

for message in st.session_state[chat_history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(input_placeholder):
    st.session_state[chat_history_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    formatted_chat_history = []
    for msg in st.session_state[chat_history_key]:
        if msg["role"] == "user":
            formatted_chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            formatted_chat_history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag_chain.invoke({
                "input": prompt,
                "chat_history": formatted_chat_history
            })
            response = result["answer"]
            st.markdown(response)
            
            sources = [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in result["context"]]
            unique_sources = sorted(list(set(sources)))
            if unique_sources:
                with st.expander("📚 Sources"):
                    for source in unique_sources:
                        st.info(source)

    st.session_state[chat_history_key].append({"role": "assistant", "content": response})