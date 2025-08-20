# 🔬 AI Research Navigator

**A Retrieval-Augmented Generation (RAG) chatbot for exploring and understanding complex AI research papers.**

---

### **[Live Demo Link Here]** 👈 *(Paste your Hugging Face Spaces URL here)*

---

## 📝 Description

The AI Research Navigator is an advanced, conversational AI tool designed to help users navigate dense technical literature. Users can select from different collections of research papers (e.g., Large Language Models, Computer Vision), ask specific questions, and request summaries on various topics.

The application leverages a sophisticated RAG pipeline, including hybrid search and a reranker, to provide accurate, fast, and contextually-grounded answers with source citations.

## ✨ Features

- **Conversational Q&A:** Ask questions in natural language and get precise answers based on the text.
- **Hybrid Search:** Combines semantic (FAISS) and keyword (BM25) search for robust and relevant document retrieval.
- **High-Quality Reranking:** Uses Cohere's Rerank model to refine search results and improve answer quality.
- **Switchable Knowledge Bases:** Seamlessly switch between different collections of research papers (e.g., LLMs vs. Computer Vision).
- **Summarization Mode:** Ask the chatbot to summarize key concepts, papers, or topics.
- **Fast & Interactive UI:** Built with Streamlit and powered by the high-speed Groq API for near-instant responses.
- **Source Citation:** Every answer is accompanied by the source documents used, promoting trust and verification.

## 🛠️ Tech Stack

- **Framework:** LangChain, Streamlit
- **LLM:** `llama3-8b-8192` via Groq API
- **Embedding Model:** `BAAI/bge-small-en-v1.5`
- **Vector Store:** FAISS (for dense retrieval)
- **Search:** BM25 (for sparse retrieval), Ensemble Retriever
- **Reranker:** Cohere Rerank
- **Deployment:** Hugging Face Spaces

## 🚀 Setup and Installation (Local)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[YOUR_USERNAME]/[YOUR_REPO_NAME].git
    cd ai-research-navigator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    - Create a file named `.env` in the project root.
    - Add your API keys to this file:
      ```
      GROQ_API_KEY="gsk_..."
      COHERE_API_KEY="..."
      HF_TOKEN="hf_..."
      ```

5.  **Download and process the data:**
    - The app requires pre-built vector stores. First, download the papers:
      ```bash
      python download_papers.py --collection llm
      python download_papers.py --collection cv
      ```
    - Then, ingest them to create the FAISS indexes:
      ```bash
      python ingest.py --data_dir data/large_language_models --db_dir vectorstores/llm_db
      python ingest.py --data_dir data/computer_vision --db_dir vectorstores/cv_db
      ```

6.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## ❓ Example Queries

- **(Large Language Models)** "What is the main idea behind the Transformer architecture?"
- **(Large Language Models)** "Summarize the BERT paper."
- **(Computer Vision)** "What problem does ResNet solve compared to plain, very deep networks?"
- **(Computer Vision)** "Explain the U-Net architecture."

## Limitations and Known Issues

- **Context Bleeding Bug:** There is a persistent bug where, after the application has been running and a collection has been switched, the retrieval mechanism may incorrectly pull context from the previously selected collection. This appears to be a complex state management issue between the cached LangChain objects and Streamlit's execution model. A fresh browser refresh after switching collections is the most reliable workaround.