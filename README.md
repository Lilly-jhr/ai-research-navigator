# 🔬 AI Research Navigator

**An advanced, conversational RAG system for exploring and understanding complex AI research papers.**

---

### 🚀 **** 👈 *(Paste your Streamlit Community Cloud or Hugging Face Spaces URL here)*

---

## 📝 Project Description

The AI Research Navigator is an interactive web application designed to help users navigate dense technical literature. Users can select from different collections of research papers (e.g., Large Language Models, Computer Vision), ask specific questions, and request summaries on various topics.

The application leverages a sophisticated Retrieval-Augmented Generation (RAG) pipeline to deliver accurate, fast, and contextually-grounded answers with source citations, making complex information more accessible.

## ✨ Key Features

- **Conversational Q&A:** Ask questions in natural language and get precise, grounded answers.
- **Advanced Hybrid Search:** Combines semantic (FAISS) and keyword (BM25) search to find the most relevant document chunks.
- **High-Quality Reranking:** Uses the Cohere Rerank model to refine search results for maximum accuracy.
- **Switchable Knowledge Bases:** Seamlessly switch between different research collections.
- **Summarization Mode:** Ask the chatbot to summarize key concepts, papers, or topics with a simple toggle.
- **Fast & Interactive UI:** Built with Streamlit and powered by the high-speed Groq API for near-instant AI responses.
- **Source Citation:** Every answer is accompanied by the source documents used, promoting trust and easy verification.

## 🛠️ Tech Stack

-   **Framework:** LangChain, Streamlit
-   **LLM:** `llama3-8b-8192` via Groq API
-   **Embedding Model:** `BAAI/bge-small-en-v1.5`
-   **Vector Store:** FAISS (for dense retrieval)
-   **Search:** BM25 (for sparse retrieval), Ensemble Retriever
-   **Reranker:** Cohere Rerank
-   **Deployment:** Streamlit Community Cloud / Hugging Face Spaces

## 🚀 Local Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/lilly-jhr/ai-research-navigator.git
    cd ai-research-navigator
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables:**
    -   Create a file named `.env` in the project root.
    -   Add your API keys to this file:
        ```env
        GROQ_API_KEY="gsk_..."
        COHERE_API_KEY="..."
        HF_TOKEN="hf_..."
        ```

5.  **Download and Process Data:**
    -   The application requires pre-built vector stores. First, download the papers using the provided script:
        ```bash
        python download_papers.py --collection llm
        python download_papers.py --collection cv
        ```
    -   Then, ingest the papers to create the FAISS indexes:
        ```bash
        python ingest.py --data_dir data/large_language_models --db_dir vectorstores/llm_db
        python ingest.py --data_dir data/computer_vision --db_dir vectorstores/cv_db
        ```

6.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```

## ❓ Example Queries

-   **Select "Large Language Models" collection:**
    -   *(Q&A Mode)*: "What is the main idea behind the Transformer architecture?"
    -   *(Summarization Mode)*: "Summarize the key contributions of the BERT paper."

-   **Select "Computer Vision" collection:**
    -   *(Q&A Mode)*: "What problem does ResNet solve compared to plain, very deep networks?"
    -   *(Summarization Mode)*: "Provide a summary of the U-Net architecture."

## ⚠️ Known Issues and Limitations

-   **Initial Context Bug:** On some occasions, when switching between document collections, the retriever may incorrectly use context from the previously selected collection. This appears to be a complex state management interaction between the cached LangChain objects and Streamlit's execution model.
-   **Workaround:** A hard refresh of the browser page (Ctrl+F5 or Cmd+Shift+R) after switching collections is the most reliable way to ensure the correct context is loaded.