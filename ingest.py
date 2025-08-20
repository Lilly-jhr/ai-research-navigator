import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import argparse


from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def ingest_pdfs(data_dir: str, db_dir: str):
    """
    Load PDFs from a directory, split into chunks, embed, and save FAISS index.
    """
    print(f"📂 Loading documents from: {data_dir}")
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")

    # Chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # ~1k chars per chunk
        chunk_overlap=150  # small overlap for context
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📝 Split into {len(chunks)} chunks")

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},  
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(db_dir, exist_ok=True)
    vectorstore.save_local(db_dir)
    print(f"💾 FAISS index saved to: {db_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into a FAISS vector store.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory to load PDFs from.")
    parser.add_argument("--db_dir", type=str, required=True, help="Directory to save the FAISS index to.")
    args = parser.parse_args()
    
    ingest_pdfs(data_dir=args.data_dir, db_dir=args.db_dir)
    # ingest_pdfs(data_dir="data/cv_papers", db_dir="vectorstores/cv_db")
