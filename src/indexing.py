from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embedding

import os
from dotenv import load_dotenv

# LOAD .env
load_dotenv()

# CONFIG dari .env
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATA_DIR = os.getenv("DATA_DIR", "data")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")

print("FILE INDEXING TERJALANKAN")

def build_index_langchain():

    print("=" * 50)
    print("Memulai Pipeline Indexing (LangChain)")
    print("=" * 50)

    print("\n📄 Memuat dokumen...")

    # ===== LOAD DATA =====
    loader1 = PyPDFLoader(f"{DATA_DIR}/indikator-pertanian-provinsi-jawa-timur-2021.pdf")
    docs1 = loader1.load()

    loader2 = PyPDFLoader(f"{DATA_DIR}/indikator-pertanian-provinsi-jawa-timur-2024.pdf")
    docs2 = loader2.load()

    csv_loader = CSVLoader(f"{DATA_DIR}/Data_Narasi_Pertanian_Jatim_Siap_RAG.csv")
    csv_docs = csv_loader.load()

    documents = docs1 + docs2 + csv_docs

    print(f"   Total dokumen: {len(documents)}")

    print("\n✂️ Memecah dokumen...")

    # ===== SPLITTER =====
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print(f"   Total chunks: {len(chunks)}")

    print("\n🧠 Membuat embedding...")

    embedding_model = get_embedding()

    print("\n💾 Menyimpan ke ChromaDB...")

    # ===== VECTOR DB =====
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTORSTORE_DIR
    )

    print("\n✅ Indexing selesai!")


if __name__ == "__main__":
    print("RUN INDEXING...")
    build_index_langchain()