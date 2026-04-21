from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.embeddings import get_embedding

import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
DATA_DIR = os.getenv("DATA_DIR", "data")

VS_DATA = "vectorstore_data"
VS_PDF = "vectorstore_pdf"

print("FILE INDEXING TERJALANKAN")


# =========================
# PARSE TXT → DOCUMENT + METADATA
# =========================
def parse_txt_to_docs(file_path):
    docs = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_level = "unknown"

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if "[LEVEL: PROVINSI]" in line:
            current_level = "provinsi"
            continue
        elif "[LEVEL: KABUPATEN]" in line:
            current_level = "kabupaten"
            continue

        docs.append(
            Document(
                page_content=line,
                metadata={"level": current_level}
            )
        )

    return docs


# =========================
# MAIN INDEXING
# =========================
def build_index_langchain():

    print("=" * 50)
    print("Memulai Pipeline Indexing (FINAL)")
    print("=" * 50)

    # ===== LOAD TXT =====
    print("\n📊 Load data produksi...")
    txt_docs = parse_txt_to_docs(f"{DATA_DIR}/Narasi_Produksi_Padi_Jatim.txt")

    # ===== LOAD PDF =====
    print("📘 Load dokumen PDF...")
    loader1 = PyPDFLoader(f"{DATA_DIR}/indikator-pertanian-provinsi-jawa-timur-2021.pdf")
    loader2 = PyPDFLoader(f"{DATA_DIR}/indikator-pertanian-provinsi-jawa-timur-2024.pdf")

    docs_pdf = loader1.load() + loader2.load()

    # ===== SPLITTER =====
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks_txt = splitter.split_documents(txt_docs)
    chunks_pdf = splitter.split_documents(docs_pdf)

    print(f"TXT chunks: {len(chunks_txt)}")
    print(f"PDF chunks: {len(chunks_pdf)}")

    # ===== EMBEDDING =====
    embedding_model = get_embedding()

    # ===== VECTORSTORE DATA =====
    print("\n💾 Simpan DATA...")
    Chroma.from_documents(
        documents=chunks_txt,
        embedding=embedding_model,
        persist_directory=VS_DATA
    )

    # ===== VECTORSTORE PDF =====
    print("💾 Simpan PDF...")
    Chroma.from_documents(
        documents=chunks_pdf,
        embedding=embedding_model,
        persist_directory=VS_PDF
    )

    print("\n✅ Indexing selesai!")


if __name__ == "__main__":
    print("RUN INDEXING...")
    build_index_langchain()