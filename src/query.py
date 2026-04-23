"""
=============================================================
PIPELINE QUERY — RAG UTS Data Engineering (MMR)
=============================================================
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# CONFIG
TOP_K     = int(os.getenv("TOP_K", 15))
VS_DIR    = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")


# =============================================================
# LOAD VECTORSTORE
# =============================================================
def load_vectorstore():
    from langchain_chroma import Chroma
    from src.embeddings import get_embedding

    if not VS_DIR.exists():
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{VS_DIR}'.\n"
            "Jalankan dulu: python -m src.indexing"
        )

    embedding_model = get_embedding()

    vectorstore = Chroma(
        persist_directory=str(VS_DIR),
        embedding_function=embedding_model
    )

    return vectorstore


# =============================================================
# RETRIEVE CONTEXT (MMR)
# =============================================================
def retrieve_context(vectorstore, question: str, top_k: int = TOP_K) -> list:
    """
    Menggunakan Max Marginal Relevance (MMR)
    agar hasil tidak hanya relevan tapi juga beragam
    """

    docs = vectorstore.max_marginal_relevance_search(
        query=question,
        k=top_k,
        fetch_k=50,       # ambil kandidat lebih banyak dulu
        lambda_mult=0.5   # 0.5 = balance relevansi & diversity
    )

    contexts = []
    for doc in docs:
        contexts.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": "MMR Selected"
        })

    return contexts


# =============================================================
# BUILD PROMPT
# =============================================================
def build_prompt(question: str, contexts: list) -> str:
    context_text = "\n\n---\n\n".join(
        [f"[Sumber: {c['source']}]\n{c['content']}" for c in contexts]
    )

    prompt = f"""Kamu adalah asisten AI yang membantu analisis data produksi padi di Jawa Timur.

INSTRUKSI:
- Jawab HANYA berdasarkan konteks
- Jangan mengarang
- Fokus hanya pada kabupaten, bulan, dan tahun yang sama dengan pertanyaan
- Pastikan nama kabupaten harus sesuai dengan pertanyaan
- Jangan ambil data dari kabupaten lain
- Jika tidak ditemukan, katakan tidak ditemukan
- Jawab singkat, jelas, Bahasa Indonesia

KONTEKS:
{context_text}

PERTANYAAN:
{question}

JAWABAN:"""

    return prompt


# =============================================================
# GROQ LLM
# =============================================================
def get_answer(prompt: str) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY belum diatur di .env")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )

    return response.choices[0].message.content


# =============================================================
# MAIN PIPELINE
# =============================================================
def answer_question(question: str, vectorstore=None) -> dict:

    if vectorstore is None:
        vectorstore = load_vectorstore()

    print(f"\n🔍 Mencari konteks (MMR) untuk: {question}")

    contexts = retrieve_context(vectorstore, question)
    print(f"   ✅ {len(contexts)} dokumen relevan ditemukan")

    prompt = build_prompt(question, contexts)

    print("🤖 Mengirim ke LLM (Groq)...")

    answer = get_answer(prompt)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts
    }


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":

    print("=" * 55)
    print("🤖 RAG System — MMR Mode")
    print("=" * 55)

    try:
        vs = load_vectorstore()
        print("✅ Vector database berhasil dimuat")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("\n❓ Masukkan pertanyaan: ").strip()

    if not question:
        print("⚠️ Pertanyaan kosong")
        exit(0)

    try:
        result = answer_question(question, vs)

        print("\n" + "─" * 55)
        print("💬 JAWABAN:")
        print(result["answer"])

        print("\n📚 SUMBER:")
        for i, ctx in enumerate(result["contexts"], 1):
            print(f"[{i}] {ctx['score']} | {ctx['source']}")
            print(f"     {ctx['content'][:120]}...")

        print("─" * 55)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Cek API key / koneksi internet")