"""
=============================================================
PIPELINE QUERY — RAG UTS Data Engineering
=============================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# CONFIG
TOP_K     = int(os.getenv("TOP_K", 3))
VS_DIR    = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")


# =============================================================
# LOAD VECTORSTORE
# =============================================================
def load_vectorstore():
    from langchain_community.vectorstores import Chroma
    from src.embeddings import get_embedding

    if not VS_DIR.exists():
        raise FileNotFoundError(
            f"Vector store tidak ditemukan di '{VS_DIR}'.\n"
            "Jalankan dulu: python src/indexing.py"
        )

    embedding_model = get_embedding()

    vectorstore = Chroma(
        persist_directory=str(VS_DIR),
        embedding_function=embedding_model
    )

    return vectorstore


# =============================================================
# RETRIEVE CONTEXT
# =============================================================
def retrieve_context(vectorstore, question: str, top_k: int = TOP_K) -> list:
    results = vectorstore.similarity_search_with_score(question, k=top_k)

    contexts = []
    for doc, score in results:
        contexts.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score": round(float(score), 4)
        })

    return contexts


# =============================================================
# BUILD PROMPT
# =============================================================
def build_prompt(question: str, contexts: list) -> str:
    context_text = "\n\n---\n\n".join(
        [f"[Sumber: {c['source']}]\n{c['content']}" for c in contexts]
    )

    prompt = f"""Kamu adalah asisten AI yang membantu analisis data pertanian di Jawa Timur.

INSTRUKSI:
- Jawab HANYA berdasarkan konteks
- Jangan mengarang
- Jika tidak ada, katakan tidak ditemukan
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
def get_answer_groq(prompt: str) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY belum diatur di file .env")

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024
    )

    return response.choices[0].message.content


# =============================================================
# MAIN PIPELINE
# =============================================================
def answer_question(question: str, vectorstore=None) -> dict:

    if vectorstore is None:
        vectorstore = load_vectorstore()

    print(f"\n🔍 Mencari konteks untuk: {question}")

    contexts = retrieve_context(vectorstore, question)
    print(f"   ✅ {len(contexts)} dokumen ditemukan")

    prompt = build_prompt(question, contexts)

    print("🤖 Mengirim ke LLM (Groq)...")

    answer = get_answer_groq(prompt)

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
    print("🤖 RAG System — UTS Data Engineering")
    print("Ketik 'keluar' untuk exit")
    print("=" * 55)

    try:
        vs = load_vectorstore()
        print("✅ Vector database berhasil dimuat")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)

    while True:
        question = input("\n❓ Pertanyaan: ").strip()

        if question.lower() in ["keluar", "exit", "quit"]:
            print("👋 Selesai")
            break

        if not question:
            print("⚠️ Pertanyaan kosong")
            continue

        try:
            result = answer_question(question, vs)

            print("\n" + "─" * 55)
            print("💬 JAWABAN:")
            print(result["answer"])

            print("\n📚 SUMBER:")
            for i, ctx in enumerate(result["contexts"], 1):
                print(f"[{i}] Score: {ctx['score']} | {ctx['source']}")
                print(f"     {ctx['content'][:120]}...")

            print("─" * 55)

        except Exception as e:
            print(f"❌ Error: {e}")
            print("Cek API key / koneksi internet")