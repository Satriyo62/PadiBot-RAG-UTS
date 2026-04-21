import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 5))
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")

VS_DATA = Path("./vectorstore_data")
VS_PDF = Path("./vectorstore_pdf")


# =========================
# LOAD VECTORSTORE
# =========================
def load_vs(path):
    from langchain_community.vectorstores import Chroma
    from src.embeddings import get_embedding

    return Chroma(
        persist_directory=str(path),
        embedding_function=get_embedding()
    )


# =========================
# DETECT QUERY TYPE
# =========================
def detect_query_type(question):
    q = question.lower()

    if any(k in q for k in ["tren", "mengapa", "kenapa", "analisis"]):
        return "analisis"

    return "data"


# =========================
# DETECT LEVEL
# =========================
def detect_level(question):
    q = question.lower()

    if "provinsi" in q:
        return "provinsi"
    elif "kabupaten" in q:
        return "kabupaten"

    return None


# =========================
# RETRIEVE
# =========================
def retrieve(question):

    query_type = detect_query_type(question)
    level = detect_level(question)

    vs_data = load_vs(VS_DATA)
    vs_pdf = load_vs(VS_PDF)

    if query_type == "data":
        print("➡️ Mode: DATA")

        if level:
            print(f"🎯 Filter level: {level}")
            return vs_data.similarity_search(
                question,
                k=TOP_K,
                filter={"level": level}
            )

        return vs_data.similarity_search(question, k=TOP_K)

    elif query_type == "analisis":
        print("➡️ Mode: ANALISIS")
        return vs_pdf.similarity_search(question, k=TOP_K)


# =========================
# BUILD PROMPT
# =========================
def build_prompt(question, docs):

    context = "\n\n---\n\n".join([d.page_content for d in docs])

    return f"""
Kamu adalah asisten AI ahli statistik pertanian Jawa Timur.

INSTRUKSI:
- Jawab hanya dari konteks
- Jika pertanyaan menyebut PROVINSI → gunakan hanya data provinsi
- DILARANG menggunakan data kabupaten untuk menjawab provinsi
- Jika tren → bandingkan angka
- Jangan mengarang

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""


# =========================
# GROQ
# =========================
def get_answer(prompt):
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    res = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return res.choices[0].message.content


# =========================
# MAIN
# =========================
def answer_question(q):
    docs = retrieve(q)
    print (docs)
    prompt = build_prompt(q, docs)
    return get_answer(prompt)


# =========================
# CLI
# =========================
if __name__ == "__main__":

    print("=" * 50)
    print("RAG FINAL READY 🔥")
    print("=" * 50)

    while True:
        q = input("\n❓ ").strip()

        if q.lower() in ["exit", "keluar"]:
            break

        print("\n💬", answer_question(q))