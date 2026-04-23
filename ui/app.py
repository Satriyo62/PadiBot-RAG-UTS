"""
=============================================================
ANTARMUKA STREAMLIT — RAG UTS Data Engineering
=============================================================
"""

import sys
from pathlib import Path

# Import dari src/
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─── Config ─────────────────────────────────────────
st.set_page_config(
    page_title="PadiBot RAG",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 PadiBot — RAG System")
st.caption("Sistem Tanya Jawab Data Produksi Padi Jawa Timur")
st.divider()

# ─── Sidebar ────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Pengaturan")

    show_context = st.checkbox("Tampilkan konteks", value=True)

    st.divider()

    st.markdown("""
    **Proyek:** PadiBot RAG  
    **LLM:** Groq (LLaMA 3)  
    **Vector DB:** ChromaDB  
    **Embedding:** MiniLM Multilingual  
    """)

# ─── Load Vectorstore ───────────────────────────────
@st.cache_resource
def load_vs():
    try:
        from query import load_vectorstore
        return load_vectorstore(), None
    except Exception as e:
        return None, str(e)


vectorstore, error = load_vs()

if error:
    st.error(error)
    st.stop()

st.success("Vector database siap digunakan!")

# ─── Chat State ─────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Tampilkan Chat ─────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and show_context and "contexts" in msg:
            with st.expander("📚 Konteks"):
                for i, ctx in enumerate(msg["contexts"], 1):
                    st.markdown(f"**[{i}]** `{ctx['source']}`")
                    st.text(ctx["content"][:300] + "...")
                    st.divider()

# ─── Input User ─────────────────────────────────────
if question := st.chat_input("Tanyakan sesuatu tentang produksi padi..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Mencari jawaban..."):

            try:
                from query import answer_question

                result = answer_question(question, vectorstore)

                st.write(result["answer"])

                if show_context:
                    with st.expander("📚 Konteks yang digunakan"):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"**[{i}]** `{ctx['source']}`")
                            st.text(ctx["content"][:300] + "...")
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "contexts": result["contexts"]
                })

            except Exception as e:
                st.error(f"Error: {e}")

# ─── Reset Chat ─────────────────────────────────────
if st.session_state.messages:
    if st.button("🧹 Hapus Chat"):
        st.session_state.messages = []
        st.rerun()