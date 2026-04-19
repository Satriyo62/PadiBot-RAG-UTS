from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding():
    """
    Mengembalikan model embedding yang digunakan untuk RAG
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )