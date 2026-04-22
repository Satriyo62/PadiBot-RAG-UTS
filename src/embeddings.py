import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def get_embedding():
    # Menggunakan model multilingual yang lebih pintar untuk Bahasa Indonesia
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"🔄 Memuat embedding model: {model_name}...")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        # Menggunakan CPU, atau GPU jika tersedia
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )