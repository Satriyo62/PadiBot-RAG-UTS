<<<<<<< HEAD
# 🤖 Sistem Tanya Jawab Berbasis Retrieval-Augmented Generation (RAG) untuk Analisis Data Produksi Padi di Jawa Timur

> **Retrieval-Augmented Generation** — Sistem Tanya-Jawab Cerdas Berbasis Dokumen

Starter pack ini adalah **kerangka awal** proyek RAG untuk UTS Data Engineering D3/D4.

## 📖 Deskripsi Proyek

**PadiBot** adalah sistem tanya-jawab cerdas berbasis dokumen yang mengimplementasikan arsitektur *Retrieval-Augmented Generation* (RAG). Proyek ini dikembangkan secara spesifik untuk memproses dan menjawab pertanyaan seputar domain **Pertanian**, berfokus pada dokumen referensi yang diberikan (seperti data Indikator Pertanian Provinsi).

Alih-alih hanya mengandalkan pengetahuan umum bawaan dari AI, PadiBot bekerja dengan cara:
1. Membaca dan mengekstrak informasi dari dokumen lokal kita.
2. Menyimpan potongan informasi tersebut ke dalam *vector database* (ChromaDB).
3. Mencari konteks yang paling relevan saat ada pertanyaan, lalu memprosesnya melalui LLM (Groq) untuk menghasilkan jawaban yang akurat dan sesuai fakta dari dokumen.

Proyek ini dibangun menggunakan *framework* LangChain dengan antarmuka Streamlit, sebagai bentuk implementasi *pipeline* Data Engineering untuk pemenuhan tugas Ujian Tengah Semester (UTS).

---

## 👥 Identitas Kelompok

| Nama | NIM | Tugas Utama |
|------|-----|-------------|
| Jimli Dwi Assiddiqi  | 244311015 | Data Analyst         |
| Muhsyam Fahriel Septiansyah  | 244311021 | Data Engineer         |
| Satriyo Wicaksono Yunan Mubarok  | 244311027 | Project Manager         |

**Topik Domain:** *Pertanian*  
**Stack yang Dipilih:** *LangChain*  
**LLM yang Digunakan:** *Groq*  
**Vector DB yang Digunakan:** *ChromaDB*

---

## 🗂️ Struktur Proyek

```
padibot-rag-uts/
├── data/                    # Dokumen sumber Anda (PDF, TXT, dll.)
│   ├── Data_Narasi_Pertanian_Jatim_Siap_RAG.csv
│   ├── indikator-pertanian-provinsi-jawa-timur-2021.pdf
│   └── indikator-pertanian-provinsi-jawa-timur-2024.pdf         
├── src/
│   ├── indexing.py          
│   ├── query.py             
│   └── embeddings.py        
├── docs/
│   └── arsitektur.png       # 📌 Diagram arsitektur
├── evaluation/
│   └── hasil_evaluasi.xlsx  # 📌 Tabel evaluasi 10 pertanyaan
├── .env.example             # Template environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Cara Memulai (Quickstart)

### 1. Clone & Setup

```bash
# Clone repository ini
git clone https://github.com/Satriyo62/PadiBot-RAG-UTS.git
cd padibot-rag-uts

# Buat virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# atau: venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Konfigurasi API Key

```bash
# Salin template env
cp .env.example .env

# Edit .env.example dan isi API key Anda
# JANGAN commit file .env ke GitHub!
```

### 3. Siapkan Dokumen

Letakkan dokumen sumber Anda di folder `data/`:
```bash
# Contoh: salin PDF atau TXT ke folder data
cp indikator-pertanian-provinsi-jawa-timur-2024.pdf data/
```

### 4. Jalankan Indexing (sekali saja)

```bash
python -m src.indexing.py
```

### 5. Jalankan Sistem RAG

```bash
# Via CLI
python -m src.query.py
```

---

## 🔧 Konfigurasi

Semua konfigurasi utama ada di `.env.example` (atau langsung di setiap file):

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `CHUNK_SIZE` | 500 | Ukuran setiap chunk teks (karakter) |
| `CHUNK_OVERLAP` | 50 | Overlap antar chunk |
| `TOP_K` | 3 | Jumlah dokumen relevan yang diambil |
| `MODEL_NAME` | *llama3* | Nama model LLM yang digunakan |

---

## 📊 Hasil Evaluasi


| # | Pertanyaan | Jawaban Sistem | Jawaban Ideal | Skor (1-5) |
|---|-----------|----------------|---------------|-----------|
| 1 | ... | ... | ... | ... |
| 2 | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... |
| 4 | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... |

**Rata-rata Skor:** ...  
**Analisis:** ...

---

## 🏗️ Arsitektur Sistem

![alt text](https://github.com/Satriyo62/PadiBot-RAG-UTS/blob/main/docs/arsitektur.png?raw=true)

```
[Dokumen] → [Loader] → [Splitter] → [Embedding] → [Vector DB]
                                                         ↕
[User Query] → [Query Embed] → [Retriever] → [Prompt] → [LLM] → [Jawaban]
```

---

## 📚 Referensi & Sumber

- Framework: *(LangChain docs / LlamaIndex docs)*
- LLM: *Groq*
- Vector DB: *ChromaDB*
- Tutorial yang digunakan: *https://reference.langchain.com/python/langchain-mongodb/utils/maximal_marginal_relevance*

---

## 👨‍🏫 Informasi UTS

- **Mata Kuliah:** Data Engineering
- **Program Studi:** D4 Teknologi Rekayasa Perangkat Lunak
- **Deadline:** *23 April 2026*
=======

>>>>>>> ad10b743c360c8333a1b2356e83805154222005e
