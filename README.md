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
├── docs/
│   └── arsitektur.png       # 📌 Diagram arsitektur
├── evaluation/
│   └── hasil_evaluasi.csv  # 📌 Tabel evaluasi 10 pertanyaan         
├── src/
│   ├── indexing.py          
│   ├── query.py             
│   └── embeddings.py
├── ui/
│   └── app.py        
├── .env.example             # Template environment variables
├── .gitignore
├── README.md
└── requirements.txt

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
# Dengan Streamlit UI
python -m streamlit run ui.app.py

# Atau via CLI
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
| 1 | Sebutkan angka produksi panen padi di Kabupaten Madiun pada bulan Januari tahun 2018 dan bulan Maret tahun 2018, lalu bandingkan apakah terjadi peningkatan atau penurunan? | Berikut adalah angka produksi panen padi di Kabupaten Madiun pada bulan Januari tahun 2018 dan bulan Maret tahun 2018:Januari 2018: 812.18 tonMaret 2018: 94455.58 tonUntuk membandingkan, kita hitung selisih antara kedua angka tersebut:$94455.58 - 812.18 = 93643.40$Karena angka produksi panen padi pada bulan Maret tahun 2018 (94455.58 ton) lebih besar daripada pada bulan Januari tahun 2018 (812.18 ton), maka terjadi peningkatan produksi panen padi sebesar 93643.40 ton. | ... | ... |
| 2 | Berapa produksi padi di Kabupaten Blitar pada bulan Januari tahun 2021? | Produksi padi Kabupaten Blitar bulan januari tahun 2021 adalah 10497.28 ton. | ... | ... |
| 3 | Berapa produksi padi di Kabupaten Ngawi pada bulan Februari tahun 2021? | Produksi padi di Kabupaten Ngawi pada bulan Februari tahun 2021 adalah 36572.31 ton. | ... | ... |
| 4 | Apakah produksi padi di Kabupaten Blitar meningkat dari Januari ke Maret tahun 2021? | Untuk mengetahui apakah produksi padi di Kabupaten Blitar meningkat dari Januari ke Maret tahun 2021  kita perlu mencari data produksi padi di Kabupaten Blitar pada bulan Januari dan Maret tahun 2021 Namun tidak ada data produksi padi di Kabupaten Blitar pada bulan Januari tahun 2021 dalam konteks yang diberikan. Oleh karena itu saya tidak dapat memberikan jawaban yang akurat Jika Anda memiliki data tambahan atau konteks yang lebih luas, saya dapat membantu Anda menganalisis data tersebut. | ... | ... |
| 5 | Apakah produksi padi di Kabupaten Blitar meningkat dari Januari ke Maret tahun 2018? | Untuk mengetahui apakah produksi padi di Kabupaten Blitar meningkat dari Januari ke Maret tahun 2018  kita perlu mencari data produksi padi di Kabupaten Blitar pada bulan Januari dan Maret tahun 2018 Namun tidak ada data produksi padi di Kabupaten Blitar pada bulan Januari tahun 2018 dalam konteks yang diberikan. | ... | ... |
| 6 | Faktor apa saja yang mempengaruhi perubahan produksi padi di Jawa Timur berdasarkan data yang tersedia? | Berdasarkan data yang tersedia faktor yang mempengaruhi perubahan produksi padi di Jawa Timur adalah: El Nino: El Nino menyebabkan turunnya luas panen padi sehingga produksi padi mengalami penurunan Penggunaan metode Kerangka Sampel Area (KSA): Pada tahun 2018 penghitungan luas panen komoditas padi mulai menggunakan metode KSA sehingga produksi dan luas panen komoditas padi tidak dapat diperbandingkan dengan luas panen dan produksi pada tahun sebelumnya. Pengaruh cuaca: Cuaca yang tidak mendukung dapat mempengaruhi produksi padi seperti pada tahun 2021 yang mengalami penurunan produksi padi karena cuaca yang tidak mendukung. Pengaruh faktor lingkungan: Faktor lingkungan seperti kelembaban suhu dan curah hujan dapat mempengaruhi produksi padi. Namun perlu diingat bahwa data yang tersedia tidak menyebutkan secara spesifik faktor-faktor yang mempengaruhi perubahan produksi padi di Jawa Timur. | ... | ... |
| 7 | Apakah produksi padi di Kabupaten Magetan meningkat dari bulan Januari ke bulan Maret tahun 2019 | Produksi padi di Kabupaten Magetan pada bulan Januari tahun 2019 adalah 1764.91 ton, sedangkan pada bulan Maret tahun 2019 adalah 73480.11 ton. Jadi, produksi padi di Kabupaten Magetan meningkat dari bulan Januari ke bulan Maret tahun 2019. | ... | ... |
| 8 | Bagaimana gambaran umum kondisi produksi padi di Jawa Timur berdasarkan data yang tersedia? | Berdasarkan data yang tersedia, produksi padi di Jawa Timur terbesar berada di Kabupaten Lamongan, diikuti Kabupaten Ngawi dan Kabupaten Bojonegoro. Kontribusi produksi padi Provinsi Jawa Timur sebesar 17,44% terhadap produksi padi nasional. | ... | ... |
| 9 | Apakah produksi atau produktivitas padi di Jawa Timur mengalami fluktuasi dalam beberapa tahun terakhir? | Ya, berdasarkan data yang tersedia, produktivitas padi di Jawa Timur mengalami fluktuasi dalam beberapa tahun terakhir. Pada tahun 2018, produktivitas tertinggi mencapai 5,76 ton per hektar, sedangkan pada tahun 2020, produktivitas terendah mencapai 5,60 ton per hektar. | ... | ... |
| 10 | Berapa produksi padi Kabupaten Ngawi bulan Januari tahun 2021? | Produksi padi Kabupaten Ngawi bulan Januari tahun 2021 adalah 4448.31 ton. | ... | ... |

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

- Framework: *https://docs.langchain.com/oss/python/langchain/quickstart*
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
