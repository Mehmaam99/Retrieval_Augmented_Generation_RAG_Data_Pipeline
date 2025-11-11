# Retrieval-Augmented Generation (RAG) Data Pipeline

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** data pipeline designed for handling large-scale **multimedia data** such as audio and video files. It automates the process from **data ingestion** to **vector embedding**, enabling efficient retrieval and context-aware generative AI responses.

The pipeline is modular and built for scalability, integrating **speech-to-text transcription**, **speaker diarization**, **FAISS-based vector storage**, and **retrieval mechanisms** for generative AI models.

---

## 🧩 Architecture

The RAG Data Pipeline consists of the following major components:

### 1. **AWS S3 Integration** (`aws_bucket.py`)

* Downloads multimedia files (e.g., `.webm`) from an **S3 bucket**.
* Stores them in the local directory `downloads/`.
* Manages file flow between S3 and the local environment.

### 2. **Audio Transcription Module** (`transcription.py`)

* Uses **Whisper** (Hugging Face Transformers pipeline) for converting audio to text.
* Handles long-form audio files with chunking support.
* Outputs **.txt transcription files** in the directory `transcriptions/downloads/`.

### 3. **Embedding Engine** (`embedding_engine.py`)

* Reads transcription `.txt` files.
* Uses **SentenceTransformers (all-MiniLM-L6-v2)** to generate embeddings.
* Builds a **FAISS vector index** for efficient semantic search.
* Saves index files in `data/faiss_index/`.

### 4. **Main Orchestrator** (`main.py`)

* Handles the entire workflow:

  * Downloads files from S3.
  * Processes audio files for transcription.
  * Generates embeddings and stores them in FAISS.
* Includes logging, error handling, and retry mechanisms.

### 5. **Configuration** (`config.py`)

* Defines parameters for Whisper and other components.
* Can be customized for CPU or GPU processing.

---

## 🗂️ Folder Structure

```
Multimedia_RAG_Data_Pipeline/
│
├── src/
│   ├── main.py                  # Orchestration logic
│   ├── aws_bucket.py            # S3 download pipeline
│   ├── transcription.py         # Whisper transcription
│   ├── embedding_engine.py      # FAISS embedding builder
│   ├── config.py                # Configuration settings
│   └── utils/                   # Helper functions (optional)
│
├── downloads/                   # Temporary audio storage
├── transcriptions/downloads/     # Output transcriptions (.txt)
├── data/faiss_index/             # Vector database
├── logs/                         # Application and error logs
├── requirements.txt              # Dependencies
├── .env                          # Environment variables (HF_TOKEN, S3_BUCKET_NAME)
└── README.md                     # Documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mehmaam99/Retrieval_Augmented_Generation_RAG_Data_Pipeline.git
cd Retrieval_Augmented_Generation_RAG_Data_Pipeline
```

### 2. Create a Virtual Environment

```bash
conda create -n myenv python=3.10
conda activate myenv
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
HF_TOKEN=your_huggingface_token
S3_BUCKET_NAME=your_bucket_name
AWS_ACCESS_KEY_ID=your_aws_access_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key

```

---

## 🧠 Usage

### Run the Main Pipeline

```bash
python src/main.py
```

This will:

1. Download audio files from S3.
2. Transcribe them into text.
3. Generate embeddings and build FAISS vector stores.

## 📊 Logging

Logs are stored in the `logs/` directory:

* `main_execution.log` — Overall workflow logs.
* `download.log` — S3 file download logs.
* `stats.log` — Processing statistics.

---

## 💡 Key Features

✅ Fully automated audio-to-text + vector embedding pipeline./n
✅ Modular architecture for extensibility./n
✅ Handles long-form audio and multilingual transcription./n
✅ Scalable to large datasets (tens of thousands of files). /n
✅ Built-in FAISS vector search for efficient retrieval. /n
✅ Hugging Face Whisper integration for transcription. /n

---

## 🧰 Tech Stack

* **Python 3.10**
* **FAISS** – Vector similarity search
* **SentenceTransformers** – Text embeddings
* **Whisper (Hugging Face)** – Audio transcription
* **boto3** – AWS S3 integration
* **pandas**, **numpy**, **logging**, **dotenv** – Utility and data management

---

## 🧩 Future Enhancements

* Integration with **LangChain** or **LlamaIndex** for retrieval-augmented chatbot functionality.
* Support for real-time transcription.
* Cloud-native orchestration with **Airflow** or **AWS Lambda**.
* Multi-modal support for image/video captions.

---

## 👨‍💻 Author

**Muhammad Mehmaam**
Data Engineer & AI Developer
🔗 [GitHub Profile](https://github.com/Mehmaam99)

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.



