# Tudlo Essay Summarizer with RAG + VectorDB

AI-powered essay summarization service using FastAPI, LlamaIndex, and Groq.

## 🚀 Setup Guide

### Prerequisites
- Python 3.11+ (recommended)
- Groq API key ([Get it here](https://console.groq.com/keys))
- Linux/MacOS (Windows requires WSL for ChromaDB)

### 1. Setup Virtual Environment
```bash
    python3 -m venv .venv
```
```bash
    source .venv/bin/activate
```

### 2. Setup .env file (See .env.example for format)

### 3. Install Dependencies
```bash
    pip install -r requirements.txt
```

### 3. Run the Application
```bash
    uvicorn main:app --reload
```

## Troubleshooting
### Missing python.h
```bash
  sudo apt install python3-dev
```

### Cannot install chromadb
```bash
  sudo apt install -y python3.13-dev build-essential cmake g++
```