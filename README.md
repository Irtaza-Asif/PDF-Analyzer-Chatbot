# 🤖 PDF Analyzer Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDFs and ask questions about their content. Containerized with Docker and deployed on AWS ECS Fargate.

## 🚀 Features

- 📄 Upload and analyze PDFs
- 💬 ChatGPT-style interface
- 🧠 Context-aware answers using RAG
- 📚 Source citations with page numbers
- ⚡ Fast semantic search with FAISS
- 🐳 Fully containerized with Docker
- ☁️ Deployed on AWS ECS Fargate

## 🏗 Architecture

```
User → ALB (port 80) → ECS Fargate (Docker container) → FAISS + flan-t5-base
```

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| RAG Pipeline | LangChain |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | google/flan-t5-base (local, no API key needed) |
| Containerization | Docker |
| Registry | AWS ECR |
| Deployment | AWS ECS Fargate |
| Load Balancer | AWS ALB |

## 🖥 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Irtaza-Asif/PDF-Analyzer-Chatbot.git
cd PDF-Analyzer-Chatbot
```

### 2. Create virtual environment
```bash
# Windows
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1

# Mac/Linux
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open `http://localhost:8501`

## ☁️ Deployment

This project is containerized with Docker and deployed on **AWS ECS Fargate** with an **Application Load Balancer** for public access. Model weights for `flan-t5-base` and `all-MiniLM-L6-v2` are baked into the Docker image at build time for fast cold starts.

## 🌐 Live Demo

http://pdf-analyzer-alb-242297481.eu-north-1.elb.amazonaws.com

> **Note:** The demo may be scaled down to avoid idle costs. If it doesn't load, it can be live within 2-3 minutes on request.

## 🎯 Roadmap

- [x] RAG pipeline with local HuggingFace models
- [x] Streamlit chat interface with source citations
- [x] Docker containerization
- [x] AWS ECS Fargate deployment with ALB
- [ ] Multi-PDF support
- [ ] Chat memory
- [ ] HTTPS with custom domain

## 👤 Author

Irtaza Asif