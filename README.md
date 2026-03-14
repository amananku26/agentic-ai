# 🧠 IFC Autonomous AI Agent

> A self-refining, multi-tool AI agent built with **LangGraph**, **Streamlit**, and **Google Vertex AI** that intelligently answers complex questions using a Planner → Executor → Synthesizer → Critique loop.

---

## 📸 Overview

This project is an **Agentic RAG (Retrieval-Augmented Generation)** system designed to answer questions about the **IFC Annual Report** while also leveraging **live web search** when needed. The agent autonomously decides which tool to use, collects evidence, drafts an answer, and self-critiques — retrying until it is satisfied or hits a maximum iteration limit.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌─────────┐
│ Planner │ ──► │ Executor │ ──► │ Synthesizer │ ──► │ Critique│
└─────────┘     └──────────┘     └─────────────┘     └────┬────┘
     ▲                                                      │
     │              SATISFIED / MAX ITERATIONS              │
     └──────────────────────────────────────────────────────┘
```

| Node | Role |
|------|------|
| **Planner** | Selects the right tool (`document_search` or `web_search`) and rewrites the query |
| **Executor** | Runs the chosen tool and collects raw evidence |
| **Synthesizer** | Drafts a complete answer from all collected evidence |
| **Critique** | Evaluates the draft and decides to continue refining or finish |

---

## ✨ Features

- 🔁 **Self-Refining Loop** — Agent retries with improved queries if the answer is incomplete
- 📄 **Document Search** — Semantic search over IFC Annual Report using FAISS + Vertex AI Embeddings
- 🌐 **Web Search** — Falls back to DuckDuckGo for general or current knowledge
- 🧠 **Gemini 2.0 Flash** — Powered by Google's latest LLM via Vertex AI
- 💬 **Streamlit Chat UI** — Clean chat interface with live agent thinking steps
- 🔍 **Transparent Reasoning** — Real-time display of plan, execution, and critique steps

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 2.0 Flash (Vertex AI) |
| **Embeddings** | `text-embedding-004` (Vertex AI) |
| **Vector Store** | FAISS |
| **Agent Framework** | LangGraph |
| **Web Search** | DuckDuckGo (`langchain-community`) |
| **Frontend** | Streamlit |
| **Cloud** | Google Cloud Platform (GCP) |

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit app + LangGraph agent
├── faiss_index/            # Pre-built FAISS vector store (from IFC PDF)
│   ├── index.faiss
│   └── index.pkl
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.9+
- A **Google Cloud Project** with Vertex AI API enabled
- Application Default Credentials (ADC) configured:
  ```bash
  gcloud auth application-default login
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`** should include:
```
streamlit
langchain
langchain-google-vertexai
langchain-community
langgraph
faiss-cpu
duckduckgo-search
```

### 3. Build the FAISS Index

Before running the app, ingest your IFC PDF to create the vector store:

```python
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("ifc_annual_report.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)

embeddings = VertexAIEmbeddings(model_name="text-embedding-004", project="YOUR_PROJECT_ID", location="us-central1")
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.save_local("faiss_index")
```

### 4. Configure Project Settings

In `app.py`, update these constants:

```python
PROJECT_ID = "your-gcp-project-id"
LOCATION = "us-central1"          # or your preferred region
MAX_ITERATIONS = 3                # max self-refinement loops
```

### 5. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 💡 Example Queries

- *"What was IFC's total investment commitment in FY2023?"*
- *"Compare IFC's climate finance strategy with current World Bank trends."*
- *"What sectors did IFC focus on in emerging markets?"*

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PROJECT_ID` | `gd-gcp-gridu-genai` | GCP Project ID |
| `LOCATION` | `us-central1` | Vertex AI region |
| `MAX_ITERATIONS` | `3` | Max agent refinement loops |
| `VECTOR_STORE_PATH` | `faiss_index` | Path to FAISS index folder |

---

## 🔒 Notes

- The FAISS index is loaded with `allow_dangerous_deserialization=True` — only use indexes you have built yourself.
- Ensure your GCP service account has the `Vertex AI User` role.
- DuckDuckGo search is rate-limited; for production use consider a paid search API.

---

## 📄 License

This project is for educational and internal demonstration purposes.

---

## 🙏 Acknowledgements

- [IFC (International Finance Corporation)](https://www.ifc.org/)
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [Streamlit](https://streamlit.io/)
