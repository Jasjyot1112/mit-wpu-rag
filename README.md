# MIT-WPU AI-Powered Intelligent Assessment & Personalized Learning Platform
## Standalone Grounded Retrieval-Augmented Generation (RAG) Service

Production-grade, end-to-end RAG microservice developed for **MIT-WPU (Maharashtra Institute of Technology - World Peace University)** to power intelligent assessment, question generation, personalized learning, and grounded AI tutoring.

---

## 1. Key Capabilities & Architectural Highlights

- **Multi-Format Document Ingestion**:
  - **PPTX**: Native slide-level extraction, slide titles, bullet points, text boxes, tables, speaker notes, and OCR for embedded images. Slide numbers preserved across all downstream citations.
  - **PDF**: PyMuPDF extraction with selective Tesseract OCR fallback for scanned pages.
  - **DOCX**: Hierarchical section- and heading-aware extraction preserving structural context.
- **Intelligent Structure-Aware Chunking**:
  - Slide-aware for PPTX (keeps related slide items unified while retaining slide ranges).
  - Page-aware for PDF (records `page_start` and `page_end`).
  - Section-aware for DOCX.
- **Official Google Gemini Integrations**:
  - Gemini Embedding 2 / `gemini-embedding-004` (768 dimensions) with batched requests and exponential backoff.
  - `gemini-2.0-flash` generation with strict prompt anti-injection grounding rules.
- **Actual Pinecone Vector Database**:
  - Serverless index using Cosine Similarity.
  - Institutional namespace isolation (`MIT-WPU`).
  - Strict metadata filtering (`document_id`, `institution_id`).
- **Hybrid Retrieval & RRF**:
  - Combines Pinecone dense semantic retrieval with local BM25 keyword retrieval.
  - Fused via **Reciprocal Rank Fusion (RRF)** ($k=60$).
- **Lightweight Local Cross-Encoder Reranking**:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` optimized for Intel i5 and 16 GB RAM CPU inference.
- **Strict Grounding & Citation Guarantees**:
  - Verifiable source citations (`document_name`, `page_start`, `slide_start`, `chunk_ids`).
  - Mandatory `INSUFFICIENT_CONTEXT` detection for out-of-domain questions.
  - Parametric general-knowledge responses are strictly isolated and labeled separately.

---

## 2. Technology Stack

| Layer | Component | Version / Identifier |
|---|---|---|
| **Web Framework** | FastAPI + Uvicorn | `fastapi==0.115.6`, `uvicorn==0.32.1` |
| **Validation** | Pydantic v2 | `pydantic==2.10.4`, `pydantic-settings==2.7.0` |
| **Database** | SQLite + SQLAlchemy | `sqlalchemy==2.0.36` |
| **LLM Generation** | Google GenAI SDK | `google-genai==1.14.0` (`gemini-2.0-flash`) |
| **Embeddings** | Gemini Embeddings | `gemini-embedding-004` (768 dimensions) |
| **Vector DB** | Pinecone Serverless | `pinecone==5.4.2` |
| **Keyword Retrieval** | BM25 Okapi | `rank-bm25==0.2.2` |
| **Reranker** | Cross-Encoder | `sentence-transformers==3.3.1` (MiniLM) |
| **Document Parsers** | PyMuPDF, python-pptx, python-docx | `pymupdf==1.25.1`, `python-pptx==1.0.2`, `python-docx==1.1.2` |
| **OCR** | Tesseract OCR | `pytesseract==0.3.13`, `Pillow==11.1.0` |

---

## 3. Project Structure

```
rag-service/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── health.py            # System health and model info
│   │       ├── documents.py         # Upload, list, status, chunks, reprocess, delete
│   │       ├── retrieval.py         # Hybrid search endpoint with diagnostics
│   │       ├── generation.py        # /ai/ask and /generation/questions
│   │       ├── tutor.py             # /ai/tutor doubt solver
│   │       └── learning.py          # /learning/summary, /flashcards, /revision-notes
│   ├── core/
│   │   ├── config.py                # Pydantic BaseSettings with .env support
│   │   ├── security.py              # API key auth & institution scoping
│   │   ├── logging.py               # Masked secret logging
│   │   └── exceptions.py            # Typed domain exceptions
│   ├── schemas/                     # Pydantic v2 request & response models
│   ├── models/                      # SQLAlchemy database entities
│   ├── extractors/                  # Base, PDF, PPTX, and DOCX extractors
│   ├── services/                    # Ingestion, OCR, Chunking, BM25, Hybrid, Reranking, etc.
│   ├── integrations/
│   │   ├── gemini/                  # Embeddings & Generation SDK wrappers
│   │   └── pinecone/                # Serverless vector store client
│   ├── repositories/                # Document and Chunk CRUD
│   ├── workers/                     # Background job worker with retry & startup recovery
│   └── main.py                      # FastAPI application entry point
├── tests/
│   ├── unit/                        # Tests for chunking, cleaning, BM25, context builder
│   ├── security/                    # Authentication and input validation tests
│   └── integration/                 # FastAPI testclient endpoints
├── evaluation/
│   ├── sample_questions.json        # 20 curated benchmark test questions
│   └── evaluation_results/          # Evaluation reports
├── docs/
│   └── MERN_INTEGRATION.md          # REST integration contract for MERN backend
├── scripts/
│   ├── setup_windows.ps1            # Automated Windows PowerShell setup
│   └── run_evaluation.py            # Retrieval & Generation benchmark runner
├── storage/                         # Upload directory & BM25 serialized index
├── .env.example                     # Environment configuration template
├── requirements.txt                 # Exact pinned dependencies
└── Dockerfile                       # Container deployment definition
```

---

## 4. Setup & Running Instructions (Windows)

### Step 1: Install Python 3.12 & Tesseract OCR
1. Download and install **Python 3.12 (64-bit)** from [python.org](https://www.python.org/downloads/).
   - ⚠️ Check **"Add python.exe to PATH"** during installation.
2. Download and install **Tesseract OCR (Windows 64-bit)** from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
   - Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   - Ensure the **English** language pack is selected.

### Step 2: Automated PowerShell Setup
In PowerShell from the project root (`F:\RAGG\rag-service`):
```powershell
.\scripts\setup_windows.ps1
```
Or manually:
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env` and fill in credentials:
```ini
RAG_API_KEY=your_secure_random_key_here
GEMINI_API_KEY=AIzaSy...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=mit-wpu-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

### Step 4: Start the Service
```powershell
.\venv\Scripts\activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation will be available at:
`http://127.0.0.1:8000/docs`

---

## 5. Running the Test Suite
```powershell
.\venv\Scripts\activate
pytest tests/ -v
```

---

## 6. Running Benchmark Evaluation
```powershell
.\venv\Scripts\activate
python scripts/run_evaluation.py
```
This tests the 20 benchmark questions across:
- **Dense-only**
- **BM25-only**
- **Hybrid (RRF)**
- **Hybrid + Cross-Encoder Reranking**
Measuring Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG@5.
