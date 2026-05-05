<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" height="40"/>
  <img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white" height="40"/>
  <img src="https://img.shields.io/badge/Qdrant-FF4D4D?style=for-the-badge&logo=qdrant&logoColor=white" height="40"/>
  <img src="https://img.shields.io/badge/Sentence_Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" height="40"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chain&logoColor=white" height="40"/>
  <img src="https://img.shields.io/badge/Qwen-00BFFF?style=for-the-badge&logo=alibabacloud&logoColor=white" height="40"/>
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" height="40"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" height="40"/>
</p>

# Persian FAQ RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for Persian FAQ documents, combining semantic search with local LLM inference.

## Overview

This system provides intelligent question answering for FAQ documents using semantic search and local LLM generation. It leverages Qdrant vector database for efficient similarity search and integrates with LM Studio for running local LLMs (Qwen2.5-14B). The system includes a web-based chat interface built with Django.

## Architecture

The system implements a Retrieval-Augmented Generation (RAG) pipeline with a modular, layered architecture designed for Persian FAQ documents.

### System Layers

**1. Presentation Layer (Django Web Application)**
- Handles HTTP requests and renders the chat interface
- Provides REST API endpoint for question answering (`/ask/`)
- Manages CSRF protection and request validation

**2. Application Layer (RAG Chain)**
- Orchestrates the retrieval and generation flow
- Builds context from retrieved documents
- Constructs prompts and sends them to the LLM
- Returns formatted responses with source attribution

**3. Semantic Search Layer**
- **Embedder**: Converts text to vector embeddings using multilingual Sentence Transformers
- **Vector Store**: Qdrant database for storing and searching document vectors
- Uses cosine similarity to find the most relevant documents for each query

**4. Document Processing Layer**
- Loads FAQ data from CSV files with category, question, and answer columns
- Supports TXT and PDF formats as well
- Splits documents into chunks (configurable size and overlap)
- Enriches text with metadata for better search quality

**5. LLM Layer (LM Studio)**
- Runs Qwen2.5-14B-Instruct model locally
- Provides OpenAI-compatible API endpoint
- Generates final answers based on retrieved context
- Operates offline with no external API calls

### Component Details

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| Web Interface | Django | Renders chat UI, handles user input |
| RAG Chain | Python | Manages retrieval and generation flow |
| Embedder | Sentence Transformers | Converts text to vector embeddings |
| Vector Store | Qdrant | Stores and searches document vectors |
| Document Loader | LangChain | Reads CSV, TXT, PDF files |
| LLM Server | LM Studio | Runs Qwen2.5-14B model locally |

### Data Flow

1. **User sends question** → Django receives POST request
2. **Semantic Search** → Embedder converts question to vector → Qdrant finds similar documents
3. **Context Building** → Retrieved documents are combined into prompt
4. **LLM Generation** → Prompt sent to LM Studio (Qwen model)
5. **Response** → Answer + sources returned to user interface


### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10/11, Linux, macOS | Windows 11 / Ubuntu 22.04 |
| Python | 3.10 | 3.10 |
| RAM | 8GB | 16GB |
| Storage | 10GB | 20GB |
| GPU | No (CPU works) | NVIDIA GPU with 8GB+ VRAM |
| CUDA | - | CUDA 12.8 |

**1. Clone and create virtual environment**

```bash
git clone https://github.com/yourusername/faq-rag-system.git
cd faq-rag-system
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure environment**

Create .env and config.yaml files (see example below).

**4. Prepare data**

Place your FAQ CSV file in data/ folder with columns: category, question, answer.

**5. Run the system**

```bash
cd core
python manage.py migrate
python manage.py runserver
```
**6. Start LM Studio**

Load Qwen2.5-14B-Instruct model and start server on port 1234.

**7. Open browser**

Navigate to ```http://localhost:8000```

## Configuration Files

**.env**
```bash
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL_NAME=local-model
```

**config.yaml**
``` bash
embedding:
  model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  device: "cuda"

chunking:
  chunk_size: 2000
  chunk_overlap: 0

vector_db:
  path: "./qdrant_storage"
  collection_name: "rag_docs"

retrieval:
  top_k: 6

llm:
  base_url: "http://localhost:1234/v1"
  model_name: "local-model"
  temperature: 0.2
  max_tokens: 512
```

## CSV Format Example

The system expects a CSV file with three columns: `category`, `question`, and `answer`.

| Column | Description |
|--------|-------------|
| category | FAQ category (e.g., ودیعه رهن, ضمانت‌نامه) |
| question | Persian question text |
| answer | Persian answer text |

**Example:**

```csv
category,question,answer
ودیعه رهن,مراحل دریافت تسهیلات ودیعه رهن محل کار چگونه است؟,ابتدا متقاضی درخواست خود را ثبت می‌کند. سپس کارشناس صندوق مدارک را بررسی کرده و پس از تایید، تسهیلات پرداخت می‌شود.
ضمانت‌نامه,شرایط اخذ ضمانت‌نامه از صندوق چیست؟,شرکت دانش‌بنیان باید حداقل یک سال از تاسیس آن گذشته باشد و صورت‌های مالی معتبر ارائه دهد.
کمک بلاعوض,مبلغ کمک‌های بلاعوض چقدر است؟,حداکثر مبلغ کمک‌های بلاعوض ۲۰۰ میلیون تومان می‌باشد.
```
**Usage**
Web interface: ```http://localhost:8000```

API endpoint: ```POST /ask/``` with ```{"query": "your question"}```

