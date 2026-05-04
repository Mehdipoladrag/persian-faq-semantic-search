import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from .rag_core.config import config
from .rag_core.semantic_search.embedder import Embedder
from .rag_core.semantic_search.vector_store import VectorStore
from .rag_core.semantic_search.rag_chain import RAGChain


class SimpleDocument:
    """Simple document class compatible with LangChain splitters"""
    def __init__(self, page_content: str, metadata: Dict[str, Any]) -> None:
        self.page_content = page_content
        self.metadata = metadata


class DocumentLoader:
    """Load documents from CSV, TXT and PDF files"""

    @staticmethod
    def _load_csv(file_path: Path) -> List[SimpleDocument]:
        """Load FAQ data from CSV file with category, question, answer columns"""
        documents = []
        with open(file_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                category = row.get("category", "").strip()
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                if not question or not answer:
                    continue
                text = f"Category: {category}\nQuestion: {question}\nAnswer: {answer}"
                metadata = {"source": file_path.name, "category": category, "question": question}
                documents.append(SimpleDocument(text, metadata))
        return documents

    @staticmethod
    def _load_text(file_path: Path) -> List[SimpleDocument]:
        """Load plain text file"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return [SimpleDocument(doc.page_content, doc.metadata) for doc in loader.load()]

    @staticmethod
    def _load_pdf(file_path: Path) -> List[SimpleDocument]:
        """Load PDF file"""
        loader = PyPDFLoader(str(file_path))
        return [SimpleDocument(doc.page_content, doc.metadata) for doc in loader.load()]

    @classmethod
    def load_from_directory(cls, data_dir: str = "data") -> List[SimpleDocument]:
        """Load all supported documents from directory"""
        documents = []
        base_path = Path(__file__).resolve().parent.parent
        data_path = base_path / data_dir
        if not data_path.exists():
            return documents
        for file_path in data_path.iterdir():
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".csv":
                    documents.extend(cls._load_csv(file_path))
                elif suffix == ".txt":
                    documents.extend(cls._load_text(file_path))
                elif suffix == ".pdf":
                    documents.extend(cls._load_pdf(file_path))
            except Exception as e:
                print(f"[ERROR] Failed to load {file_path.name}: {e}")
        return documents


class DocumentChunker:
    """Split documents into smaller chunks for embedding"""
    def __init__(self, chunk_size: int, chunk_overlap: int, separators: List[str]) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
        )

    def chunk_documents(self, documents: List[SimpleDocument]) -> Tuple[List[str], List[Dict]]:
        """Split documents into text chunks with metadata"""
        if not documents:
            return [], []
        # Convert to LangChain compatible format
        langchain_docs = []
        for doc in documents:
            doc_obj = type("TempDoc", (), {"page_content": doc.page_content, "metadata": doc.metadata})()
            langchain_docs.append(doc_obj)
        chunks = self._splitter.split_documents(langchain_docs)
        text_chunks = [chunk.page_content for chunk in chunks]
        metadata_list = [{"source": chunk.metadata.get("source", "unknown")} for chunk in chunks]
        return text_chunks, metadata_list


class VectorStoreManager:
    """Manage vector database operations"""
    def __init__(self, cfg: Any, embedder: Embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        self.store = VectorStore(cfg, embedder)

    def is_empty(self) -> bool:
        """Check if database has any documents"""
        if not self.store.collection_ready:
            return True
        result = self.store.client.scroll(collection_name=self.store.collection, limit=0, with_payload=False)
        count = result[1] if result[1] is not None else 0
        return count == 0

    def get_point_count(self) -> int:
        """Get number of documents in database"""
        if not self.store.collection_ready:
            return 0
        result = self.store.client.scroll(collection_name=self.store.collection, limit=0, with_payload=False)
        return result[1] if result[1] is not None else 0

    def add_documents(self, texts: List[str], metadata: List[Dict]) -> None:
        """Add document chunks to vector database"""
        self.store.add_documents(texts, metadata)


def _check_gpu_status() -> str:
    """Check if GPU is available and return status string"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return f"GPU Active: {gpu_name}"
        else:
            return "Running on CPU"
    except ImportError:
        return "PyTorch not installed - running on CPU"


class RAGInitializer:
    """Singleton RAG chain initializer"""
    _instance: Optional[RAGChain] = None

    @classmethod
    def get_chain(cls) -> RAGChain:
        """Get or create RAG chain singleton"""
        if cls._instance is None:
            cls._instance = cls._build_chain()
        return cls._instance

    @classmethod
    def _build_chain(cls) -> RAGChain:
        """Build and configure RAG chain"""
        print("[INFO] Loading RAG system...")
        print(f"[INFO] {_check_gpu_status()}")
        
        embedder = Embedder(config.embedding["model_name"], device=config.embedding["device"])
        vector_manager = VectorStoreManager(config, embedder)

        if vector_manager.is_empty():
            print("[INFO] Loading documents from data folder...")
            documents = DocumentLoader.load_from_directory("data")
            if documents:
                chunker = DocumentChunker(
                    chunk_size=config.chunking["chunk_size"],
                    chunk_overlap=config.chunking["chunk_overlap"],
                    separators=config.chunking["separators"],
                )
                texts, metadata = chunker.chunk_documents(documents)
                vector_manager.add_documents(texts, metadata)
                print(f"[INFO] Added {len(texts)} chunks to database")
            else:
                print("[WARNING] No documents found in data folder")
        else:
            print(f"[INFO] Existing documents in database: {vector_manager.get_point_count()}")

        rag_chain = RAGChain(vector_manager.store, config)
        print("[INFO] RAG system ready")
        return rag_chain


def get_rag_chain() -> RAGChain:
    """Get RAG chain instance (singleton)"""
    return RAGInitializer.get_chain()