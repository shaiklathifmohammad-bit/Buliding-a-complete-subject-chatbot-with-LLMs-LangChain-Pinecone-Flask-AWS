"""
data_pipeline.py

All the logic for:
- loading PDFs from the data/ folder
- converting them into "minimal" documents (just text + source path)
- splitting into chunks
- creating the HuggingFace embedding model

Other modules (like store_index.py and rag_chat.py) should import and use
these functions instead of re-writing the same code.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import settings


# -------------------------------------------------------------------
# 1. Load PDFs from data/ → List[Document]
# -------------------------------------------------------------------
def load_pdf_files(data_dir: Path | None = None) -> List[Document]:
    """
    Load all PDFs from the given directory using LangChain's DirectoryLoader.

    Parameters
    ----------
    data_dir : Path | None
        Folder that contains your PDFs. If None, uses settings.DATA_DIR.

    Returns
    -------
    List[Document]
        One Document per PDF page.
    """
    if data_dir is None:
        data_dir = settings.DATA_DIR

    loader = DirectoryLoader(
        str(data_dir),
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    docs = loader.load()
    print(f"[data_pipeline] Loaded {len(docs)} pages from {data_dir}")
    return docs


# -------------------------------------------------------------------
# 2. Convert to "minimal" docs (text + simple 'source' metadata)
# -------------------------------------------------------------------
def to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Strip documents down to:
    - page_content (the text)
    - metadata: { "source": file_path }

    This keeps Pinecone metadata simple and consistent.
    """
    minimal: List[Document] = []

    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "unknown"
        minimal.append(
            Document(
                page_content=d.page_content,
                metadata={"source": src},
            )
        )

    print(f"[data_pipeline] Minimized: {len(minimal)} pages")
    return minimal


# -------------------------------------------------------------------
# 3. Split docs into smaller text chunks
# -------------------------------------------------------------------
def split_into_chunks(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 20,
) -> List[Document]:
    """
    Split documents into overlapping text chunks using
    RecursiveCharacterTextSplitter.

    Returns a new list of Document objects (each one is a chunk).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"[data_pipeline] Chunks created: {len(chunks)}")
    return chunks


# -------------------------------------------------------------------
# 4. Create / return the HuggingFace embedding model
# -------------------------------------------------------------------
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create the HuggingFace embedding model specified in config.EMBED_MODEL.

    This is used both when:
    - building the index (PineconeVectorStore.from_documents)
    - querying the index (retriever)
    """
    print(f"[data_pipeline] Loading embeddings model: {settings.EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)

    # Optional quick smoke test:
    test_vec = embeddings.embed_query("hello world")
    print(f"[data_pipeline] Embedding dim: {len(test_vec)} (should match {settings.PINECONE_DIMENSION})")

    return embeddings


# -------------------------------------------------------------------
# 5. Convenience: full pipeline → chunks + embeddings
# -------------------------------------------------------------------
def prepare_corpus() -> Tuple[List[Document], HuggingFaceEmbeddings]:
    """
    Convenience helper to run the *full* pipeline:

    1) Load PDFs from data/
    2) Minimize docs to text + source
    3) Split into chunks
    4) Create the embedding model

    Returns
    -------
    chunks : List[Document]
        The text chunks you will store in Pinecone.
    embeddings : HuggingFaceEmbeddings
        The embedding function you pass to PineconeVectorStore.
    """
    raw_docs = load_pdf_files()
    minimal_docs = to_minimal_docs(raw_docs)
    chunks = split_into_chunks(minimal_docs)
    embeddings = get_embedding_model()
    return chunks, embeddings