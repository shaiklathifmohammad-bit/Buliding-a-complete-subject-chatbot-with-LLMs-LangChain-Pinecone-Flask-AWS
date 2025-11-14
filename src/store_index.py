"""
store_index.py

Build / update the Pinecone index from your PDFs.

This script:
- loads PDFs from the data/ folder
- converts them into text chunks
- embeds them using HuggingFace
- uploads everything into a Pinecone serverless index

Run this from the project root:

    conda activate subjectbot
    python -m src.store_index
"""

from __future__ import annotations

from typing import List

from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from .config import settings
from .data_pipeline import prepare_corpus


# ---------------------------------------------------
# 1. Create / connect Pinecone index
# ---------------------------------------------------
def ensure_pinecone_index(pc: Pinecone) -> None:
    """
    Create the Pinecone index if it doesn't exist yet.

    Uses settings from config.py:
        - settings.INDEX_NAME
        - settings.PINECONE_DIMENSION
        - settings.PINECONE_METRIC
        - settings.PINECONE_CLOUD
        - settings.PINECONE_REGION
    """
    index_name = settings.INDEX_NAME

    # list_indexes() returns an object with .indexes (SDK v5+)
    existing = {idx["name"] for idx in pc.list_indexes().indexes}
    if index_name not in existing:
        print(f"[store_index] Creating Pinecone index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=settings.PINECONE_DIMENSION,
            metric=settings.PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=settings.PINECONE_CLOUD,
                region=settings.PINECONE_REGION,
            ),
        )
        print(f"[store_index] ✅ Created index: {index_name}")
    else:
        print(f"[store_index] Using existing index: {index_name}")


# ---------------------------------------------------
# 2. Upsert chunks into Pinecone
# ---------------------------------------------------
def upsert_chunks(
    chunks: List[Document],
    embeddings,
    pc: Pinecone,
) -> None:
    """
    Given a list of text chunks and an embedding model:
    - connects to the Pinecone index
    - uses LangChain's PineconeVectorStore to upsert all chunks
    """
    index_name = settings.INDEX_NAME

    # Just to be explicit – connect to the index with low-level client
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"[store_index] Connected to index '{index_name}', stats: {stats.get('total_vector_count', 0)} vectors currently.")

    # Now upsert via LangChain adapter
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print(f"[store_index] ✅ Upserted {len(chunks)} chunks into index '{index_name}'.")


# ---------------------------------------------------
# 3. Main entry point
# ---------------------------------------------------
def main():
    # Pinecone client from config settings
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    print("[store_index] Pinecone client initialised.")

    # Ensure index exists
    ensure_pinecone_index(pc)

    # Prepare data: PDFs → minimal docs → chunks + embeddings
    chunks, embeddings = prepare_corpus()
    print(f"[store_index] Corpus ready: {len(chunks)} chunks.")

    # Upsert into Pinecone
    upsert_chunks(chunks, embeddings, pc)

    print("[store_index] ✅ Done. Index is ready for querying.")


if __name__ == "__main__":
    main()