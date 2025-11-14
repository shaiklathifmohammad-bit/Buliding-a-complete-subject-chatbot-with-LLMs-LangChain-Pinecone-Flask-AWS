# rag_chat.py
from __future__ import annotations

import re
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .config import settings
from .data_pipeline import get_embedding_model
from .prompt import HYBRID_SYSTEM_PROMPT


# --------------------------
# Vectorstore
# --------------------------
def get_vectorstore() -> PineconeVectorStore:
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    existing = {i["name"] for i in pc.list_indexes().indexes}

    if settings.INDEX_NAME not in existing:
        raise RuntimeError(f"Index '{settings.INDEX_NAME}' not found.")

    embeddings = get_embedding_model()
    return PineconeVectorStore.from_existing_index(
        index_name=settings.INDEX_NAME,
        embedding=embeddings,
    )


def get_retriever(vs=None):
    if vs is None:
        vs = get_vectorstore()

    return vs.as_retriever(search_kwargs={"k": 3})


# --------------------------
# LLM
# --------------------------
def get_llm():
    return ChatOpenAI(
        model=settings.OPENROUTER_MODEL,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=0.5,
    )


# --------------------------
# RAG CHAIN
# --------------------------
def _format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


def get_rag_chain():
    retriever = get_retriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", HYBRID_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# --------------------------
# HYBRID ANSWER
# --------------------------
def ask_question_hybrid(question: str, rag_chain):
    """
    1. Run RAG
    2. If answer says “not enough context” → general LLM fallback
    """

    answer = rag_chain.invoke(question)

    # If RAG says no context → fallback
    bad_patterns = [
        "not enough information",
        "based on the provided context",
        "context does not include",
        "no relevant context",
    ]

    if any(p in answer.lower() for p in bad_patterns):
        llm = get_llm()
        general = llm.invoke(question).content
        return general

    return answer