
import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "customer_support_knowledge"
PERSIST_DIRECTORY = "./chroma_db"


def create_embeddings():
    """
    Configure the embedding model.
    The actual API request happens when we embed documents.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=os.getenv("OPENAI_API_KEY")
    )


def generate_chunk_ids(chunks: list[Document]) -> list[str]:
    """
    Generate deterministic IDs for document chunks.
    This helps us identify the same chunk across indexing runs.
    """
    ids = []

    for chunk in chunks:
        unique_content = (
            f"{chunk.metadata.get('source', '')}:"
            f"{chunk.metadata.get('start_index', '')}:"
            f"{chunk.page_content}"
        )

        chunk_id = hashlib.sha256(
            unique_content.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    return ids


def create_vector_store(chunks: list[Document]) -> Chroma:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing")

    embeddings = create_embeddings()
    chunk_ids = generate_chunk_ids(chunks)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        ids=chunk_ids,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY
    )

    return vector_store


def load_vector_store() -> Chroma:
    """
    Load an existing ChromaDB collection from disk.
    No document indexing is performed here.
    """
    embeddings = create_embeddings()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
