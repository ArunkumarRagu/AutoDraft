
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_knowledge_document(file_path: str) -> list[Document]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge document not found: {path}"
        )

    content = path.read_text(encoding="utf-8")

    document = Document(
        page_content=content,
        metadata={
            "source": path.name
        }
    )

    return [document]


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 80
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    return chunks


def load_and_split_knowledge(file_path: str) -> list[Document]:

    documents = load_knowledge_document(file_path)

    chunks = split_documents(documents)

    return chunks
