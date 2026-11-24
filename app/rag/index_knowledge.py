
from app.rag.rag import load_and_split_knowledge
from app.rag.vector_store import create_vector_store


if __name__ == "__main__":
    print("Loading and splitting knowledge document...")

    chunks = load_and_split_knowledge(
        "data/knowledge_base.md"
    )

    print(f"Total chunks: {len(chunks)}")

    print("\nGenerating embeddings and indexing in ChromaDB...")

    vector_store = create_vector_store(chunks)

    print("\nKnowledge indexing completed!")
    print("Vector database location: ./chroma_db")
