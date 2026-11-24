
from app.rag.vector_store import load_vector_store


def retrieve_knowledge(
    question: str,
    k: int = 3
) -> list[dict]:
    """
    Retrieve the top-k most relevant knowledge chunks.
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=question,
        k=k
    )

    retrieved_chunks = []

    for document, distance in results:
        retrieved_chunks.append({
            "content": document.page_content,
            "source": document.metadata.get("source"),
            "start_index": document.metadata.get("start_index"),
            "distance": float(distance)
        })

    return retrieved_chunks


if __name__ == "__main__":
    question = "Can I pay using UPI?"

    results = retrieve_knowledge(question, k=3)

    print("\nCustomer question:", question)
    print("\n--- RETRIEVED KNOWLEDGE ---")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Start index:", result["start_index"])
        print("Distance:", result["distance"])
        print("Content:", result["content"])
