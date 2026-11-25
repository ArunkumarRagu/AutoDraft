
from app.rag.vector_store import load_vector_store


def retrieve_knowledge(
    question: str,
    k: int = 3,
    max_distance: float | None = None
) -> list[dict]:
    """
    Retrieve the top-k chunks.

    If max_distance is supplied, discard chunks whose
    distance is greater than that value.
    """

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(
        query=question,
        k=k,
        filter={"category": "billing"}
    )

    retrieved_chunks = []

    for document, distance in results:
        distance = float(distance)

        # Lower squared-L2 distance means closer vectors.
        if max_distance is not None and distance > max_distance:
            continue

        retrieved_chunks.append({
            "content": document.page_content,
            "source": document.metadata.get("source"),
            "start_index": document.metadata.get("start_index"),
            "distance": distance
        })

    return retrieved_chunks


if __name__ == "__main__":
    question = "Do you support cryptocurrency payments?"

    results = retrieve_knowledge(
        question=question,
        k=3,
        max_distance=1.3
    )

    for result in results:
        print("Distance:", result["distance"])
        print("Content:", result["content"])
        print()
