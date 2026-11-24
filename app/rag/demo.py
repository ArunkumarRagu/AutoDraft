
from app.rag.rag import load_and_split_knowledge


if __name__ == "__main__":

    chunks = load_and_split_knowledge(
        "data/knowledge_base.md"
    )

    print("Total chunks:", len(chunks))

    for index, chunk in enumerate(chunks):

        print(f"\n--- CHUNK {index + 1} ---")

        print("Metadata:", chunk.metadata)

        print("Content:")
        print(chunk.page_content)

        print("Character count:", len(chunk.page_content))
