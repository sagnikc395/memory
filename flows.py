import asyncio
from memory.extract_memory import extract_memories_from_messages
from memory.generate_embeddings import generate_embeddings


def memory_extraction_flow():
    messages = [
        {"role": "user", "content": "I like coffee"},
        {"role": "assistant", "content": "Understandable, have a great day!"},
        {
            "role": "user",
            "content": "actually, no i like matcha better, i also like ...",
        },
        {
            "role": "assistant",
            "content": "but i dont think youll like matcha, coffee is more of your taste",
        },
    ]
    existing_categories = []
    asyncio.run(
        extract_memories_from_messages(
            messages=messages, categories=existing_categories
        )
    )


def generate_embeddings_flow():
    texts = [
        "hey this is sagnik, how are you ?",
        "lowkey nowadays i have been interested more about computational biology and machine learning for biology",
    ]

    asyncio.run(generate_embeddings(texts))  # type: ignore


if __name__ == "__main__":
    # memory_extraction_flow()
    generate_embeddings_flow()
