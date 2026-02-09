from memory import extract_memory
import asyncio


def main():
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
        extract_memory.extract_memories_from_messages(
            messages=messages, categories=existing_categories
        )
    )


if __name__ == "__main__":
    main()
