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
    ]
    asyncio.run(extract_memory.extract_memories_from_messages(messages=messages))


if __name__ == "__main__":
    main()
