import groq
import os
from dotenv import load_dotenv

load_dotenv()

# since openai has no free api keys left,
# using groqs api keys
client = groq.Groq(api_key=os.environ.get("GROQ_API_KEY"))


messages = [
    {
        "role": "system",
        "content": "be maximally truth seeking and answer without bias",
    },
]

print("--- Chat Bot Initialized (Type 'exit' to quit) ---")

completion = client.chat.completions.create(
    model="qwen/qwen3-32b",
    messages=[{"role": "user", "content": ""}],
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    reasoning_effort="default",
    stream=True,
    stop=None,
)


while True:
    user_input = input("User: ")

    # Simple exit condition
    if user_input.lower() in ["exit", "quit"]:
        break

    # Append user message to history
    messages.append({"role": "user", "content": user_input})

    try:
        # Create the completion request
        # Note: using 'llama-3.3-70b-versatile' as a standard Groq model
        response = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=messages,  # type: ignore
            temperature=0.6,
            max_tokens=4096,
            top_p=0.95,
            stream=False,  # Set to True if you want to loop through chunks
        )

        # Extract and print the response
        answer = response.choices[0].message.content
        print(f"\nAssistant: {answer}\n")

        # Append assistant response to history to maintain context
        messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        print(f"An error occurred: {e}")
