import dspy
from litellm import json

from .constants import LM
from dotenv import load_dotenv
import os

load_dotenv()


class MemoryExtracer(dspy.Signature):
    """
    Docstring for MemoryExtracer
    gets the relevant info from the conversation.
    creates memory entries that we should remember while speaking with the user later.
    """

    transcript: str = dspy.InputField()
    information: str = dspy.OutputField(description="usually 3-4 words")


memory_extractor = dspy.Predict(MemoryExtracer)


async def extract_memories_from_messages(messages):
    transcript = json.dumps(messages)
    lm_instance = dspy.LM(
        LM,
        api_key=os.environ.get("GROQ_API_KEY"),
        api_base="https://api.groq.com/openai/v1",
    )
    with dspy.context(lm=lm_instance):
        out = await memory_extractor.acall(transcript=transcript)

    print(out)
