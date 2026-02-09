import dspy
from litellm import json

from .config import LM
from dotenv import load_dotenv
import os

load_dotenv()
from pydantic import BaseModel  # noqa: E402


class Memory(BaseModel):
    information: str
    predicted_categories: list[str]


class MemoryExtracer(dspy.Signature):
    """
    Docstring for MemoryExtracer
    gets the relevant info from the conversation.
    creates memory entries that we should remember while speaking with the user later.

    will be provided with a list of the existing categories in the memory database.
    when predicting the categories of this information, we can decide to create new categories,
    or pick from an existing one if it exists.

    extract information piece by piece creating atomic units of memory.

    if transcript does not contain any information worth extracting , set no_info to True, else False.
    """

    transcript: str = dspy.InputField()
    existing_categories: list[str] = dspy.InputField()
    no_info: bool = dspy.OutputField(description="set true if no info to be extracted")
    information: list[str] = dspy.OutputField(
        description="keep the information dense, usually 3-4 words"
    )
    predicted_categories: list[Memory] = dspy.OutputField()
    memories: list[Memory] = dspy.OutputField()


memory_extractor = dspy.Predict(MemoryExtracer)


async def extract_memories_from_messages(messages, categories=[]):
    transcript = json.dumps(messages)
    lm_instance = dspy.LM(
        LM,
        api_key=os.environ.get("GROQ_API_KEY"),
        api_base="https://api.groq.com/openai/v1",
    )
    with dspy.context(lm=lm_instance):
        out = await memory_extractor.acall(
            transcript=transcript, existing_categories=categories
        )

    print(out)
    return out.values
