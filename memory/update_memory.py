import dspy
from litellm import json

from memory.memory_model import MemoryWithIds, EmbeddedMemory, RetrievedMemory, Memory
from memory.generate_embeddings import generate_embeddings
from datetime import datetime

from memory.db import insert_memories, delete_records, get_all_categories
from memory.extract_memory import extract_memories_from_messages
from .config import LM
import os
from dotenv import load_dotenv

load_dotenv()


class UpdateMemorySignature(dspy.Signature):
    """
    Given the conversation between usr and assistant and some
    similar memories from the database.
    The goal is to decide how to combine the new memories into the database with the existing memories.

    Actions meaning :
    - ADD - add new memories into the database as a new memory.
    - UPDATE - update an existing memory with richer information
    - DELETE - remove memory items from the database that aren't required anymore due to new information.
    - NOOP - no need to take any action.

    If no action is required, you can finish.

    Execute on the actions.
    """

    messages: list[dict] = dspy.InputField()
    existing_memories: list[RetrievedMemory] = dspy.InputField()
    summary: str = dspy.OutputField(
        description="Summarize what you did. Explain in fewer than 10 words"
    )

    # Define tools as part of the signature
    add_memory: dspy.ToolCode = dspy.ToolCode(
        name="add_memory",
        args=[
            dspy.ToolCodeParam(name="memory_text", type=str, desc="The text of the new memory."),
            dspy.ToolCodeParam(name="categories", type=list[str], desc="A list of categories for the new memory."),
        ],
        desc="Adds a new memory to the database. Call this for new information that should be remembered.",
    )
    update: dspy.ToolCode = dspy.ToolCode(
        name="update",
        args=[
            dspy.ToolCodeParam(name="memory_id", type=int, desc="The index of the memory in existing_memories to update."),
            dspy.ToolCodeParam(name="updated_memory_text", type=str, desc="The new text for the updated memory."),
            dspy.ToolCodeParam(name="categories", type=list[str], desc="A list of categories for the updated memory."),
        ],
        desc="Updates an existing memory in the database. Call this when existing information is refined or changed.",
    )
    delete: dspy.ToolCode = dspy.ToolCode(
        name="delete",
        args=[
            dspy.ToolCodeParam(name="memory_ids", type=list[int], desc="A list of indices of memories in existing_memories to delete."),
        ],
        desc="Deletes one or more existing memories from the database. Call this when existing information is no longer relevant or has been superseded.",
    )
    noop: dspy.ToolCode = dspy.ToolCode(
        name="noop",
        args=[],
        desc="Performs no action. Call this when no changes are needed to the existing memories.",
    )


class UpdateMemory(dspy.TypedPredictor):
    def __init__(self):
        super().__init__(UpdateMemorySignature)


async def update_memories_agent(
    user_id: int, messages: list[dict], existing_memories: list[RetrievedMemory]
):
    def get_point_id_from_memory_id(memory_id):
        # This assumes memory_id is an index into existing_memories
        return existing_memories[memory_id].point_id

    async def add_memory(memory_text: str, categories: list[str]) -> str:
        """
        Docstring for add_memory
        Add the new_memory into the database. No need to pass args
        :param memory_text: Description
        :type memory_text: str
        :param categories: Description
        :type categories: list[str]
        :return: Description
        :rtype: str
        """
        print(f"Adding memory: {memory_text}")
        embeddings = await generate_embeddings([memory_text])
        await insert_memories(
            memories=[
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=memory_text,
                    categories=categories,
                    _date=datetime.now(),
                    embedding=embeddings[0],
                )
            ]
        )
        return f"Memory : '{memory_text}' was added to DB"

    async def update(memory_id: int, updated_memory_text: str, categories: list[str]):
        """
        Docstring for update

        Updating memory_id to use updated_memory_text

        :param memory_id: Description
        :type memory_id: int ; integer index of the memory to replace
        :param updated_memory_text: Description ; simple atomic fact to replace
        the old memory with the new memory
        :type updated_memory_text: str
        :param categories: Description ; use existing categories to create new ones if required.
        :type categories: list[str]
        """
        print(
            f"Memory updating \nOriginal : {existing_memories[memory_id]}\nNew Memory Text: {updated_memory_text}"
        )
        point_id_to_delete = get_point_id_from_memory_id(memory_id)
        await delete_records([point_id_to_delete])

        embeddings = await generate_embeddings([updated_memory_text])

        await insert_memories(
            memories=[
                EmbeddedMemory(
                    user_id=user_id,
                    memory_text=updated_memory_text,
                    categories=categories,
                    _date=datetime.now(),
                    embedding=embeddings[0],
                )
            ]
        )
        return f"Memory {memory_id} has been updated to {updated_memory_text}"

    async def noop():
        """
        Docstring for noop
        call this when no action is required
        """
        return "No action done"

    async def delete(memory_ids: list[int]):
        """
        Docstring for delete
        remove these memory_ids from the database
        :param memory_ids: Description
        :type memory_ids: list[int]
        """
        print("Deleting these memories")
        point_ids_to_delete = [get_point_id_from_memory_id(mid) for mid in memory_ids]
        for memory_id in memory_ids:
            print(existing_memories[memory_id].memory_text)

        await delete_records(point_ids_to_delete)
        return f"Memory {memory_ids} deleted"

    # Define tools for the DSPy module
    tools = [add_memory, update, delete, noop]

    lm_instance = dspy.settings.lm
    # Set the lm and tools context
    with dspy.context(lm=lm_instance, tools=tools):
        # The following lines are commented out as they are not used in the updated flow
        # extracted_memories = await extract_memories_from_messages(
        #     messages=messages, categories=await get_all_categories(user_id)
        # )
        # print("extracted memories are ", extracted_memories)

        update_module = UpdateMemory()
        prediction = await update_module.acall(
            messages=messages, existing_memories=existing_memories
        )

        return prediction.summary
