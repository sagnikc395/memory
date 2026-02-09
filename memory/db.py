# store our embeddings into a vector database
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, models

from memory.memory_model import EmbeddedMemory
from .config import DB_PORT, COLLECTION_NAME
from uuid import uuid4
from typing import Optional


client = AsyncQdrantClient(url=f"http://localhost:{DB_PORT}")


async def create_memory_collection():
    # first check if the db even exists or not
    if not (await client.collection_exists("memories")):
        await client.create_collection(
            collection_name="memories",
            vectors_config=VectorParams(size=128, distance=Distance.DOT),
        )

        # user id should be of type UUID
        await client.create_payload_index(
            collection_name="memories",
            field_name="user_id",
            field_schema=models.PayloadSchemaType.UUID,
        )

        # indexing on the categories field
        await client.create_payload_index(
            collection_name="memories",
            field_name="categories",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        print("Collection has been created !")
    else:
        print("Collection already exists !")


async def insert_memories(memories: list[EmbeddedMemory]):
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=uuid4().hex,
                payload={
                    "user_id": memory.user_id,
                    "categories": memory.categories,
                    "memory_text": memory.memory_text,
                    "_date": memory._date,
                },
                vector=memory.embedding,
            )
            for memory in memories
        ],
    )


async def search_memories(
    search_vector: list[float], user_id: int, categories: Optional[list[str]] = None
):
    must_condition: list[models.Condition] = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]

    if categories:
        must_condition.append(
            models.FieldCondition(
                key="categories", match=models.MatchAny(any=categories)
            )
        )

    outs = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        with_payload=True,
        query_filter=models.Filter(must=must_condition),
        score_threshold=0.1,
        limit=4,
    )

    return outs.points
