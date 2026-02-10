# store our embeddings into a vector database
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, models

from memory.memory_model import EmbeddedMemory, RetrievedMemory
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


async def delete_user_records(user_id):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            )
        ),
    )


async def delete_records(point_ids):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
    )


async def fetch_all_user_records(user_id):
    out = await client.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            ]
        ),
    )

    return [convert_retrieved_records(point) for point in out.points]


def convert_retrieved_records(point) -> RetrievedMemory:
    return RetrievedMemory(
        point_id=point.id,
        user_id=point.payload["user_id"],
        memory_text=point.payload["memory_text"],
        categories=point.payload["categories"],
        _date=point.payload["date"],
        score=point.score,
    )


async def get_all_categories(user_id):
    # using the facet feature to get all the unique categories
    # from the indexed 'categories' field
    facet_filter = models.Filter(
        must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ]
    )

    # use the facet now to get unique values from the indexed field
    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=facet_filter,
        limit=1000,  # max unique categories to return
    )

    unique_categories = [hit.value for hit in facet_result.hits]

    return unique_categories

