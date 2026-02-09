# store our embeddings into a vector database
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, models
from .config import DB_PORT


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
