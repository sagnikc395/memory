from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL
import os
from dotenv import load_dotenv

load_dotenv()


# groq doesnt have embeddings models, using sentence-transformers
# package to generate the embeddings


async def generate_embeddings(strings: list[str]):
    print(f"\n Embedding input : {strings}")
    # out = await client.embeddings.create(
    #     input=strings,
    #     model="nomic-embed-text-v1_5",
    # )

    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL not set in config.py")

    model = SentenceTransformer(EMBEDDING_MODEL, token=os.environ.get("HF_TOKEN"))
    embeddings = model.encode(strings)
    # embeddings = np.stack([item.embedding for item in out.data])
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings
