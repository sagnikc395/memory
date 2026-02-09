from pydantic import BaseModel  # noqa: E402
from datetime import date


class Memory(BaseModel):
    information: str
    predicted_categories: list[str]


class EmbeddedMemory(BaseModel):
    user_id: int
    memory_text: str
    categories: list[str]
    _date: date
    embedding: list[float]
