from dataclasses import dataclass


@dataclass
class Question:
    text: str
    similarity_score: float
