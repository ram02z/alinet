from dataclasses import dataclass


@dataclass
class Reference:
    file_name: str
    text: str


@dataclass
class TextWithReferences:
    text: str
    ref: list[Reference] | None


@dataclass
class Question:
    text: str
    similarity_score: float
    refs: list[Reference]

@dataclass
class Document:
    name: str
    content: bytes
