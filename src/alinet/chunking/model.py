from dataclasses import dataclass


@dataclass
class TimeChunk:
    text: str
    start_time: float
    end_time: float


