from typing import Optional, List

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import fitz as pymupdf
import torch


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: Optional[str] = None

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    # PyMuPDF
    TEXT_FLAGS: int = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES
    # Layout model
    BAD_SPAN_TYPES: List[str] = [
        "Caption",
        "Footnote",
        "Formula",
        "Page-footer",
        "Page-header",
        "Picture",
        "List-item",
        "Table",
    ]
    LAYOUT_MODEL_MAX: int = 512
    LAYOUT_CHUNK_OVERLAP: int = 64
    LAYOUT_DPI: int = 96
    LAYOUT_MODEL_NAME: str = "vikp/layout_segmenter"
    LAYOUT_BATCH_SIZE: int = 8 # Max 512 tokens means high batch size

    # Ordering model
    ORDERER_BATCH_SIZE: int = 32 # This can be high, because max token count is 128
    ORDERER_MODEL_NAME: str = "vikp/column_detector"

    @computed_field
    @property
    def CUDA(self) -> bool:
        return "cuda" in self.TORCH_DEVICE

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cuda":
            return torch.bfloat16
        else:
            return torch.float32

    @computed_field
    @property
    def TEXIFY_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
