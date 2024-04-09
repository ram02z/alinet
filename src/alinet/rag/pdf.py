from dataclasses import dataclass
import fitz

from marker.convert import convert_to_text_blocks
from marker.ordering import load_ordering_model
from marker.segmentation import load_layout_model


@dataclass
class DocumentSection:
    heading: str
    text: str


# TODO: move this to initialisation of server
LAYOUT_MODEL = load_layout_model()
ORDER_MODEL = load_ordering_model()

SECTIONS_TO_REMOVE = ["references", "acknowledgements", "bibliography", "appendix"]


def get_text_sections(doc: fitz.Document) -> list[str]:
    text_blocks = convert_to_text_blocks(doc, LAYOUT_MODEL, ORDER_MODEL)

    sections = []
    current_heading = None
    section_text = ""
    for block in text_blocks:
        block_type = block.block_type
        if block_type in ["Title", "Section-header"]:
            if current_heading and section_text:
                sections.append(
                    DocumentSection(heading=current_heading, text=section_text)
                )
                section_text = ""
            current_heading = block.text
        section_text += block.text

    if current_heading and section_text:
        sections.append(DocumentSection(heading=current_heading, text=section_text))

    filtered_sections = [
        section.text
        for section in sections
        if section.heading.strip().lower() not in SECTIONS_TO_REMOVE
    ]

    return filtered_sections
