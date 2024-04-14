import fitz as pymupdf

from marker.cleaners.headers import filter_header_footer, filter_common_titles
from marker.extract_text import get_single_page_blocks
from marker.ordering import order_blocks
from marker.segmentation import detect_document_block_types
from marker.markdown import merge_spans, merge_lines
from marker.schema import FullyMergedBlock, Page, BlockType
from typing import List
from marker.settings import settings


def annotate_spans(blocks: List[Page], block_types: List[BlockType]):
    for i, page in enumerate(blocks):
        page_block_types = block_types[i]
        page.add_block_types(page_block_types)



def convert_to_text_blocks(
    doc: pymupdf.Document, layoutlm_model, order_model
) -> List[FullyMergedBlock]:
    blocks = []
    for pnum in range(len(doc)):
        page_blocks = get_single_page_blocks(doc, pnum)
        page_bbox = doc[pnum].bound()
        page_obj = Page(blocks=page_blocks, pnum=pnum, bbox=page_bbox)
        blocks.append(page_obj)

    block_types = detect_document_block_types(
        doc, blocks, layoutlm_model, batch_size=settings.LAYOUT_BATCH_SIZE
    )

    # Find headers and footers
    bad_span_ids = filter_header_footer(blocks)

    annotate_spans(blocks, block_types)
    blocks = order_blocks(
        doc,
        blocks,
        order_model,
        batch_size=settings.ORDERER_BATCH_SIZE,
    )

    for page in blocks:
        for block in page.blocks:
            block.filter_spans(bad_span_ids)
            block.filter_bad_span_types()

    # Copy to avoid changing original data
    merged_lines = merge_spans(blocks)
    text_blocks = merge_lines(merged_lines, blocks)
    text_blocks = filter_common_titles(text_blocks)

    return text_blocks


