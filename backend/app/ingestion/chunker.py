"""Recursive token/character-based chunking with deterministic metadata and IDs."""

import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from .loader import LoadedDocument, DocumentPage


class DocumentChunk(BaseModel):
    """Represents an atomic chunk of a document with full provenance metadata."""

    chunk_id: str = Field(description="Deterministic unique ID for this chunk")
    text: str = Field(description="The textual content of the chunk")
    source: str = Field(description="Original filename")
    page: int = Field(description="1-based page or section number", default=1)
    chunk_index: int = Field(description="Zero-based index of this chunk within the document")
    section: str = Field(description="Section heading or topic", default="General")
    char_count: int = Field(description="Character length of chunk text")
    token_count: int = Field(description="Estimated token count of chunk text")
    metadata: Dict = Field(default_factory=dict, description="Additional custom metadata")


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 characters per token for English)."""
    return max(1, len(text) // 4)


def sanitize_id_part(text: str) -> str:
    """Sanitize string for safe chunk ID creation."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", text).strip("_")


def recursive_split_text(
    text: str,
    max_chars: int = 2400,  # ~600 tokens
    overlap_chars: int = 360,  # ~15% overlap
    separators: Optional[List[str]] = None,
) -> List[str]:
    """Recursively split text by natural paragraph, sentence, and word boundaries."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    if len(text) <= max_chars:
        cleaned = text.strip()
        return [cleaned] if cleaned else []

    # Find highest priority separator that exists in text
    sep_to_use = ""
    for sep in separators:
        if sep in text:
            sep_to_use = sep
            break

    if not sep_to_use:
        # Fallback to hard character slicing if no separator exists
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += max_chars - overlap_chars
        return chunks

    splits = text.split(sep_to_use)
    chunks: List[str] = []
    current_chunk = ""

    for s in splits:
        candidate = f"{current_chunk}{sep_to_use}{s}" if current_chunk else s
        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Check if single split segment exceeds max_chars itself
            if len(s) > max_chars:
                remaining_seps = separators[separators.index(sep_to_use) + 1 :]
                sub_splits = recursive_split_text(s, max_chars, overlap_chars, remaining_seps)
                chunks.extend(sub_splits)
                current_chunk = ""
            else:
                # Add overlap from tail of previous chunk if possible
                if overlap_chars > 0 and len(current_chunk) > overlap_chars:
                    overlap_seed = current_chunk[-overlap_chars:]
                    current_chunk = f"{overlap_seed}{sep_to_use}{s}"
                else:
                    current_chunk = s

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def chunk_documents(
    doc: LoadedDocument,
    max_tokens: int = 600,
    overlap_pct: float = 0.15,
) -> List[DocumentChunk]:
    """Chunk a loaded document into DocumentChunk objects with deterministic IDs."""
    max_chars = max_tokens * 4
    overlap_chars = int(max_chars * overlap_pct)
    chunks: List[DocumentChunk] = []
    
    clean_src = sanitize_id_part(doc.filename)
    overall_chunk_index = 0

    for page in doc.pages:
        raw_text = page.text.strip()
        if not raw_text:
            continue

        text_splits = recursive_split_text(
            raw_text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for split in text_splits:
            if not split.strip():
                continue
            
            # Deterministic chunk ID: {filename}_p{page}_c{index}
            chunk_id = f"{clean_src}_p{page.page_number}_c{overall_chunk_index}"
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=split,
                source=doc.filename,
                page=page.page_number,
                chunk_index=overall_chunk_index,
                section=page.section_heading,
                char_count=len(split),
                token_count=estimate_tokens(split),
                metadata={
                    "source": doc.filename,
                    "page": page.page_number,
                    "chunk_id": chunk_id,
                    "chunk_index": overall_chunk_index,
                    "section": page.section_heading,
                    "file_type": doc.file_type,
                },
            )
            chunks.append(chunk)
            overall_chunk_index += 1

    return chunks
