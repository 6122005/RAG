"""Document loaders for PDF, Markdown, and TXT files with metadata extraction."""

import re
from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader


class DocumentPage:
    """Represents an extracted page or major section of a loaded document."""

    def __init__(self, text: str, page_number: int = 1, section_heading: str = "General"):
        self.text = text
        self.page_number = page_number
        self.section_heading = section_heading

    def __repr__(self) -> str:
        return f"<DocumentPage page={self.page_number} section='{self.section_heading}' len={len(self.text)}>"


class LoadedDocument:
    """Container for document content and metadata."""

    def __init__(self, filename: str, pages: List[DocumentPage], file_type: str):
        self.filename = filename
        self.pages = pages
        self.file_type = file_type


def detect_markdown_heading(text: str) -> str:
    """Extract the first prominent markdown heading if available."""
    match = re.search(r"^(?:#{1,4})\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "General"


def load_pdf(file_path: Path) -> LoadedDocument:
    """Extract text from PDF page by page using pypdf."""
    reader = PdfReader(str(file_path))
    pages: List[DocumentPage] = []
    
    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        clean_text = raw_text.strip()
        if not clean_text:
            continue
        
        # Check first non-empty lines for heading
        first_lines = clean_text.split("\n", 3)
        heading = first_lines[0].strip()[:60] if first_lines else f"Page {idx + 1}"
        pages.append(DocumentPage(text=clean_text, page_number=idx + 1, section_heading=heading))
        
    if not pages:
        pages.append(DocumentPage(text="[Empty PDF Document]", page_number=1, section_heading="Empty"))
        
    return LoadedDocument(filename=file_path.name, pages=pages, file_type="pdf")


def load_text_or_markdown(file_path: Path) -> LoadedDocument:
    """Extract text from TXT or MD files, splitting by top-level markdown headings if present."""
    content = file_path.read_text(encoding="utf-8", errors="replace")
    clean_content = content.strip()
    ext = file_path.suffix.lower()
    
    if ext == ".md" and "# " in clean_content:
        # Split into logical sections by major heading
        sections = re.split(r"(?=(?:^|\n)#{1,3}\s+)", clean_content)
        pages: List[DocumentPage] = []
        page_num = 1
        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue
            heading = detect_markdown_heading(sec_clean)
            pages.append(DocumentPage(text=sec_clean, page_number=page_num, section_heading=heading))
            page_num += 1
        return LoadedDocument(filename=file_path.name, pages=pages, file_type=ext.lstrip("."))
    
    # Standard single-page text document
    heading = clean_content.split("\n", 1)[0].strip()[:60] if clean_content else "General"
    page = DocumentPage(text=clean_content, page_number=1, section_heading=heading)
    return LoadedDocument(filename=file_path.name, pages=[page], file_type=ext.lstrip(".") or "txt")


def load_document(file_path: Path) -> LoadedDocument:
    """Load document based on file extension (.pdf, .md, .txt)."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in [".md", ".markdown", ".txt", ".rst"]:
        return load_text_or_markdown(file_path)
    else:
        # Default fallback to plain text reader
        return load_text_or_markdown(file_path)
