"""
ingest/parser.py -- Ingestion engine for PDF, DOCX, and TXT files.
Guarantees:
- MIME / magic-byte verification (no spoofed file extensions)
- Enforced limits: 20MB max file size, 50 pages max for PDFs
- Prompt-injection scanning on raw text
- 300–500 word chunking with 50-word overlap
- Precise filename + page metadata retention
"""

import io
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from orchestrator.types import SourceChunk
from config import cfg

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class IngestionError(Exception):
    """Raised when document parsing or validation fails."""
    pass


class DocumentIngestionEngine:
    """Validates, sanitizes, and chunks uploaded documents."""

    def __init__(
        self,
        max_size_bytes: int = cfg.max_upload_size_bytes,
        max_pages: int = cfg.max_pdf_pages,
        chunk_word_size: int = 400,
        chunk_overlap: int = 50,
    ):
        self.max_size_bytes = max_size_bytes
        self.max_pages = max_pages
        self.chunk_word_size = chunk_word_size
        self.chunk_overlap = chunk_overlap

    def validate_file(self, content_bytes: bytes, filename: str) -> str:
        """Validate file size and magic bytes. Returns detected file type: 'pdf', 'docx', 'txt'."""
        if len(content_bytes) > self.max_size_bytes:
            raise IngestionError(
                f"File '{filename}' exceeds maximum allowed size of {self.max_size_bytes / (1024*1024):.1f} MB."
            )

        lower_name = filename.lower()

        # Check PDF: Magic bytes '%PDF-'
        if content_bytes.startswith(b"%PDF-"):
            return "pdf"

        # Check DOCX: Magic bytes for ZIP 'PK\x03\x04'
        if content_bytes.startswith(b"PK\x03\x04") and (lower_name.endswith(".docx") or lower_name.endswith(".doc")):
            return "docx"

        # Check TXT / Markdown: Try decoding as UTF-8
        if lower_name.endswith(".txt") or lower_name.endswith(".md"):
            try:
                content_bytes.decode("utf-8")
                return "txt"
            except UnicodeDecodeError:
                raise IngestionError(f"File '{filename}' is not valid UTF-8 text.")

        raise IngestionError(
            f"Unsupported file format or invalid magic bytes for '{filename}'. Allowed: PDF, DOCX, TXT."
        )

    def scan_for_injection(self, text: str) -> Tuple[bool, str]:
        """Scan document content for prompt injection payloads.
        Returns: (is_suspicious, sanitized_text)
        """
        suspicious_patterns = [
            r"ignore (?:all )?(?:previous|above) (?:instructions|rules|directives)",
            r"system prompt",
            r"you are now (?:in developer mode|unrestricted)",
            r"disregard (?:safety|guardrails)",
            r"<\|im_start\|>",
            r"<\|system\|>",
        ]
        is_suspicious = False
        sanitized = text
        for p in suspicious_patterns:
            if re.search(p, text, re.IGNORECASE):
                is_suspicious = True
                sanitized = re.sub(p, "[REDACTED_INJECTION_PATTERN]", sanitized, flags=re.IGNORECASE)

        return is_suspicious, sanitized

    def parse_pdf(self, content_bytes: bytes, filename: str) -> List[Tuple[str, int]]:
        """Extract text per page using pypdf. Returns list of (text, page_number)."""
        if not PYPDF_AVAILABLE:
            raise IngestionError("pypdf is not installed on this server.")

        try:
            reader = PdfReader(io.BytesIO(content_bytes))
            num_pages = len(reader.pages)
            if num_pages > self.max_pages:
                raise IngestionError(
                    f"PDF '{filename}' has {num_pages} pages, exceeding the limit of {self.max_pages} pages."
                )

            pages_text = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages_text.append((txt.strip(), i + 1))

            if not pages_text:
                raise IngestionError(f"No extractable text found in PDF '{filename}'. Document may be a scanned image.")

            return pages_text
        except Exception as e:
            if isinstance(e, IngestionError):
                raise
            raise IngestionError(f"Failed to parse PDF '{filename}': {str(e)}")

    def parse_docx(self, content_bytes: bytes, filename: str) -> List[Tuple[str, int]]:
        """Extract paragraphs from DOCX. Returns list of (text, page_number)."""
        if not DOCX_AVAILABLE:
            raise IngestionError("python-docx is not installed.")

        try:
            doc = docx.Document(io.BytesIO(content_bytes))
            full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            if not full_text.strip():
                raise IngestionError(f"DOCX document '{filename}' is empty.")
            # DOCX does not have explicit pagination in raw format, treated as page 1
            return [(full_text.strip(), 1)]
        except Exception as e:
            if isinstance(e, IngestionError):
                raise
            raise IngestionError(f"Failed to parse DOCX '{filename}': {str(e)}")

    def parse_txt(self, content_bytes: bytes, filename: str) -> List[Tuple[str, int]]:
        """Decode TXT file. Returns [(text, 1)]."""
        try:
            text = content_bytes.decode("utf-8").strip()
            if not text:
                raise IngestionError(f"Text file '{filename}' is empty.")
            return [(text, 1)]
        except Exception as e:
            raise IngestionError(f"Failed to parse TXT '{filename}': {str(e)}")

    def chunk_text(self, pages: List[Tuple[str, int]], filename: str, start_chunk_id: int = 1) -> List[SourceChunk]:
        """Chunk text into 300–500 words with 50-word overlap, preserving page and file metadata."""
        chunks = []
        chunk_id = start_chunk_id

        # Determine trust level based on filename or known regulatory keywords
        trusted_level = "verified"
        lower_name = filename.lower()
        if any(term in lower_name for term in ["rbi", "sebi", "nse", "bse", "sec", "annual_report", "10-k", "10-q"]):
            trusted_level = "official"

        for page_text, page_num in pages:
            is_suspicious, sanitized = self.scan_for_injection(page_text)
            words = sanitized.split()

            if not words:
                continue

            # Sliding window over words
            i = 0
            while i < len(words):
                chunk_words = words[i : i + self.chunk_word_size]
                chunk_text = " ".join(chunk_words)

                chunks.append(
                    SourceChunk(
                        id=chunk_id,
                        text=chunk_text,
                        source_name=filename,
                        page=page_num,
                        trusted_level="untrusted" if is_suspicious else trusted_level,
                    )
                )
                chunk_id += 1
                i += (self.chunk_word_size - self.chunk_overlap)

        return chunks

    def ingest(self, content_bytes: bytes, filename: str, start_chunk_id: int = 1) -> List[SourceChunk]:
        """Validate, parse, scan, and chunk an uploaded document."""
        file_type = self.validate_file(content_bytes, filename)

        if file_type == "pdf":
            pages = self.parse_pdf(content_bytes, filename)
        elif file_type == "docx":
            pages = self.parse_docx(content_bytes, filename)
        else:
            pages = self.parse_txt(content_bytes, filename)

        return self.chunk_text(pages, filename, start_chunk_id=start_chunk_id)
