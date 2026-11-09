"""
Pydantic v2 schemas for learning-utility endpoints:
  - /summary          → SummaryRequest / SummaryResponse
  - /flashcards       → FlashcardsRequest / FlashcardsResponse
  - /revision-notes   → RevisionNotesRequest / RevisionNotesResponse

MIT-WPU RAG Engine
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# CitationSource is defined in retrieval_schemas and shared across all response layers.
from app.schemas.retrieval_schemas import CitationSource  # noqa: F401 – re-exported for convenience


# ---------------------------------------------------------------------------
# Summary schemas
# ---------------------------------------------------------------------------

class SummaryRequest(BaseModel):
    """Request body for the topic-summary endpoint."""

    institution_id: str = Field(default="MIT-WPU")
    topic: str = Field(..., description="Topic to summarise, e.g. 'Operating System Scheduling Algorithms'.")
    subject: Optional[str] = Field(
        None, description="Subject area to help narrow retrieval, e.g. 'Operating Systems'."
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Restrict context retrieval to these document UUIDs. When None, all institution documents are searched.",
    )
    detail_level: str = Field(
        default="medium",
        pattern="^(brief|medium|detailed)$",
        description="Controls summary verbosity: 'brief' (~100 words), 'medium' (~250 words), 'detailed' (~500 words).",
    )


class SummaryResponse(BaseModel):
    """Response body for the topic-summary endpoint."""

    topic: str = Field(..., description="The topic that was summarised.")
    summary: str = Field(..., description="Generated summary of the topic.")
    key_points: List[str] = Field(
        ..., description="Bulleted list of the most important points extracted from the summary."
    )
    citations: List[CitationSource] = Field(
        default_factory=list,
        description="Source documents and locations that the summary was derived from.",
    )
    grounded: bool = Field(
        ...,
        description="True when the summary is based on retrieved document context.",
    )


# ---------------------------------------------------------------------------
# Flashcard schemas
# ---------------------------------------------------------------------------

class Flashcard(BaseModel):
    """A single two-sided flashcard."""

    front: str = Field(..., description="Question or term shown on the front of the card.")
    back: str = Field(..., description="Answer or definition shown on the back of the card.")
    topic: str = Field(..., description="Topic this flashcard belongs to.")
    difficulty: Optional[str] = Field(
        None, description="Difficulty rating of this card: 'easy', 'medium', or 'hard'."
    )
    source_document_name: str = Field(..., description="Human-readable name of the source document.")
    page_start: Optional[int] = Field(None, description="First page number this card's content was drawn from (1-indexed).")
    page_end: Optional[int] = Field(None, description="Last page number this card's content was drawn from (1-indexed).")
    slide_start: Optional[int] = Field(None, description="First slide number this card's content was drawn from (1-indexed).")
    slide_end: Optional[int] = Field(None, description="Last slide number this card's content was drawn from (1-indexed).")


class FlashcardsRequest(BaseModel):
    """Request body for the flashcard generation endpoint."""

    institution_id: str = Field(default="MIT-WPU")
    topic: str = Field(..., description="Topic for which flashcards should be generated.")
    subject: Optional[str] = Field(None, description="Subject area to narrow retrieval.")
    document_ids: Optional[List[str]] = Field(
        None,
        description="Restrict context to these document UUIDs.",
    )
    count: int = Field(
        default=10, ge=1, le=30, description="Number of flashcards to generate."
    )


class FlashcardsResponse(BaseModel):
    """Response body for the flashcard generation endpoint."""

    topic: str
    flashcards: List[Flashcard]
    generated_count: int = Field(..., description="Actual number of flashcards generated.")
    citations: List[CitationSource] = Field(default_factory=list)
    grounded: bool


# ---------------------------------------------------------------------------
# Revision Notes schemas
# ---------------------------------------------------------------------------

class RevisionNotesRequest(BaseModel):
    """Request body for the structured revision-notes endpoint."""

    institution_id: str = Field(default="MIT-WPU")
    topic: str = Field(..., description="Topic for which revision notes should be generated.")
    subject: Optional[str] = Field(None, description="Subject area to narrow retrieval.")
    document_ids: Optional[List[str]] = Field(
        None,
        description="Restrict context to these document UUIDs.",
    )


class RevisionSection(BaseModel):
    """A single headed section within a set of revision notes."""

    heading: str = Field(..., description="Section heading or sub-topic title.")
    content: str = Field(..., description="Prose explanation for this section.")
    definitions: Optional[List[Dict[str, str]]] = Field(
        None,
        description=(
            "List of term-definition pairs, each formatted as "
            "{\"term\": \"...\", \"definition\": \"...\"}."
        ),
    )
    examples: Optional[List[str]] = Field(
        None, description="Illustrative examples or worked problems for this section."
    )


class RevisionNotesResponse(BaseModel):
    """Response body for the structured revision-notes endpoint."""

    topic: str
    sections: List[RevisionSection] = Field(
        ..., description="Ordered list of revision sections covering the topic."
    )
    citations: List[CitationSource] = Field(default_factory=list)
    grounded: bool = Field(
        ...,
        description="True when the notes are derived from retrieved document context.",
    )
