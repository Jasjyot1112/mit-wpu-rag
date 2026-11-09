"""
Pydantic v2 schemas for LLM-powered generation endpoints:
  - /ask            → AskRequest / AskResponse
  - /tutor          → TutorRequest / TutorResponse
  - /generate-mcqs  → QuestionGenerationRequest / QuestionGenerationResponse

MIT-WPU RAG Engine
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# CitationSource is shared across generation and retrieval layers.
from app.schemas.retrieval_schemas import CitationSource  # noqa: F401 – re-exported for convenience


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AnswerStatus(str, Enum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    PARTIAL = "PARTIAL"


# ---------------------------------------------------------------------------
# Ask (Q&A) schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Request body for the grounded Q&A endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The student's question in natural language.",
    )
    institution_id: str = Field(
        default="MIT-WPU",
        description="Institution namespace used to scope document retrieval.",
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Restrict retrieval to these document UUIDs. When None, all institution documents are searched.",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Identifier for an ongoing conversation session. Pass this back to continue a thread.",
    )
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description=(
            "Prior turns of the conversation in [{\"role\": \"user|assistant\", \"content\": \"...\"}] format. "
            "Provide at most the last 10 turns to stay within context limits."
        ),
    )
    include_general_knowledge: bool = Field(
        default=False,
        description=(
            "When True and the retrieved context is insufficient, the LLM may fall back to its "
            "parametric knowledge and populate general_knowledge_answer."
        ),
    )
    student_level: str = Field(
        default="undergraduate",
        pattern="^(school|undergraduate|postgraduate|professional)$",
        description="Academic level of the student; controls answer vocabulary and depth.",
    )


class AskResponse(BaseModel):
    """Response body for the grounded Q&A endpoint."""

    status: AnswerStatus = Field(..., description="Outcome of the generation attempt.")
    grounded: bool = Field(
        ...,
        description="True when the answer is derived from retrieved document context.",
    )
    answer: Optional[str] = Field(
        None,
        description="Generated answer grounded in the retrieved documents.",
    )
    general_knowledge_answer: Optional[str] = Field(
        None,
        description=(
            "Parametric-knowledge fallback answer. Only populated when include_general_knowledge=True "
            "and status is INSUFFICIENT_CONTEXT."
        ),
    )
    citations: List[CitationSource] = Field(
        default_factory=list,
        description="Source documents and locations that support the answer.",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Conversation session identifier (created or echoed back for continuity).",
    )
    insufficient_context_message: Optional[str] = Field(
        None,
        description="Explanation returned when status is INSUFFICIENT_CONTEXT.",
    )
    retrieved_chunks_count: int = Field(
        default=0,
        description="Number of chunks retrieved from the vector store for this question.",
    )


# ---------------------------------------------------------------------------
# Tutor schemas
# ---------------------------------------------------------------------------

class TutorRequest(BaseModel):
    """Request body for the AI-tutor explanation endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The concept or question the student wants explained.",
    )
    institution_id: str = Field(default="MIT-WPU")
    subject: Optional[str] = Field(
        None, description="Subject area, e.g. 'Data Structures', 'Thermodynamics'."
    )
    topic: Optional[str] = Field(
        None, description="Specific topic within the subject, e.g. 'Binary Search Trees'."
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Restrict context retrieval to these document UUIDs.",
    )
    conversation_id: Optional[str] = Field(None)
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Prior conversation turns in [{\"role\": ..., \"content\": ...}] format.",
    )
    student_level: str = Field(
        default="undergraduate",
        description="Academic level of the student.",
    )
    request_examples: bool = Field(
        default=True,
        description="When True, ask the LLM to include worked examples in the explanation.",
    )


class TutorResponse(BaseModel):
    """Response body for the AI-tutor explanation endpoint."""

    status: AnswerStatus
    grounded: bool
    explanation: Optional[str] = Field(
        None, description="Step-by-step or conceptual explanation of the topic."
    )
    examples: Optional[List[str]] = Field(
        None, description="Worked examples or illustrative scenarios (when request_examples=True)."
    )
    key_concepts: Optional[List[str]] = Field(
        None, description="Bullet-point list of key concepts distilled from the explanation."
    )
    citations: List[CitationSource] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    insufficient_context_message: Optional[str] = None


# ---------------------------------------------------------------------------
# MCQ Generation schemas
# ---------------------------------------------------------------------------

class MCQOption(BaseModel):
    """A single multiple-choice option."""

    key: str = Field(..., description="Option label: A, B, C, or D.")
    text: str = Field(..., description="Text of the option.")


class GeneratedQuestion(BaseModel):
    """A fully-formed multiple-choice question produced by the LLM."""

    question_text: str = Field(..., description="The question stem.")
    options: List[MCQOption] = Field(..., description="Exactly four options labelled A–D.")
    correct_answer: str = Field(
        ..., description="Key of the correct option: A, B, C, or D."
    )
    explanation: str = Field(
        ..., description="Why the correct answer is correct, grounded in the source material."
    )
    topic: str = Field(..., description="Topic this question tests.")
    difficulty: str = Field(
        ..., description="Difficulty band: 'easy', 'medium', or 'hard'."
    )
    source_document_id: str = Field(..., description="UUID of the primary source document.")
    source_document_name: str = Field(..., description="Human-readable name of the source document.")
    source_pages: Optional[List[int]] = Field(
        None, description="1-indexed page numbers from which content was drawn (PDF/DOCX)."
    )
    source_slides: Optional[List[int]] = Field(
        None, description="1-indexed slide numbers from which content was drawn (PPTX)."
    )
    supporting_chunk_ids: List[str] = Field(
        ..., description="UUIDs of the chunks that provided content for this question."
    )


class QuestionGenerationRequest(BaseModel):
    """Request body for the MCQ generation endpoint."""

    institution_id: str = Field(default="MIT-WPU")
    subject: str = Field(..., description="Subject area for which questions are generated.")
    topic: str = Field(..., description="Specific topic within the subject.")
    document_ids: Optional[List[str]] = Field(
        None,
        description="Restrict context to these document UUIDs. When None, all institution documents are searched.",
    )
    easy_count: int = Field(
        default=3, ge=0, le=10, description="Number of easy-difficulty questions to generate."
    )
    medium_count: int = Field(
        default=4, ge=0, le=10, description="Number of medium-difficulty questions to generate."
    )
    hard_count: int = Field(
        default=3, ge=0, le=10, description="Number of hard-difficulty questions to generate."
    )


class QuestionGenerationResponse(BaseModel):
    """Response body for the MCQ generation endpoint."""

    subject: str
    topic: str
    requested_count: int = Field(
        ..., description="Total number of questions that were requested (easy + medium + hard)."
    )
    generated_count: int = Field(
        ..., description="Actual number of questions successfully generated."
    )
    questions: List[GeneratedQuestion]
    warning: Optional[str] = Field(
        None,
        description="Present when fewer questions than requested could be generated, e.g. due to limited context.",
    )
    citations: List[CitationSource] = Field(default_factory=list)
