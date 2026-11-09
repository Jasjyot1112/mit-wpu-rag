# MIT-WPU RAG Engine — MERN Integration Contract

## 1. Overview
The MIT-WPU RAG Engine is an autonomous, standalone microservice providing information retrieval, citation grounding, and LLM-assisted educational intelligence.

The main MERN application (MongoDB, Express, React, Node.js) delegates all document understanding, semantic search, and grounded AI generation tasks to this service via HTTP REST APIs.

---

## 2. Authentication & Headers
All requests from the Express backend to the RAG service must include:

```http
X-API-Key: <CONFIGURED_RAG_API_KEY>
Content-Type: application/json
```

---

## 3. Key Integration Workflows

### Workflow A: Teacher Course Material Upload
When a teacher uploads a PowerPoint presentation (`.pptx`), PDF (`.pdf`), or lecture document (`.docx`):

- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/documents/upload`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `file`: binary file payload
  - `institution_id`: `"MIT-WPU"`

**Response (HTTP 202 Accepted)**:
```json
{
  "id": "c7a8b4e2-8b9a-4d2b-9e12-3f8d9a2c1b4e",
  "filename": "Distributed Systems Unit 2.pptx",
  "processing_status": "PENDING",
  "job_id": "job_01e9d8c7",
  "message": "Document uploaded successfully and queued for ingestion."
}
```

The MERN backend stores `rag_document_id = id` in its MongoDB `CourseMaterial` schema.

---

### Workflow B: Grounded MCQ Generation for Teacher Approval
When a teacher selects a Subject and Topic and requests 10 question drafts:

- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/generation/questions`
- **Payload**:
```json
{
  "institution_id": "MIT-WPU",
  "subject": "Distributed Systems",
  "topic": "Bully Election Algorithm",
  "document_ids": ["c7a8b4e2-8b9a-4d2b-9e12-3f8d9a2c1b4e"],
  "easy_count": 3,
  "medium_count": 4,
  "hard_count": 3
}
```

**Response (HTTP 200 OK)**:
```json
{
  "subject": "Distributed Systems",
  "topic": "Bully Election Algorithm",
  "requested_count": 10,
  "generated_count": 10,
  "questions": [
    {
      "question_text": "In the Bully Election Algorithm, which process becomes the coordinator upon leader failure?",
      "options": [
        {"key": "A", "text": "The process with the highest process ID that is currently alive"},
        {"key": "B", "text": "The process that first detects the failure"},
        {"key": "C", "text": "The process with the lowest process ID"},
        {"key": "D", "text": "The predecessor process in the ring"}
      ],
      "correct_answer": "A",
      "explanation": "According to Slide 18 of Distributed Systems Unit 2.pptx, the Bully algorithm ensures that the surviving process with the highest ID assumes coordinator role.",
      "topic": "Bully Election Algorithm",
      "difficulty": "easy",
      "source_document_id": "c7a8b4e2-8b9a-4d2b-9e12-3f8d9a2c1b4e",
      "source_document_name": "Distributed Systems Unit 2.pptx",
      "source_slides": [18],
      "supporting_chunk_ids": ["chk_7812"]
    }
  ],
  "citations": [
    {
      "document_id": "c7a8b4e2-8b9a-4d2b-9e12-3f8d9a2c1b4e",
      "document_name": "Distributed Systems Unit 2.pptx",
      "source_type": "PPTX",
      "slide_start": 18,
      "slide_end": 19,
      "chunk_ids": ["chk_7812"]
    }
  ]
}
```

> [!IMPORTANT]
> The MERN application holds these questions in a `DRAFT` state for teacher review. RAG never approves questions or modifies exam schedules directly.

---

### Workflow C: Student AI Tutor & Doubt Solver
When a student asks for concept clarification:

- **Method**: `POST`
- **URL**: `http://localhost:8000/api/v1/ai/tutor`
- **Payload**:
```json
{
  "question": "How does Lamport Logical Clocks handle message receiving?",
  "institution_id": "MIT-WPU",
  "subject": "Distributed Systems",
  "topic": "Logical Clocks",
  "student_level": "undergraduate",
  "request_examples": true
}
```

**Response (HTTP 200 OK)**:
```json
{
  "status": "ANSWERED",
  "grounded": true,
  "explanation": "When a process receives a message with timestamp Tm, it advances its local clock Ci according to Ci = max(Ci, Tm) + 1...",
  "citations": [
    {
      "document_id": "doc_456",
      "document_name": "Distributed Systems Unit 2.pptx",
      "source_type": "PPTX",
      "slide_start": 25,
      "slide_end": 26,
      "chunk_ids": ["chk_9910"]
    }
  ]
}
```

---

### Workflow D: Personalized Revision Notes & Flashcards
When the MERN analytics engine detects a student has a weak topic (e.g., `Bully Algorithm`):

- **URL**: `POST http://localhost:8000/api/v1/learning/flashcards`
- **URL**: `POST http://localhost:8000/api/v1/learning/revision-notes`
- Generates tailored revision aids for that specific weak topic based directly on their teacher's authorized slides.
