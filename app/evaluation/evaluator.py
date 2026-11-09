"""
RAG Evaluation Framework for Information Retrieval and Grounded Generation.

Computes:
- Retrieval: Recall@K, Precision@K, Mean Reciprocal Rank (MRR), nDCG@K
- Generation: Groundedness, Citation Accuracy, Insufficient-Context Detection Accuracy
"""

import math
from typing import Dict, List, Optional, Any


class RetrievalEvaluator:
    """Computes ranking and retrieval quality metrics against benchmark ground truth."""

    @staticmethod
    def is_match(retrieved_item: dict, expected_doc: Optional[str], slide_start: Optional[int], slide_end: Optional[int]) -> bool:
        """Check if a retrieved chunk matches ground truth document and slide window."""
        if not expected_doc:
            return False

        doc_name = (retrieved_item.get("document_name") or retrieved_item.get("source_filename") or "").lower()
        if expected_doc.lower() not in doc_name:
            return False

        if slide_start is not None:
            r_start = retrieved_item.get("slide_start") or retrieved_item.get("page_start")
            r_end = retrieved_item.get("slide_end") or retrieved_item.get("page_end") or r_start
            if r_start is not None and r_end is not None:
                # Check overlap between [r_start, r_end] and [slide_start, slide_end]
                target_end = slide_end if slide_end is not None else slide_start
                if not (r_end < slide_start or r_start > target_end):
                    return True
        return True

    def recall_at_k(self, retrieved: List[dict], expected_doc: Optional[str], slide_start: Optional[int], slide_end: Optional[int], k: int) -> float:
        if not expected_doc:
            return 1.0  # Out of domain query: recall not applicable / 1.0
        top_k = retrieved[:k]
        hit = any(self.is_match(item, expected_doc, slide_start, slide_end) for item in top_k)
        return 1.0 if hit else 0.0

    def precision_at_k(self, retrieved: List[dict], expected_doc: Optional[str], slide_start: Optional[int], slide_end: Optional[int], k: int) -> float:
        if not expected_doc:
            return 0.0 if retrieved[:k] else 1.0
        top_k = retrieved[:k]
        if not top_k:
            return 0.0
        hits = sum(1 for item in top_k if self.is_match(item, expected_doc, slide_start, slide_end))
        return hits / len(top_k)

    def reciprocal_rank(self, retrieved: List[dict], expected_doc: Optional[str], slide_start: Optional[int], slide_end: Optional[int]) -> float:
        if not expected_doc:
            return 0.0
        for rank, item in enumerate(retrieved, start=1):
            if self.is_match(item, expected_doc, slide_start, slide_end):
                return 1.0 / rank
        return 0.0

    def ndcg_at_k(self, retrieved: List[dict], expected_doc: Optional[str], slide_start: Optional[int], slide_end: Optional[int], k: int) -> float:
        if not expected_doc:
            return 1.0
        top_k = retrieved[:k]
        dcg = 0.0
        for rank, item in enumerate(top_k, start=1):
            rel = 1.0 if self.is_match(item, expected_doc, slide_start, slide_end) else 0.0
            dcg += rel / math.log2(rank + 1)

        # IDCG for 1 relevant item at rank 1 is 1.0 / log2(2) = 1.0
        idcg = 1.0
        return dcg / idcg


class GenerationEvaluator:
    """Evaluates answer faithfulness, citation accuracy, and insufficient-context handling."""

    @staticmethod
    def evaluate_insufficient_context_detection(predicted_status: str, is_in_domain: bool) -> bool:
        """True if the system correctly signals INSUFFICIENT_CONTEXT for out-of-domain questions."""
        if not is_in_domain:
            return predicted_status == "INSUFFICIENT_CONTEXT"
        else:
            return predicted_status in ("ANSWERED", "PARTIAL")

    @staticmethod
    def evaluate_citation_accuracy(
        citations: List[dict], expected_doc: Optional[str], slide_start: Optional[int], slide_end: Optional[int]
    ) -> float:
        """Returns 1.0 if citations accurately reference the expected document and slides."""
        if not expected_doc:
            return 1.0 if not citations else 0.0

        if not citations:
            return 0.0

        for cit in citations:
            doc_name = (cit.get("document_name") or "").lower()
            if expected_doc.lower() in doc_name:
                c_start = cit.get("slide_start") or cit.get("page_start")
                c_end = cit.get("slide_end") or cit.get("page_end") or c_start
                if slide_start is not None and c_start is not None:
                    target_end = slide_end if slide_end is not None else slide_start
                    if not (c_end < slide_start or c_start > target_end):
                        return 1.0
                else:
                    return 1.0
        return 0.0
