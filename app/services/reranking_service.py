"""
Reranking Service using a lightweight cross-encoder.

Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
CPU-compatible, runs on Intel i5 / 16 GB RAM.

Benchmarks actual reranking performance and reports timing.
Never falsely reports reranking as enabled when it is disabled or failed.
"""
import logging
import time
from typing import List, Optional, Tuple

from app.core.logging import get_logger

logger = get_logger("reranking_service")

_MODEL_LOAD_TIMEOUT = 120  # seconds to wait for model load


class RerankingService:
    """
    Cross-encoder reranker for improving retrieval precision.

    Uses sentence-transformers CrossEncoder.
    Model is loaded lazily on first rerank() call to avoid startup delay.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = True,
        max_candidates: int = 30,
    ):
        self._model_name = model_name
        self._enabled = enabled
        self._max_candidates = max_candidates
        self._model = None
        self._model_load_time_ms: Optional[float] = None
        self._load_error: Optional[str] = None

    def is_enabled(self) -> bool:
        return self._enabled

    def is_available(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._enabled and self._model is not None and self._load_error is None

    def load_model(self) -> None:
        """
        Pre-load the cross-encoder model.
        Call this during application startup for faster first request.
        """
        if not self._enabled:
            logger.info("Reranking is disabled in configuration.")
            return

        if self._model is not None:
            return

        logger.info("Loading reranker model: %s", self._model_name)
        start = time.time()
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self._model_name,
                device="cpu",
                max_length=512,
            )
            self._model_load_time_ms = (time.time() - start) * 1000
            logger.info(
                "Reranker model loaded successfully in %.1fms (model: %s).",
                self._model_load_time_ms, self._model_name,
            )
        except ImportError:
            self._load_error = "sentence-transformers not installed"
            logger.error("Reranker unavailable: sentence-transformers not installed. Run: pip install sentence-transformers")
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("Failed to load reranker model '%s': %s", self._model_name, exc)

    def rerank(
        self,
        query: str,
        candidates: List[dict],
        top_k: Optional[int] = None,
    ) -> Tuple[List[dict], dict]:
        """
        Rerank candidate chunks using the cross-encoder.

        Args:
            query: The search query.
            candidates: List of chunk dicts with 'text' field.
            top_k: Number of top reranked results to return.

        Returns:
            Tuple of (reranked_candidates, diagnostics_dict)
            diagnostics_dict contains reranking_enabled, model_name, latency_ms, error.

        Note: If reranking is disabled or fails, returns original candidates
              and diagnostics indicating the actual state.
        """
        diagnostics = {
            "reranking_enabled": self._enabled,
            "reranking_performed": False,
            "model_name": self._model_name if self._enabled else None,
            "reranking_time_ms": None,
            "error": None,
        }

        if not self._enabled:
            diagnostics["error"] = "Reranking disabled in configuration"
            return candidates[:top_k] if top_k else candidates, diagnostics

        if not candidates:
            return [], diagnostics

        # Lazy-load model
        if self._model is None:
            self.load_model()

        if self._load_error or self._model is None:
            diagnostics["error"] = self._load_error or "Model failed to load"
            logger.warning("Reranking skipped: %s", diagnostics["error"])
            return candidates[:top_k] if top_k else candidates, diagnostics

        # Limit candidates to avoid excessive CPU load
        candidates_to_rerank = candidates[:self._max_candidates]

        # Build query-passage pairs for cross-encoder
        pairs = [(query, c.get("text", "")) for c in candidates_to_rerank]

        try:
            start = time.time()
            scores = self._model.predict(pairs, show_progress_bar=False)
            rerank_time_ms = (time.time() - start) * 1000

            # Attach rerank scores
            scored = []
            for i, candidate in enumerate(candidates_to_rerank):
                updated = dict(candidate)
                updated["rerank_score"] = float(scores[i])
                scored.append(updated)

            # Sort by rerank score descending
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)

            # Assign final ranks
            for i, item in enumerate(scored):
                item["final_rank"] = i + 1

            result = scored[:top_k] if top_k else scored

            diagnostics["reranking_performed"] = True
            diagnostics["reranking_time_ms"] = rerank_time_ms
            diagnostics["candidates_reranked"] = len(candidates_to_rerank)
            diagnostics["candidates_returned"] = len(result)

            logger.info(
                "Reranking complete: %d candidates → %d results in %.1fms.",
                len(candidates_to_rerank), len(result), rerank_time_ms,
            )
            return result, diagnostics

        except Exception as exc:
            logger.error("Reranking failed: %s. Using original order.", exc)
            diagnostics["error"] = str(exc)
            return candidates[:top_k] if top_k else candidates, diagnostics

    def benchmark(self, query: str, num_candidates: int = 20, text_length: int = 500) -> dict:
        """
        Benchmark reranking performance on this hardware.
        Returns timing and resource metrics.
        """
        import os
        import random
        import string

        logger.info("Starting reranker benchmark: %d candidates, %d chars each.", num_candidates, text_length)

        # Generate synthetic candidates
        candidates = [
            {
                "chunk_id": f"bench_{i}",
                "text": " ".join(
                    "".join(random.choices(string.ascii_lowercase, k=5))
                    for _ in range(text_length // 5)
                ),
            }
            for i in range(num_candidates)
        ]

        # Load model if needed
        if self._model is None:
            self.load_model()

        if self._model is None:
            return {"error": "Model could not be loaded", "reranking_available": False}

        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss / 1024 / 1024  # MB
        except Exception:
            mem_before = None

        _, diagnostics = self.rerank(query, candidates, top_k=num_candidates)

        try:
            mem_after = proc.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
        except Exception:
            mem_used = None

        return {
            "reranking_available": True,
            "model_name": self._model_name,
            "model_load_time_ms": self._model_load_time_ms,
            "reranking_time_ms": diagnostics.get("reranking_time_ms"),
            "num_candidates": num_candidates,
            "memory_used_mb": mem_used,
            "device": "cpu",
        }
