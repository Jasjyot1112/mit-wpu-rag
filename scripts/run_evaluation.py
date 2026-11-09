"""
Benchmark Evaluation Runner for MIT-WPU RAG Engine.

Executes test questions from evaluation/sample_questions.json,
compares Dense vs BM25 vs Hybrid vs Hybrid+Reranker,
and records verified metrics.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure app package is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.evaluator import GenerationEvaluator, RetrievalEvaluator
from app.core.config import get_settings

logger = logging.getLogger("rag_engine.evaluation")


def run_retrieval_benchmark(questions: List[dict], app_state) -> Dict[str, Any]:
    evaluator = RetrievalEvaluator()
    configs = ["dense_only", "bm25_only", "hybrid", "hybrid_rerank"]
    results = {c: {"recall@1": [], "recall@3": [], "recall@5": [], "mrr": [], "ndcg@5": [], "latencies_ms": []} for c in configs}

    for item in questions:
        q = item["question"]
        doc = item["expected_document"]
        s_start = item["expected_slide_start"]
        s_end = item["expected_slide_end"]
        in_domain = item["is_in_domain"]

        if not in_domain:
            continue

        # Config A: Dense Only
        t0 = time.time()
        q_emb = app_state.embedding_service.embed_query(q)
        dense_matches = app_state.vector_store.query_vectors(q_emb, top_k=20)
        dense_chunks = [m["metadata"] for m in dense_matches]
        dt = (time.time() - t0) * 1000
        _record_metrics(results["dense_only"], evaluator, dense_chunks, doc, s_start, s_end, dt)

        # Config B: BM25 Only
        t0 = time.time()
        bm25_chunks = app_state.bm25_service.search(q, top_k=20)
        bt = (time.time() - t0) * 1000
        _record_metrics(results["bm25_only"], evaluator, bm25_chunks, doc, s_start, s_end, bt)

        # Config C: Hybrid (RRF)
        t0 = time.time()
        hybrid_res = app_state.hybrid_retrieval_service.retrieve(q, top_k=10)
        ht = (time.time() - t0) * 1000
        _record_metrics(results["hybrid"], evaluator, hybrid_res["candidates"], doc, s_start, s_end, ht)

        # Config D: Hybrid + Reranker
        t0 = time.time()
        reranked_chunks, _ = app_state.reranking_service.rerank(q, hybrid_res["candidates"], top_k=5)
        hrt = (time.time() - t0) * 1000
        _record_metrics(results["hybrid_rerank"], evaluator, reranked_chunks, doc, s_start, s_end, hrt)

    summary = {}
    for c in configs:
        summary[c] = {
            "avg_recall@1": round(sum(results[c]["recall@1"]) / len(results[c]["recall@1"]), 4),
            "avg_recall@3": round(sum(results[c]["recall@3"]) / len(results[c]["recall@3"]), 4),
            "avg_recall@5": round(sum(results[c]["recall@5"]) / len(results[c]["recall@5"]), 4),
            "mean_reciprocal_rank": round(sum(results[c]["mrr"]) / len(results[c]["mrr"]), 4),
            "avg_ndcg@5": round(sum(results[c]["ndcg@5"]) / len(results[c]["ndcg@5"]), 4),
            "avg_latency_ms": round(sum(results[c]["latencies_ms"]) / len(results[c]["latencies_ms"]), 2),
        }

    return summary


def _record_metrics(target_dict, evaluator, chunks, doc, s_start, s_end, latency_ms):
    target_dict["recall@1"].append(evaluator.recall_at_k(chunks, doc, s_start, s_end, 1))
    target_dict["recall@3"].append(evaluator.recall_at_k(chunks, doc, s_start, s_end, 3))
    target_dict["recall@5"].append(evaluator.recall_at_k(chunks, doc, s_start, s_end, 5))
    target_dict["mrr"].append(evaluator.reciprocal_rank(chunks, doc, s_start, s_end))
    target_dict["ndcg@5"].append(evaluator.ndcg_at_k(chunks, doc, s_start, s_end, 5))
    target_dict["latencies_ms"].append(latency_ms)


def main():
    q_file = Path("evaluation/sample_questions.json")
    if not q_file.exists():
        print(f"File not found: {q_file}")
        return

    with open(q_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} evaluation questions.")
    print("Ready to run benchmark evaluation once services are started.")


if __name__ == "__main__":
    main()
