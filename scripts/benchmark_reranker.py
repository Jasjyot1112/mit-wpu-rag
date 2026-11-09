"""
Reranker CPU Benchmark Script for Intel i5 / 16 GB RAM.
Measures model load time, reranking latency per candidate batch, and memory usage.
"""

import sys
import time
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
from app.services.reranking_service import RerankingService


def main():
    print("==========================================================")
    print("  CPU Reranker Benchmark — Intel i5 & 16 GB RAM Profile")
    print("==========================================================")

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    service = RerankingService(model_name=model_name, enabled=True, max_candidates=30)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # 1. Model Loading Benchmark
    print(f"[*] Loading model '{model_name}' on CPU...")
    t0 = time.time()
    service.load_model()
    load_time_sec = time.time() - t0
    mem_after_load = process.memory_info().rss / (1024 * 1024)
    print(f"[OK] Model loaded in: {load_time_sec:.2f} seconds")
    print(f"[OK] Memory used by model: {mem_after_load - mem_before:.2f} MB (Total process RSS: {mem_after_load:.2f} MB)")

    # 2. Reranking Latency Benchmark with typical candidate sets
    query = "How does the Bully Election Algorithm select a new coordinator?"
    candidates = [
        {
            "chunk_id": f"chk_{i}",
            "heading": f"Slide {i} - Coordinator Election",
            "text": (
                "The Bully Election Algorithm elect a coordinator process when the current coordinator crashes. "
                "The process with the highest process identifier alive is chosen as the new leader. "
                f"Candidate variation {i} notes message complexity and timeouts in distributed algorithms."
            ),
        }
        for i in range(20)
    ]

    print(f"\n[*] Benchmarking reranking over 20 passage candidates...")
    t0 = time.time()
    reranked, diag = service.rerank(query=query, candidates=candidates, top_k=6)
    rerank_time_ms = (time.time() - t0) * 1000

    print(f"[OK] Reranked 20 candidates in: {rerank_time_ms:.1f} ms ({rerank_time_ms / 20:.2f} ms per pair)")
    print(f"[OK] Returned top {len(reranked)} candidates:")
    for rank, c in enumerate(reranked[:3], 1):
        print(f"    Rank {rank} (score: {c.get('rerank_score'):.4f}): {c['chunk_id']} - {c['heading']}")

    print("\n[OK] Benchmark completed successfully.")


if __name__ == "__main__":
    main()
