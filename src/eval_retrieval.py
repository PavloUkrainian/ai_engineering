"""
Модуль оцінювання якості пошуку (Retrieval Evaluation).
Вимірює:
- Recall@1, Recall@3, Recall@5
- MRR (Mean Reciprocal Rank)
- Latency (ms)
Порівнює 3 конфігурації:
1. Dense (тільки векторний пошук у Qdrant)
2. Hybrid (BM25 + Dense через RRF)
3. Hybrid + Reranking (RRF + Cross-Scoring Reranker)
Зберігає результати у results/retrieval_benchmark.json.
"""

import json
import logging
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Додаємо корінь проєкту в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import Chunk
from src.embeddings import EmbeddingClient
from src.vector_store import QdrantVectorStore
from src.bm25_search import BM25SearchEngine
from src.retriever import HybridRetriever, SearchMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TESTSET_FILE = Path("evals/retrieval_testset.json")
CHUNKS_FILE = Path("data/chunks.jsonl")
RESULTS_DIR = Path("results")


def load_chunks() -> List[Chunk]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Файл чанків не знайдено: {CHUNKS_FILE}. Спочатку виконайте make step1-rag!")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                chunks.append(Chunk(**d))
    return chunks


def load_testset() -> List[dict]:
    if not TESTSET_FILE.exists():
        raise FileNotFoundError(f"Тестсет не знайдено: {TESTSET_FILE}")
    with open(TESTSET_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval_mode(retriever: HybridRetriever, testset: List[dict], mode: SearchMode) -> Dict[str, Any]:
    """Вимірює Recall@k та MRR для обраного режиму пошуку."""
    recalls_at_1 = []
    recalls_at_3 = []
    recalls_at_5 = []
    reciprocal_ranks = []
    latencies_ms = []

    for item in testset:
        query = item["query"]
        expected_chunk_ids = set(item["expected_chunk_ids"])
        expected_doc_id = item.get("expected_doc_id")

        t0 = time.perf_counter()
        results = retriever.search(query=query, mode=mode, top_k=5)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(lat_ms)

        retrieved_chunk_ids = [r.chunk_id for r in results]

        # 1. Recall@1
        top_1 = retrieved_chunk_ids[:1]
        recalls_at_1.append(1.0 if any(cid in expected_chunk_ids for cid in top_1) else 0.0)

        # 2. Recall@3
        top_3 = retrieved_chunk_ids[:3]
        recalls_at_3.append(1.0 if any(cid in expected_chunk_ids for cid in top_3) else 0.0)

        # 3. Recall@5
        top_5 = retrieved_chunk_ids[:5]
        recalls_at_5.append(1.0 if any(cid in expected_chunk_ids for cid in top_5) else 0.0)

        # 4. Reciprocal Rank (MRR)
        rr = 0.0
        for rank, cid in enumerate(retrieved_chunk_ids, 1):
            if cid in expected_chunk_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return {
        "mode": mode.value,
        "queries_count": len(testset),
        "recall_at_1_percent": round(float(sum(recalls_at_1) / len(recalls_at_1) * 100), 1),
        "recall_at_3_percent": round(float(sum(recalls_at_3) / len(recalls_at_3) * 100), 1),
        "recall_at_5_percent": round(float(sum(recalls_at_5) / len(recalls_at_5) * 100), 1),
        "mrr": round(float(sum(reciprocal_ranks) / len(reciprocal_ranks)), 3),
        "mean_latency_ms": round(float(sum(latencies_ms) / len(latencies_ms)), 2)
    }


def run_step2() -> Dict[str, Any]:
    """Повний прогін Кроку 2: індексація, гібридний пошук та бенчмарк."""
    logger.info("⚙️ Завантаження чанків...")
    chunks = load_chunks()
    logger.info(f"Завантажено {len(chunks)} чанків")

    logger.info("🧠 Розрахунок/отримання ембеддінгів (OpenAI / Cache)...")
    emb_client = EmbeddingClient()
    texts = [c.text for c in chunks]
    embeddings = emb_client.get_embeddings(texts)
    logger.info(f"Отримано {len(embeddings)} ембеддінгів")

    logger.info("📦 Індексація в QdrantVectorStore (:memory:)...")
    v_store = QdrantVectorStore(location=":memory:", vector_dim=len(embeddings[0]))
    v_store.index_chunks(chunks, embeddings)

    logger.info("📖 Індексація в BM25SearchEngine...")
    bm25_engine = BM25SearchEngine(chunks)

    retriever = HybridRetriever(v_store, bm25_engine, emb_client)

    logger.info("🎯 Завантаження тестового набору запитів (15 питань)...")
    testset = load_testset()

    logger.info("🚀 Запуск бенчмарку якості пошуку...")
    res_dense = evaluate_retrieval_mode(retriever, testset, SearchMode.DENSE)
    res_hybrid = evaluate_retrieval_mode(retriever, testset, SearchMode.HYBRID)
    res_rerank = evaluate_retrieval_mode(retriever, testset, SearchMode.HYBRID_RERANK)

    benchmark_results = {
        "dense": res_dense,
        "hybrid": res_hybrid,
        "hybrid_rerank": res_rerank
    }

    # Збереження результатів
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "retrieval_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    # Друк підсумкової порівняльної таблиці
    print("\n" + "=" * 95)
    print("📈 РЕЗУЛЬТАТИ ПОРІВНЯННЯ RETRIEVAL КОНФІГУРАЦІЙ (КРОК 2)")
    print("=" * 95)
    print(f"{'Конфігурація пошуку':<25} | {'Recall@1':<12} | {'Recall@3':<12} | {'Recall@5':<12} | {'MRR':<10} | {'Latency':<12}")
    print("-" * 95)
    for name, r in [
        ("1. Тільки Dense (Vector)", res_dense),
        ("2. Hybrid (BM25 + Dense)", res_hybrid),
        ("3. Hybrid + Reranking", res_rerank)
    ]:
        r1 = f"{r['recall_at_1_percent']}%"
        r3 = f"{r['recall_at_3_percent']}%"
        r5 = f"{r['recall_at_5_percent']}%"
        mrr = f"{r['mrr']:.3f}"
        lat = f"{r['mean_latency_ms']:.1f} ms"
        print(f"{name:<25} | {r1:<12} | {r3:<12} | {r5:<12} | {mrr:<10} | {lat:<12}")
    print("=" * 95 + "\n")

    logger.info(f"💾 Результати бенчмарку збережено у {out_path}")
    logger.info("🎉 Крок 2 успішно виконано!")
    return benchmark_results


if __name__ == "__main__":
    run_step2()
