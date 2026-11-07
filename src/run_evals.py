"""
CLI-раннер для повного прогону оцінювання (Evals Suite & Calibration) для CI та локального запуску.
Використання:
    python src/run_evals.py
    або
    make evals
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Додаємо корінь проєкту в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval_retrieval import load_chunks
from src.embeddings import EmbeddingClient
from src.vector_store import QdrantVectorStore
from src.bm25_search import BM25SearchEngine
from src.retriever import HybridRetriever
from src.rag_pipeline import RAGPipeline
from src.client import ProviderClient
from src.config import CONFIG_DETERMINISTIC
from evals.evaluator import Evaluator
from evals.judge import LLMJudge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def run_evals(ci_mode: bool = False):
    logger.info("⚙️ Ініціалізація інфраструктури для Evals...")
    chunks = load_chunks()
    emb_client = EmbeddingClient()
    embeddings = emb_client.get_embeddings([c.text for c in chunks])

    vector_store = QdrantVectorStore(location=":memory:", vector_dim=len(embeddings[0]))
    vector_store.index_chunks(chunks, embeddings)

    bm25_engine = BM25SearchEngine(chunks)
    retriever = HybridRetriever(vector_store, bm25_engine, emb_client)

    llm_client = ProviderClient()
    rag = RAGPipeline(
        retriever=retriever,
        llm_client=llm_client,
        sampling=CONFIG_DETERMINISTIC,
        token_budget=2000
    )

    judge = LLMJudge(llm_client=llm_client)

    # 1. Калібрування LLM-судді на розміченій вибірці
    print("\n" + "=" * 90)
    print("⚖️ 1. КАЛІБРУВАННЯ LLM-AS-A-JUDGE (НА РУЧНІЙ РОЗМІТЦІ)")
    print("=" * 90)
    calib_results = judge.run_calibration()
    
    calib_out_path = RESULTS_DIR / "judge_calibration.json"
    with open(calib_out_path, "w", encoding="utf-8") as f:
        json.dump(calib_results, f, ensure_ascii=False, indent=2)

    print(f"Зразків для калібрування: {calib_results['samples_count']}")
    print(f"Узгодженість із людиною (Agreement Rate): {calib_results['agreement_rate_percent']}% (допуск +/- 0.15)")
    print(f"Середня абсолютна помилка (MAE): {calib_results['mean_absolute_error']}")
    print(f"Кореляція Пірсона: {calib_results['pearson_correlation']}")
    print(f"Результати калібрування збережено у: {calib_out_path}")
    print("-" * 90)

    # 2. Повний прогін тестового набору запитів
    print("\n" + "=" * 90)
    print("🚀 2. ЗАПУСК EVALS НА ПОВНОМУ ТЕСТСЕТІ (>= 20 ПИТАНЬ)")
    print("=" * 90)
    evaluator = Evaluator(rag_pipeline=rag, judge=judge)
    summary = evaluator.run_evaluations()

    m = summary["metrics"]
    th = summary["thresholds"]
    passed = summary["passed_quality_gates"]

    print("\n" + "=" * 95)
    print("📊 ПІДСУМКОВА ТАБЛИЦЯ МЕТРИК RAG-СИСТЕМИ")
    print("=" * 95)
    print(f"{'Метрика оцінювання':<35} | {'Значення':<15} | {'CI Пороговий Gate':<20} | {'Статус':<10}")
    print("-" * 95)
    
    cit_status = "✅ PASS" if m["citation_accuracy_percent"] >= th["min_citation_accuracy"] else "❌ FAIL"
    ref_status = "✅ PASS" if m["refusal_accuracy_percent"] >= th["min_refusal_accuracy"] else "❌ FAIL"
    fth_status = "✅ PASS" if m["faithfulness_percent"] >= th["min_faithfulness"] else "❌ FAIL"
    ovr_status = "✅ PASS" if m["overall_score_percent"] >= th["min_overall_score"] else "❌ FAIL"

    val_cit = f"{m['citation_accuracy_percent']} %"
    val_facts = f"{m['key_facts_coverage_percent']} %"
    val_refusal = f"{m['refusal_accuracy_percent']} %"
    val_faith = f"{m['faithfulness_percent']} %"
    val_judge = f"{m['judge_correctness_percent']} %"
    val_overall = f"{m['overall_score_percent']} %"

    th_cit = f">= {th['min_citation_accuracy']} %"
    th_refusal = f">= {th['min_refusal_accuracy']} %"
    th_faith = f">= {th['min_faithfulness']} %"
    th_overall = f">= {th['min_overall_score']} %"

    print(f"{'Citation Accuracy (Точність джерел)':<35} | {val_cit:<15} | {th_cit:<20} | {cit_status:<10}")
    print(f"{'Key Facts Coverage (Ключові факти)':<35} | {val_facts:<15} | {'-':<20} | {'✅ PASS':<10}")
    print(f"{'Refusal Accuracy (Точність «Не знаю»)':<35} | {val_refusal:<15} | {th_refusal:<20} | {ref_status:<10}")
    print(f"{'Faithfulness (Обґрунтованість)':<35} | {val_faith:<15} | {th_faith:<20} | {fth_status:<10}")
    print(f"{'LLM Judge Correctness (Правильність)':<35} | {val_judge:<15} | {'-':<20} | {'✅ PASS':<10}")
    print(f"{'Підсумковий зважений бал (Overall)':<35} | {val_overall:<15} | {th_overall:<20} | {ovr_status:<10}")
    print("-" * 95)
    print(f"Середня затримка (Latency): {m['mean_latency_ms']} ms")
    print(f"Загальна вартість прогону: ${m['total_cost_usd']:.5f} (Розрахунково: ${m['cost_per_1000_queries_usd']:.4f} за 1000 запитів)")
    print("=" * 95 + "\n")

    if not passed:
        logger.error("❌ CI QUALITY GATES НЕ ПРОЙДЕНО! Показники нижчі встановленого порогу.")
        if ci_mode:
            sys.exit(1)
    else:
        logger.info("🎉 ВСІ CI QUALITY GATES УСПІШНО ПРОЙДЕНО!")


def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Suite")
    parser.add_argument("--ci-mode", action="store_true", help="Fail with non-zero exit code if quality gates fail")
    args = parser.parse_args()
    run_evals(ci_mode=args.ci_mode)


if __name__ == "__main__":
    main()
