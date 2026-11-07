"""
Головний модуль оркестрації оцінювання RAG-системи (Evals Suite).
Обчислює детерміновані метрики, Faithfulness, оцінки LLM-as-a-Judge та перевіряє Quality Gates для CI.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.rag_pipeline import RAGPipeline
from src.retriever import SearchMode
from evals.scorers import CitationScorer, KeyFactsScorer, RefusalScorer, FaithfulnessScorer
from evals.judge import LLMJudge

logger = logging.getLogger(__name__)

TESTSET_PATH = Path("evals/testset.json")
RESULTS_DIR = Path("results")

# Пороги проходження Quality Gates для CI:
MIN_CITATION_ACCURACY = 80.0
MIN_REFUSAL_ACCURACY = 90.0
MIN_FAITHFULNESS = 80.0
MIN_OVERALL_SCORE = 80.0


class Evaluator:
    """Оркестратор тестування та оцінювання RAG-пайплайну."""

    def __init__(self, rag_pipeline: RAGPipeline, judge: Optional[LLMJudge] = None):
        self.rag = rag_pipeline
        self.judge = judge or LLMJudge(llm_client=rag_pipeline.llm_client)

    def run_evaluations(self, testset_path: Path = TESTSET_PATH) -> Dict[str, Any]:
        """Запускає оцінку всієї вибірки тестсету."""
        if not testset_path.exists():
            raise FileNotFoundError(f"Тестсет не знайдено: {testset_path}")

        with open(testset_path, "r", encoding="utf-8") as f:
            testset = json.load(f)

        logger.info(f"Запуск evals на {len(testset)} запитаннях...")

        results_list = []
        citation_scores = []
        key_facts_scores = []
        refusal_scores = []
        faithfulness_scores = []
        judge_scores = []
        latencies_ms = []
        total_cost = 0.0

        for i, item in enumerate(testset, 1):
            q_id = item["id"]
            question = item["question"]
            is_unanswerable = item.get("is_unanswerable", False)
            expected_answer = item["expected_answer"]
            expected_sources = item.get("expected_sources", [])
            key_facts = item.get("key_facts", [])

            logger.info(f"[{i}/{len(testset)}] Тестування {q_id} ({item.get('type')})...")

            # 1. Запуск RAG
            t0 = time.perf_counter()
            rag_resp = self.rag.answer(
                query=question,
                prompt_version="v2",
                search_mode=SearchMode.HYBRID_RERANK
            )
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)
            total_cost += rag_resp.cost_usd

            # 2. Отримання контексту для оцінки Faithfulness
            context_text = rag_resp.context_text or ("\n".join(rag_resp.sources_used) + " " + rag_resp.answer)

            # 3. Детерміновані скори
            s_cit = CitationScorer.score(rag_resp, expected_sources, is_unanswerable)
            s_facts = KeyFactsScorer.score(rag_resp, key_facts, is_unanswerable)
            s_refusal = RefusalScorer.score(rag_resp, is_unanswerable)
            s_faith = FaithfulnessScorer.score(rag_resp, context_text)

            # 4. Оцінка судді (LLM-as-a-Judge)
            judge_res = self.judge.evaluate(
                question=question,
                ground_truth=expected_answer,
                candidate_answer=rag_resp.answer
            )
            s_judge = judge_res.score

            citation_scores.append(s_cit)
            key_facts_scores.append(s_facts)
            refusal_scores.append(s_refusal)
            faithfulness_scores.append(s_faith)
            judge_scores.append(s_judge)

            results_list.append({
                "id": q_id,
                "type": item.get("type"),
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": rag_resp.answer,
                "is_unanswerable": is_unanswerable,
                "is_refusal": rag_resp.is_refusal,
                "citations": [c.to_dict() for c in rag_resp.citations],
                "scores": {
                    "citation_accuracy": round(s_cit, 3),
                    "key_facts": round(s_facts, 3),
                    "refusal_accuracy": round(s_refusal, 3),
                    "faithfulness": round(s_faith, 3),
                    "judge_correctness": round(s_judge, 3)
                },
                "judge_reasoning": judge_res.reasoning,
                "latency_ms": round(lat_ms, 1),
                "cost_usd": rag_resp.cost_usd
            })

        # Агреговані метрики
        n = len(testset)
        avg_cit = (sum(citation_scores) / n) * 100.0
        avg_facts = (sum(key_facts_scores) / n) * 100.0
        avg_refusal = (sum(refusal_scores) / n) * 100.0
        avg_faith = (sum(faithfulness_scores) / n) * 100.0
        avg_judge = (sum(judge_scores) / n) * 100.0

        # Зважений підсумковий бал (Overall Quality Score)
        overall_score = (
            0.25 * avg_cit +
            0.20 * avg_facts +
            0.20 * avg_refusal +
            0.15 * avg_faith +
            0.20 * avg_judge
        )

        passed = (
            avg_cit >= MIN_CITATION_ACCURACY and
            avg_refusal >= MIN_REFUSAL_ACCURACY and
            avg_faith >= MIN_FAITHFULNESS and
            overall_score >= MIN_OVERALL_SCORE
        )

        summary = {
            "total_questions": n,
            "passed_quality_gates": passed,
            "metrics": {
                "citation_accuracy_percent": round(avg_cit, 1),
                "key_facts_coverage_percent": round(avg_facts, 1),
                "refusal_accuracy_percent": round(avg_refusal, 1),
                "faithfulness_percent": round(avg_faith, 1),
                "judge_correctness_percent": round(avg_judge, 1),
                "overall_score_percent": round(overall_score, 1),
                "mean_latency_ms": round(sum(latencies_ms) / n, 1),
                "total_cost_usd": round(total_cost, 5),
                "cost_per_1000_queries_usd": round((total_cost / n) * 1000, 4)
            },
            "thresholds": {
                "min_citation_accuracy": MIN_CITATION_ACCURACY,
                "min_refusal_accuracy": MIN_REFUSAL_ACCURACY,
                "min_faithfulness": MIN_FAITHFULNESS,
                "min_overall_score": MIN_OVERALL_SCORE
            },
            "items": results_list
        }

        # Збереження результатів у results/eval_results.json
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "eval_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Результати evals збережено у {out_path}")

        return summary
