"""
Unit та regression тести для Кроку 4: Testset, Scorers, LLM-as-a-Judge Calibration та CI Quality Gates.
"""

import json
import unittest
from pathlib import Path

from evals.scorers import CitationScorer, KeyFactsScorer, RefusalScorer, FaithfulnessScorer
from evals.judge import LLMJudge, JudgeResult
from evals.evaluator import (
    MIN_CITATION_ACCURACY,
    MIN_REFUSAL_ACCURACY,
    MIN_FAITHFULNESS,
    MIN_OVERALL_SCORE
)
from src.rag_pipeline import RAGResponse, Citation
from src.client import MockFallbackClient


class TestStep4Evals(unittest.TestCase):

    def test_testset_integrity(self):
        """Перевірка структури та повноти тестового набору (testset.json >= 20 питань)."""
        testset_path = Path("evals/testset.json")
        self.assertTrue(testset_path.exists(), "testset.json має існувати")
        
        with open(testset_path, "r", encoding="utf-8") as f:
            testset = json.load(f)

        self.assertGreaterEqual(len(testset), 20, "Тестсет повинен містити щонайменше 20 запитань")

        factual_count = sum(1 for item in testset if item.get("type") == "factual")
        unanswerable_count = sum(1 for item in testset if item.get("type") == "unanswerable")
        trap_count = sum(1 for item in testset if "trap" in item.get("type", ""))

        self.assertGreaterEqual(factual_count, 10, "Повинно бути не менше 10 фактологічних питань")
        self.assertGreaterEqual(unanswerable_count, 3, "Повинно бути не менше 3 unanswerable питань")
        self.assertGreaterEqual(trap_count, 3, "Повинно бути не менше 3 питань-пасток")

        for item in testset:
            self.assertIn("id", item)
            self.assertIn("question", item)
            self.assertIn("expected_answer", item)
            if not item.get("is_unanswerable", False):
                self.assertIn("expected_sources", item)
                self.assertIn("key_facts", item)

    def test_citation_scorer(self):
        """Перевірка детермінованого оцінювача цитувань джерел."""
        # 1. Точний збіг цитувань
        resp_correct = RAGResponse(
            query="test",
            answer="Answer text",
            citations=[Citation(doc_title="ICAO Doc 9303 Part 3", section_title="MRZ Structure")]
        )
        score_1 = CitationScorer.score(resp_correct, ["ICAO Doc 9303 Part 3", "MRZ Structure"], False)
        self.assertEqual(score_1, 1.0)

        # 2. Відсутність очікуваних джерел
        resp_wrong = RAGResponse(
            query="test",
            answer="Answer text",
            citations=[Citation(doc_title="Another Document", section_title="General")]
        )
        score_0 = CitationScorer.score(resp_wrong, ["ICAO Doc 9303 Part 3"], False)
        self.assertEqual(score_0, 0.0)

        # 3. Питання без відповіді (unanswerable)
        resp_unans = RAGResponse(query="test", answer="Не знаю", is_refusal=True, citations=[])
        self.assertEqual(CitationScorer.score(resp_unans, [], True), 1.0)

    def test_key_facts_scorer(self):
        """Перевірка детермінованого оцінювача наявності ключових фактів."""
        resp = RAGResponse(
            query="test",
            answer="Роздільна здатність становить 300 DPI та глибина кольору 24 біти."
        )
        # Усі факти наявні
        self.assertEqual(KeyFactsScorer.score(resp, ["300", "24 біти"], False), 1.0)
        # Частина фактів наявна
        self.assertEqual(KeyFactsScorer.score(resp, ["300", "5000 мегапікселів"], False), 0.5)
        # Для unanswerable питань
        resp_ref = RAGResponse(query="test", answer="Немає інформації", is_refusal=True)
        self.assertEqual(KeyFactsScorer.score(resp_ref, ["будь-що"], True), 1.0)

    def test_refusal_scorer(self):
        """Перевірка оцінювача коректності відмов («не знаю»)."""
        # Правильна відмова на unanswerable
        resp_ref = RAGResponse(query="test", answer="У наданих документах немає цієї інформації.", is_refusal=True)
        self.assertEqual(RefusalScorer.score(resp_ref, is_unanswerable=True), 1.0)

        # Помилка: галюцинація замість відмови на unanswerable
        resp_hallucination = RAGResponse(query="test", answer="Податок на Марсі складає 20%.", is_refusal=False)
        self.assertEqual(RefusalScorer.score(resp_hallucination, is_unanswerable=True), 0.0)

        # Фактологічне питання — не повинно бути відмови
        self.assertEqual(RefusalScorer.score(resp_hallucination, is_unanswerable=False), 1.0)
        self.assertEqual(RefusalScorer.score(resp_ref, is_unanswerable=False), 0.0)

    def test_faithfulness_scorer(self):
        """Перевірка оцінювача Faithfulness (відсутність галюцинацій та заземлення)."""
        # 1. Відмова без відповіді -> 100% faithful
        resp_ref = RAGResponse(query="test", answer="Не знаю", is_refusal=True)
        self.assertEqual(FaithfulnessScorer.score(resp_ref, "Будь-який контекст"), 1.0)

        # 2. Обґрунтована відповідь з цитатою та числами з контексту
        context = "Згідно з ICAO 9303 Part 3, роздільна здатність має бути 300 DPI."
        resp_grounded = RAGResponse(
            query="test",
            answer="Мінімальна роздільна здатність 300 DPI [ICAO Doc 9303 Part 3, Scope].",
            citations=[Citation(doc_title="ICAO Doc 9303 Part 3", section_title="Scope")]
        )
        score = FaithfulnessScorer.score(resp_grounded, context)
        self.assertGreaterEqual(score, 0.8, "Faithfulness для заземленої відповіді має бути >= 0.8")

        # 3. Порожня відповідь
        resp_empty = RAGResponse(query="test", answer="")
        self.assertEqual(FaithfulnessScorer.score(resp_empty, context), 0.0)

    def test_judge_calibration_file_and_metrics(self):
        """Перевірка вибірки для калібрування LLM-as-a-Judge (>= 10 зразків)."""
        calib_path = Path("evals/calibration_sample.json")
        self.assertTrue(calib_path.exists(), "calibration_sample.json має існувати")

        with open(calib_path, "r", encoding="utf-8") as f:
            samples = json.load(f)

        self.assertGreaterEqual(len(samples), 10, "Вибірка для калібрування має містити щонайменше 10 зразків")
        for s in samples:
            self.assertIn("id", s)
            self.assertIn("question", s)
            self.assertIn("ground_truth", s)
            self.assertIn("candidate_answer", s)
            self.assertIn("human_correctness_score", s)
            self.assertTrue(0.0 <= s["human_correctness_score"] <= 1.0)

    def test_quality_gate_thresholds(self):
        """Перевірка наявності коректних числових порогів для CI."""
        self.assertEqual(MIN_CITATION_ACCURACY, 80.0)
        self.assertEqual(MIN_REFUSAL_ACCURACY, 90.0)
        self.assertEqual(MIN_FAITHFULNESS, 80.0)
        self.assertEqual(MIN_OVERALL_SCORE, 80.0)


if __name__ == "__main__":
    unittest.main()
