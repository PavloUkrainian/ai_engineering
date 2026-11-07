"""
Unit та regression тести для Кроку 3: RAG Pipeline, Context Builder, Token Budget, Citations та Refusal.
"""

import unittest
from pathlib import Path

from src.vector_store import SearchResult
from src.rag_pipeline import (
    ContextBuilder,
    extract_citations,
    is_refusal_response,
    RAGPipeline,
    Citation,
    DEFAULT_TOKEN_BUDGET
)
from src.retriever import HybridRetriever, SearchMode
from src.eval_retrieval import load_chunks
from src.embeddings import EmbeddingClient
from src.vector_store import QdrantVectorStore
from src.bm25_search import BM25SearchEngine
from src.client import MockFallbackClient
from src.config import CONFIG_DETERMINISTIC


class TestStep3RAG(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chunks = load_chunks()
        cls.emb_client = EmbeddingClient()
        texts = [c.text for c in cls.chunks]
        cls.embeddings = cls.emb_client.get_embeddings(texts)
        cls.vector_store = QdrantVectorStore(location=":memory:", vector_dim=len(cls.embeddings[0]))
        cls.vector_store.index_chunks(cls.chunks, cls.embeddings)
        cls.bm25_engine = BM25SearchEngine(cls.chunks)
        cls.retriever = HybridRetriever(cls.vector_store, cls.bm25_engine, cls.emb_client)
        cls.mock_client = MockFallbackClient(model_name="mock-rag")

    def test_context_builder_budget_limit(self):
        """Перевірка жорсткого обмеження бюджету токенів на контекст."""
        dummy_chunks = [
            SearchResult(
                chunk_id=f"c_{i}",
                doc_id=f"doc_{i}",
                doc_title=f"Doc {i}",
                section_title=f"Section {i}",
                doc_type="standard",
                version_date="2023-01-01",
                text="Короткий текст для перевірки бюджету токенів. " * 5,
                score=0.9 - i * 0.1,
                rank=i + 1
            )
            for i in range(10)
        ]

        small_budget = 200
        builder = ContextBuilder(token_budget=small_budget)
        context_str, included_chunks, total_tokens = builder.build_context(dummy_chunks)

        self.assertLessEqual(total_tokens, small_budget)
        self.assertLess(len(included_chunks), len(dummy_chunks))
        self.assertGreater(len(included_chunks), 0)

    def test_extract_citations(self):
        """Перевірка вилучення цитувань у форматі [Назва документа, Розділ]."""
        text = (
            "Мінімальна роздільна здатність становить 300 DPI [DocAI Standard, 1. Minimum DPI]. "
            "Для мікродруку вимагається 600 DPI [DocAI Standard, 2. Microprint Rules]. "
            "Некоректний [ДЖЕРЕЛО 1] тег не повинен потрапити."
        )
        citations = extract_citations(text)
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0].doc_title, "DocAI Standard")
        self.assertEqual(citations[0].section_title, "1. Minimum DPI")
        self.assertEqual(citations[1].section_title, "2. Microprint Rules")

    def test_is_refusal_response(self):
        """Перевірка детекції відмов моделі."""
        self.assertTrue(is_refusal_response("У наданих документах немає цієї інформації."))
        self.assertTrue(is_refusal_response("На жаль, інформація відсутня в контексті."))
        self.assertTrue(is_refusal_response("Не знаю, оскільки текст не містить даних."))
        self.assertFalse(is_refusal_response("Вагові коефіцієнти становлять 7, 3, 1 [ICAO 9303, Section 1]."))

    def test_rag_pipeline_with_mock_client(self):
        """Перевірка повного циклу RAG-пайплайну з fallback-клієнтом."""
        rag = RAGPipeline(
            retriever=self.retriever,
            llm_client=self.mock_client,
            sampling=CONFIG_DETERMINISTIC,
            token_budget=1500
        )

        resp = rag.answer("Які вагові коефіцієнти використовуються для розрахунку контрольної суми MRZ у стандарті ICAO 9303?", prompt_version="v2")
        self.assertTrue(resp.query)
        self.assertTrue(resp.answer)
        self.assertGreater(resp.context_tokens, 0)
        self.assertEqual(resp.prompt_version, "v2")
        self.assertTrue(resp.is_mock)

    def test_rag_pipeline_unanswerable_refusal(self):
        """Перевірка відмови на unanswerable запит."""
        rag = RAGPipeline(
            retriever=self.retriever,
            llm_client=self.mock_client,
            sampling=CONFIG_DETERMINISTIC,
            token_budget=1500
        )

        resp = rag.answer("Яка ставка податку на додану вартість діє для комерційних поселень на планеті Марс?", prompt_version="v2")
        self.assertTrue(resp.is_refusal)
        self.assertIn("немає цієї інформації", resp.answer.lower())

    def test_prompts_files_exist(self):
        """Перевірка наявності версіонованих файлів промптів."""
        v1_path = Path("prompts/rag_v1.txt")
        v2_path = Path("prompts/rag_v2.txt")
        changelog_path = Path("prompts/CHANGELOG.md")

        self.assertTrue(v1_path.exists())
        self.assertTrue(v2_path.exists())
        self.assertTrue(changelog_path.exists())
        self.assertGreater(len(v1_path.read_text(encoding="utf-8")), 50)
        self.assertGreater(len(v2_path.read_text(encoding="utf-8")), 50)


if __name__ == "__main__":
    unittest.main()
