"""
Unit та integration тести для Кроку 2: Векторна база Qdrant, BM25, Hybrid Search, Reranking, Metadata Filters.
"""

import json
import unittest
from pathlib import Path

from src.chunking import Chunk
from src.embeddings import EmbeddingClient
from src.vector_store import QdrantVectorStore, SearchResult
from src.bm25_search import BM25SearchEngine, tokenize_query
from src.retriever import HybridRetriever, SearchMode, Reranker
from src.eval_retrieval import load_chunks, load_testset, evaluate_retrieval_mode


class TestStep2Retriever(unittest.TestCase):

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

    def test_qdrant_vector_store_search(self):
        """Перевірка векторного пошуку найближчих сусідів у Qdrant."""
        q_vec = self.emb_client.get_embedding("ICAO Doc 9303 MRZ format")
        results = self.vector_store.search(query_vector=q_vec, top_k=5)

        self.assertEqual(len(results), 5)
        self.assertIsInstance(results[0], SearchResult)
        self.assertGreater(results[0].score, 0.0)
        # Має знайти стандарти ICAO
        self.assertTrue(any("icao" in r.doc_id.lower() for r in results))

    def test_qdrant_metadata_filtering(self):
        """Перевірка фільтрації за метаданими у Qdrant (doc_type)."""
        q_vec = self.emb_client.get_embedding("ідентифікація клієнта")
        
        # Фільтр тільки по regulations
        reg_results = self.vector_store.search(
            query_vector=q_vec,
            top_k=5,
            filter_dict={"doc_type": "regulation"}
        )
        self.assertGreater(len(reg_results), 0)
        for r in reg_results:
            self.assertEqual(r.doc_type, "regulation")

        # Фільтр тільки по standards
        std_results = self.vector_store.search(
            query_vector=q_vec,
            top_k=5,
            filter_dict={"doc_type": "standard"}
        )
        self.assertGreater(len(std_results), 0)
        for r in std_results:
            self.assertEqual(r.doc_type, "standard")

    def test_bm25_lexical_search(self):
        """Перевірка лексичного пошуку BM25 за специфічними ключовими словами."""
        results = self.bm25_engine.search(query="Laplacian variance 100.0 blur", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].doc_id, "docai_blur_glare_shadow_detection")
        self.assertIn("Laplacian", results[0].text)

    def test_bm25_metadata_filtering(self):
        """Перевірка фільтрації за метаданими в BM25."""
        results = self.bm25_engine.search(
            query="passport",
            top_k=5,
            filter_dict={"doc_type": "standard"}
        )
        for r in results:
            self.assertEqual(r.doc_type, "standard")

    def test_hybrid_search_rrf(self):
        """Перевірка об'єднання векторного та лексичного пошуку через RRF."""
        results = self.retriever.search(
            query="вагові коефіцієнти 7 3 1 MRZ",
            mode=SearchMode.HYBRID,
            top_k=5
        )
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0].source, "hybrid")
        self.assertTrue(any("icao_doc9303_check_digit_algorithms" in r.doc_id for r in results))

    def test_hybrid_rerank_mode(self):
        """Перевірка режиму Hybrid + Reranking."""
        results = self.retriever.search(
            query="DPI для OCR мікродруку",
            mode=SearchMode.HYBRID_RERANK,
            top_k=3
        )
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].source, "hybrid_rerank")
        self.assertIn("docai_image_resolution_dpi_standards", results[0].doc_id)

    def test_retrieval_benchmark_metrics(self):
        """Перевірка розрахунку метрик Recall@k та MRR."""
        testset = load_testset()
        self.assertGreaterEqual(len(testset), 10)

        metrics = evaluate_retrieval_mode(self.retriever, testset[:5], SearchMode.DENSE)
        self.assertIn("recall_at_1_percent", metrics)
        self.assertIn("recall_at_3_percent", metrics)
        self.assertIn("recall_at_5_percent", metrics)
        self.assertIn("mrr", metrics)
        self.assertIn("mean_latency_ms", metrics)
        self.assertGreater(metrics["recall_at_5_percent"], 0.0)

    def test_benchmark_artifact_validity(self):
        """Перевірка наявності та коректності збереженого JSON-файлу бенчмарку."""
        out_path = Path("results/retrieval_benchmark.json")
        self.assertTrue(out_path.exists(), "results/retrieval_benchmark.json має існувати")
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertIn("dense", data)
            self.assertIn("hybrid", data)
            self.assertIn("hybrid_rerank", data)
            self.assertGreaterEqual(data["dense"]["recall_at_5_percent"], 80.0)


if __name__ == "__main__":
    unittest.main()
