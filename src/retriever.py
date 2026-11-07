"""
Модуль гібридного пошуку (Hybrid Retriever) та Reranking.
Реалізує:
1. Тільки векторний пошук (Dense Mode).
2. Гібридний пошук через Reciprocal Rank Fusion (RRF Mode: BM25 + Dense).
3. Гібридний пошук з Reranking (Cross-Scoring / Entity-Term Proximity).
4. Фільтрацію за метаданими (doc_type, version_date).
"""

from enum import Enum
import logging
import re
from typing import List, Dict, Any, Optional

from src.vector_store import QdrantVectorStore, SearchResult
from src.bm25_search import BM25SearchEngine, tokenize_query
from src.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class Reranker:
    """
    Reranker для перевпорядкування результатів пошуку.
    Враховує:
    1. Нормалізований комбінований RRF-ранг (Dense + Sparse).
    2. Семантичну щільність специфічних термів запиту в тексті та заголовку секції.
    3. Точний збіг ключових абревіатур та числових сутностей (ICAO, TD1, TD3, MRZ, 300 DPI, РНОКПП, FATF, PEP, OVI тощо).
    """

    @staticmethod
    def rerank(query: str, candidates: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        if not candidates:
            return []

        query_tokens = set(tokenize_query(query))
        query_upper_entities = set(re.findall(r'\b[A-Z0-9А-ЯІЇЄҐ\-]{2,}\b', query))

        # Нормалізація базового RRF скору до діапазону [0, 1]
        raw_scores = [c.score for c in candidates]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        score_range = (max_s - min_s) if max_s > min_s else 1.0

        scored_candidates = []
        for res in candidates:
            text_lower = res.text.lower()
            sec_lower = res.section_title.lower()
            doc_lower = res.doc_title.lower()

            # Нормалізований базовий RRF скор (0.0 .. 1.0)
            norm_base_score = (res.score - min_s) / score_range

            # 1. Покриття термів запиту (Query Coverage)
            token_matches = sum(1 for t in query_tokens if t in text_lower or t in sec_lower)
            coverage_score = token_matches / max(1, len(query_tokens))

            # 2. Збіг критичних сутностей та абревіатур (Entity Match)
            entity_score = 0.0
            if query_upper_entities:
                matched_entities = sum(1 for e in query_upper_entities if e in res.text or e in res.doc_title)
                entity_score = matched_entities / len(query_upper_entities)

            # 3. Бонус за наявність ключових слів безпосередньо в назві розділу
            section_bonus = sum(0.25 for t in query_tokens if t in sec_lower)
            section_bonus = min(1.0, section_bonus)

            # Фінальний збалансований скор (де базовий RRF має найбільшу вагу)
            final_rerank_score = (
                0.55 * norm_base_score +
                0.25 * coverage_score +
                0.10 * entity_score +
                0.10 * section_bonus
            )

            scored_candidates.append((final_rerank_score, res))

        # Сортування за зменшенням скору після reranking
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        reranked_results: List[SearchResult] = []
        for rank, (score, res) in enumerate(scored_candidates[:top_k], 1):
            reranked_results.append(SearchResult(
                chunk_id=res.chunk_id,
                doc_id=res.doc_id,
                doc_title=res.doc_title,
                section_title=res.section_title,
                doc_type=res.doc_type,
                version_date=res.version_date,
                text=res.text,
                score=round(float(score), 4),
                rank=rank,
                source="hybrid_rerank"
            ))

        return reranked_results


class HybridRetriever:
    """Головний оркестратор retrieval: Dense + BM25 + Fusion + Reranking."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_engine: BM25SearchEngine,
        embedding_client: EmbeddingClient,
        rrf_constant: int = 60
    ):
        self.vector_store = vector_store
        self.bm25_engine = bm25_engine
        self.embedding_client = embedding_client
        self.rrf_constant = rrf_constant

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID_RERANK,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Уніфікований пошук у трьох режимах:
        - DENSE: тільки векторний пошук у Qdrant
        - HYBRID: BM25 + Dense об'єднані за алгоритмом Reciprocal Rank Fusion (RRF)
        - HYBRID_RERANK: Hybrid + Cross-Scoring Reranker
        """
        # 1. Режим DENSE
        if mode == SearchMode.DENSE:
            query_vector = self.embedding_client.get_embedding(query)
            return self.vector_store.search(query_vector=query_vector, top_k=top_k, filter_dict=filter_dict)

        # 2. Отримання кандидатів від обох джерел (збільшений пул для злиття, наприклад 2 * top_k)
        candidate_pool_size = max(20, top_k * 2)

        # Dense кандидати
        query_vector = self.embedding_client.get_embedding(query)
        dense_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=candidate_pool_size,
            filter_dict=filter_dict
        )

        # BM25 кандидати
        bm25_results = self.bm25_engine.search(
            query=query,
            top_k=candidate_pool_size,
            filter_dict=filter_dict
        )

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF_Score(d) = sum(1 / (k + rank_i(d)))
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}

        for rank, res in enumerate(dense_results, 1):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_constant + rank))
            chunk_map[res.chunk_id] = res

        for rank, res in enumerate(bm25_results, 1):
            rrf_scores[res.chunk_id] = rrf_scores.get(res.chunk_id, 0.0) + (1.0 / (self.rrf_constant + rank))
            if res.chunk_id not in chunk_map:
                chunk_map[res.chunk_id] = res

        # Сортування за RRF скором
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        hybrid_results: List[SearchResult] = []
        for rank, (cid, score) in enumerate(sorted_chunks[:candidate_pool_size], 1):
            base_res = chunk_map[cid]
            hybrid_results.append(SearchResult(
                chunk_id=base_res.chunk_id,
                doc_id=base_res.doc_id,
                doc_title=base_res.doc_title,
                section_title=base_res.section_title,
                doc_type=base_res.doc_type,
                version_date=base_res.version_date,
                text=base_res.text,
                score=round(score, 5),
                rank=rank,
                source="hybrid"
            ))

        if mode == SearchMode.HYBRID:
            return hybrid_results[:top_k]

        # 4. Режим HYBRID_RERANK
        return Reranker.rerank(query=query, candidates=hybrid_results, top_k=top_k)
