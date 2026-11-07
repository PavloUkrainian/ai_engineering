"""
Модуль лексичного пошуку BM25 (BM25Okapi).
Підтримує токенізацію кирилиці та латиниці, ранжування за термами та фільтрацію за метаданими.
"""

import re
from typing import List, Dict, Any, Optional

from rank_bm25 import BM25Okapi
from src.chunking import Chunk
from src.vector_store import SearchResult


def tokenize_query(text: str) -> List[str]:
    """Токенізація запиту та документів для BM25 (кирилиця, латиниця, цифри)."""
    if not text:
        return []
    # Вилучаємо окремі слова та алфавітно-цифрові комбінації довжиною >= 2
    tokens = re.findall(r'[\w\u0400-\u04FF]+', text.lower())
    return [t for t in tokens if len(t) >= 2]


class BM25SearchEngine:
    """Двигун лексичного пошуку BM25."""

    def __init__(self, chunks: Optional[List[Chunk]] = None):
        self.chunks: List[Chunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        if chunks:
            self.index_chunks(chunks)

    def index_chunks(self, chunks: List[Chunk]):
        """Індексування чанків для BM25."""
        self.chunks = list(chunks)
        self.corpus_tokens = [tokenize_query(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Лексичний пошук BM25 з опціональною фільтрацією за метаданими.
        """
        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize_query(query)
        if not query_tokens:
            return []

        doc_scores = self.bm25.get_scores(query_tokens)

        # Фільтрація та ранжування
        candidates = []
        for idx, score in enumerate(doc_scores):
            if score <= 0.0:
                continue

            chunk = self.chunks[idx]

            # Перевірка метаданих-фільтрів
            if filter_dict:
                matched = True
                for k, v in filter_dict.items():
                    if k == "min_date" and chunk.version_date < v:
                        matched = False
                        break
                    elif hasattr(chunk, k) and getattr(chunk, k) != v:
                        matched = False
                        break
                if not matched:
                    continue

            candidates.append((idx, score, chunk))

        # Сортування за зменшенням BM25 score
        candidates.sort(key=lambda x: x[1], reverse=True)

        results: List[SearchResult] = []
        for rank, (idx, score, chunk) in enumerate(candidates[:top_k], 1):
            results.append(SearchResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                doc_title=chunk.doc_title,
                section_title=chunk.section_title,
                doc_type=chunk.doc_type,
                version_date=chunk.version_date,
                text=chunk.text,
                score=float(score),
                rank=rank,
                source="bm25"
            ))

        return results
