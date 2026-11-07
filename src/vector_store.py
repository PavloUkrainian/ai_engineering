"""
Модуль векторного індексу на базі Qdrant (in-memory / local storage).
Зберігає чанки з ембеддінгами та повним набором метаданих у payload.
Підтримує фільтрацію за метаданими (doc_type, version_date, doc_id).
"""

from dataclasses import dataclass, asdict
import logging
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)

from src.chunking import Chunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "kyc_docai_chunks"


@dataclass
class SearchResult:
    """Уніфікований результат пошуку (векторного, лексичного або гібридного)."""
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str
    doc_type: str
    version_date: str
    text: str
    score: float
    rank: int = 1
    source: str = "dense"

    def to_dict(self) -> dict:
        return asdict(self)


class QdrantVectorStore:
    """Векторне сховище на базі вбудованого Qdrant."""

    def __init__(self, location: str = ":memory:", collection_name: str = COLLECTION_NAME, vector_dim: int = 1536):
        self.location = location
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.client = QdrantClient(location=location)
        self._init_collection()

    def _init_collection(self):
        """Ініціалізація колекції Qdrant з косинусною відстанню."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
            )
            logger.info(f"Створено нову колекцію Qdrant: {self.collection_name} (dim={self.vector_dim})")

    def index_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """Запис чанків та їх ембеддінгів у векторну базу."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Невідповідність кількості чанків ({len(chunks)}) та ембеддінгів ({len(embeddings)})")

        points = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source_file": chunk.source_file,
                "doc_title": chunk.doc_title,
                "section_title": chunk.section_title,
                "doc_type": chunk.doc_type,
                "version_date": chunk.version_date,
                "doc_hash": chunk.doc_hash,
                "token_count": chunk.token_count,
                "char_count": chunk.char_count,
                "text": chunk.text
            }
            points.append(PointStruct(id=i + 1, vector=emb, payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Індексовано {len(points)} чанків у Qdrant")
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Векторний пошук найближчих сусідів (kNN за косинусною схожістю).
        Підтримує фільтри за метаданими (наприклад doc_type, version_date).
        """
        query_filter = None
        if filter_dict:
            must_conditions = []
            for k, v in filter_dict.items():
                if k == "min_date":
                    must_conditions.append(FieldCondition(key="version_date", range=Range(gte=v)))
                else:
                    must_conditions.append(FieldCondition(key=k, match=MatchValue(value=v)))
            if must_conditions:
                query_filter = Filter(must=must_conditions)

        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter
        )

        results: List[SearchResult] = []
        for rank, p in enumerate(res.points, 1):
            pl = p.payload or {}
            results.append(SearchResult(
                chunk_id=pl.get("chunk_id", ""),
                doc_id=pl.get("doc_id", ""),
                doc_title=pl.get("doc_title", ""),
                section_title=pl.get("section_title", ""),
                doc_type=pl.get("doc_type", ""),
                version_date=pl.get("version_date", ""),
                text=pl.get("text", ""),
                score=float(p.score) if p.score is not None else 0.0,
                rank=rank,
                source="dense"
            ))

        return results
