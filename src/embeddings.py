"""
Модуль обчислення векторних ембеддінгів (OpenAI text-embedding-3-small + Disk Cache + Fallback).
Забезпечує:
1. Виклики OpenAI Embeddings API (text-embedding-3-small, 1536 вимірів).
2. Дискове кешування у data/embeddings_cache.json (щоб не платити двічі і не звертатися до мережі повторно).
3. Детермінований офлайн-фолбек для CI / smoke-тестів без звернення до мережі.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from openai import OpenAI
from src.config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = Path("data/embeddings_cache.json")
EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingClient:
    """Клієнт для генерації та кешування ембеддінгів."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, cache_file: Path = CACHE_FILE):
        self.model_name = model_name
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, List[float]] = self._load_cache()
        self.client: Optional[OpenAI] = None

        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key and not api_key.startswith("your_") and not settings.use_mock_fallback:
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.warning(f"Не вдалося ініціалізувати OpenAI клієнт: {e}")

    def _load_cache(self) -> Dict[str, List[float]]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Помилка читання кешу ембеддінгів: {e}")
        return {}

    def _save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Не вдалося зберегти кеш ембеддінгів: {e}")

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _deterministic_mock_embedding(self, text: str) -> List[float]:
        """
        Детермінований офлайн-вектор на основі хешування термінів та n-грам.
        Нормалізований L2-вектор розмірності 1536 для офлайн-тестів та CI.
        """
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        # Хешування окремих слів і біграм
        words = text.lower().split()
        for i, w in enumerate(words):
            idx = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
            vec[idx] += 1.0
            if i < len(words) - 1:
                bigram = f"{w}_{words[i+1]}"
                idx2 = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16) % EMBEDDING_DIM
                vec[idx2] += 1.5

        # L2-нормалізація
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def get_embedding(self, text: str) -> List[float]:
        """Отримати ембеддінг для одного тексту (з кешу або генерацією)."""
        return self.get_embeddings([text])[0]

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Отримати ембеддінги для списку текстів (з кешуванням та батч-обробкою).
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        # 1. Перевірка кешу
        for idx, text in enumerate(texts):
            h = self._text_hash(text)
            if h in self.cache:
                results[idx] = self.cache[h]
            else:
                missing_indices.append(idx)
                missing_texts.append(text)

        # 2. Якщо все знайдено в кеші — повертаємо
        if not missing_texts:
            return results  # type: ignore

        # 3. Генерація через OpenAI API
        computed_embeddings: List[List[float]] = []
        if self.client is not None and not settings.use_mock_fallback:
            try:
                # Батчі по 50 текстів
                batch_size = 50
                for b_start in range(0, len(missing_texts), batch_size):
                    batch = missing_texts[b_start:b_start + batch_size]
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    for item in response.data:
                        computed_embeddings.append(item.embedding)
            except Exception as e:
                logger.warning(f"Збій OpenAI Embeddings API ({e}). Перехід на офлайн-ембеддінги.")
                computed_embeddings = [self._deterministic_mock_embedding(t) for t in missing_texts]
        else:
            computed_embeddings = [self._deterministic_mock_embedding(t) for t in missing_texts]

        # 4. Запис у кеш і в фінальний результат
        for idx, emb, text in zip(missing_indices, computed_embeddings, missing_texts):
            results[idx] = emb
            self.cache[self._text_hash(text)] = emb

        self._save_cache()
        return results  # type: ignore
