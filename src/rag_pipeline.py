"""
Модуль RAG-пайплайну (Retrieval-Augmented Generation) з джерелами та контролем контексту.
Реалізує:
1. Контроль бюджету токенів на контекст (Token Budget).
2. Обов'язкове цитування джерел [Документ, Розділ] для кожного твердження.
3. Пороговий механізм відмови («У наданих документах немає цієї інформації») без галюцинацій.
4. Підтримку версій промптів (rag_v1 та rag_v2).
"""

from dataclasses import dataclass, field, asdict
import logging
from pathlib import Path
import re
import time
from typing import List, Dict, Any, Optional, Tuple

from src.ingestion import count_tokens
from src.retriever import HybridRetriever, SearchMode
from src.vector_store import SearchResult
from src.client import BaseLLMClient, ProviderClient
from src.config import SamplingConfig, CONFIG_DETERMINISTIC

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path("prompts")
DEFAULT_TOKEN_BUDGET = 2000
DEFAULT_RELEVANCE_THRESHOLD = 0.015

REFUSAL_PHRASES = [
    "у наданих документах немає цієї інформації",
    "немає цієї інформації",
    "інформація відсутня",
    "у документах цього немає",
    "не можу відповісти",
    "не знаю"
]


@dataclass
class Citation:
    """Вилучене посилання на джерело."""
    doc_title: str
    section_title: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RAGResponse:
    """Уніфікована відповідь RAG-системи."""
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    is_refusal: bool = False
    context_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    prompt_version: str = "v2"
    context_text: str = ""
    is_mock: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citations"] = [c.to_dict() for c in self.citations]
        return d


class ContextBuilder:
    """Конструктор контексту з контролем бюджету токенів."""

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET):
        self.token_budget = token_budget

    def build_context(self, chunks: List[SearchResult]) -> Tuple[str, List[SearchResult], int]:
        """
        Формує структурований блок контексту для LLM,
        не перевищуючи жорсткий бюджет токенів.
        """
        formatted_blocks = []
        included_chunks = []
        current_tokens = 0

        for i, chunk in enumerate(chunks, 1):
            block = (
                f"[ДЖЕРЕЛО {i}]:\n"
                f"Документ: {chunk.doc_title}\n"
                f"Розділ: {chunk.section_title}\n"
                f"Тип документа: {chunk.doc_type}\n"
                f"Зміст:\n{chunk.text}\n"
            )
            block_tokens = count_tokens(block)

            # Перевірка ліміту бюджету токенів
            if current_tokens + block_tokens > self.token_budget:
                logger.info(f"Досягнуто ліміт токенів контексту ({current_tokens}/{self.token_budget}). Чанк {chunk.chunk_id} пропущено.")
                break

            formatted_blocks.append(block)
            included_chunks.append(chunk)
            current_tokens += block_tokens

        full_context_str = "\n---\n".join(formatted_blocks)
        return full_context_str, included_chunks, current_tokens


def is_refusal_response(text: str) -> bool:
    """Перевіряє, чи є відповідь моделі стандартною відмовою «не знаю»."""
    if not text:
        return True
    lower = text.strip().lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def extract_citations(text: str) -> List[Citation]:
    """
    Вилучає всі цитування з тексту формату [Назва документа, Розділ].
    """
    citations: List[Citation] = []
    # Шукаємо патерни [Документ, Розділ]
    matches = re.findall(r'\[([^\]\n]+?),\s*([^\]\n]+?)\]', text)
    seen = set()
    for doc, sec in matches:
        doc_clean = doc.strip()
        sec_clean = sec.strip()
        # Фільтруємо технічні теги типу [ДЖЕРЕЛО 1]
        if doc_clean.upper().startswith("ДЖЕРЕЛО") or not doc_clean or not sec_clean:
            continue
        key = (doc_clean, sec_clean)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(doc_title=doc_clean, section_title=sec_clean))
    return citations


class RAGPipeline:
    """Головний оркестратор RAG-системи."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: Optional[BaseLLMClient] = None,
        sampling: SamplingConfig = CONFIG_DETERMINISTIC,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD
    ):
        self.retriever = retriever
        self.llm_client = llm_client or ProviderClient()
        self.sampling = sampling
        self.context_builder = ContextBuilder(token_budget=token_budget)
        self.relevance_threshold = relevance_threshold

    def _load_prompt(self, version: str) -> str:
        """Завантажує промпт з каталогу prompts/."""
        fname = f"rag_{version}.txt" if not version.startswith("rag_") else f"{version}.txt"
        prompt_path = PROMPTS_DIR / fname
        if not prompt_path.exists():
            raise FileNotFoundError(f"Промпт не знайдено: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8").strip()

    def answer(
        self,
        query: str,
        prompt_version: str = "v2",
        search_mode: SearchMode = SearchMode.HYBRID_RERANK,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Повний цикл RAG:
        1. Retrieval (Hybrid + Rerank).
        2. Перевірка порогової релевантності.
        3. Формування контексту в межах Token Budget.
        4. Генерація відповіді через LLM.
        5. Екстракція цитат та детекція відмов.
        """
        start_time = time.perf_counter()

        # 1. Пошук релевантних чанків
        chunks = self.retriever.search(
            query=query,
            mode=search_mode,
            top_k=top_k,
            filter_dict=filter_dict
        )

        # 2. Пороговий контроль: якщо чанків немає або релевантність нульова — миттєва відмова
        if not chunks or (chunks[0].score < self.relevance_threshold and search_mode == SearchMode.HYBRID_RERANK):
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return RAGResponse(
                query=query,
                answer="У наданих документах немає цієї інформації.",
                citations=[],
                sources_used=[],
                is_refusal=True,
                context_tokens=0,
                total_tokens=0,
                latency_ms=round(latency_ms, 2),
                cost_usd=0.0,
                prompt_version=prompt_version,
                is_mock=False
            )

        # 3. Побудова контексту в межах ліміту токенів
        context_str, used_chunks, context_tokens = self.context_builder.build_context(chunks)

        # 4. Завантаження системного промпту
        system_prompt = self._load_prompt(prompt_version)

        user_prompt = (
            f"КОНТЕКСТ З ДОКУМЕНТІВ:\n"
            f"{context_str}\n\n"
            f"ЗАПИТАННЯ КОРИСТУВАЧА:\n"
            f"{query}\n\n"
            f"Надай точну відповідь строго за правилами та з обов'язковими цитатами [Документ, Розділ]."
        )

        # 5. Виклик моделі через клієнт
        llm_resp = self.llm_client.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            sampling=self.sampling
        )

        answer_text = llm_resp.content.strip()

        # 6. Екстракція цитат та джерел
        citations = extract_citations(answer_text)
        sources_used = list({f"{c.doc_title} > {c.section_title}" for c in citations})
        if not sources_used and used_chunks:
            # Джерела, які реально увійшли в контекст
            sources_used = [f"{ch.doc_title} > {ch.section_title}" for ch in used_chunks]

        is_refusal = is_refusal_response(answer_text)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RAGResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            sources_used=sources_used,
            is_refusal=is_refusal,
            context_tokens=context_tokens,
            total_tokens=llm_resp.input_tokens + llm_resp.output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd=llm_resp.cost_usd,
            prompt_version=prompt_version,
            context_text=context_str,
            is_mock=llm_resp.is_mock
        )
