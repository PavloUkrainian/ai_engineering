"""
Скрипт демонстрації та тестування Кроку 3 (RAG Pipeline).
Демонструє:
1. Відповіді з точним цитуванням джерел [Документ, Розділ].
2. Контроль бюджету токенів (Token Budget = 2000).
3. Роботу механізму відмови «Не знаю» для запитань без відповіді в корпусі.
4. Порівняння поведінки промптів v1 та v2.
"""

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
from src.retriever import HybridRetriever, SearchMode
from src.rag_pipeline import RAGPipeline
from src.client import ProviderClient
from src.config import CONFIG_DETERMINISTIC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEMO_QUERIES = [
    {
        "type": "Фактологічне запитання (наявне в документах)",
        "query": "Які вагові коефіцієнти використовуються для розрахунку контрольної суми MRZ у стандарті ICAO 9303?"
    },
    {
        "type": "Фактологічне запитання (наявне в документах)",
        "query": "Яка мінімальна роздільна здатність у DPI необхідна для читання мікродруку згідно зі стандартами DocAI?"
    },
    {
        "type": "Unanswerable запитання (відсутнє в документах -> має бути відмова)",
        "query": "Яка ставка податку на додану вартість діє для комерційних поселень на планеті Марс?"
    }
]


def run_step3():
    logger.info("⚙️ Ініціалізація компонентів RAG...")
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

    print("\n" + "=" * 90)
    print("🚀 ДЕМОНСТРАЦІЯ РОБОТИ RAG-СИСТЕМИ З ДЖЕРЕЛАМИ (КРОК 3)")
    print("=" * 90)

    for item in DEMO_QUERIES:
        q = item["query"]
        q_type = item["type"]
        print(f"\n🔹 ТИП: {q_type}")
        print(f"❓ ЗАПИТАННЯ: {q}")

        # Прогін через v2
        resp_v2 = rag.answer(q, prompt_version="v2", search_mode=SearchMode.HYBRID_RERANK)
        print(f"\n📝 ВІДПОВІДЬ (rag_v2):\n{resp_v2.answer}")
        print(f"📌 ЦИТУВАННЯ: {[f'{c.doc_title} > {c.section_title}' for c in resp_v2.citations]}")
        print(f"🛑 СТАТУС ВІДМОВИ (is_refusal): {resp_v2.is_refusal}")
        print(f"📊 МЕТРИКИ: Токени контексту: {resp_v2.context_tokens} | Час: {resp_v2.latency_ms:.1f} ms | Вартість: ${resp_v2.cost_usd:.5f}")
        print("-" * 90)

    # Порівняльний тест v1 проти v2 на unanswerable питанні
    print("\n⚖️ ПОРІВНЯННЯ rag_v1 ПРОТИ rag_v2 НА ЗАПИТАННІ БЕЗ ВІДПОВІДІ:")
    unanswerable_q = "Яка процедура заміни водійського посвідчення на космічному кораблі?"
    print(f"❓ Запитання: {unanswerable_q}")

    resp_v1 = rag.answer(unanswerable_q, prompt_version="v1", search_mode=SearchMode.HYBRID_RERANK)
    resp_v2 = rag.answer(unanswerable_q, prompt_version="v2", search_mode=SearchMode.HYBRID_RERANK)

    print(f"\n[rag_v1 Відповідь]:\n{resp_v1.answer}\n(is_refusal: {resp_v1.is_refusal})")
    print(f"\n[rag_v2 Відповідь]:\n{resp_v2.answer}\n(is_refusal: {resp_v2.is_refusal})")
    print("=" * 90 + "\n")

    logger.info("🎉 Крок 3 успішно виконано!")


if __name__ == "__main__":
    run_step3()
