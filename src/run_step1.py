"""
Скрипт запуску Кроку 1: Корпус, Ingestion, Дедуплікація та Chunking.
Зберігає:
- data/manifest.json
- data/chunks.jsonl
- results/chunking_comparison.json
"""

import json
import logging
from pathlib import Path
import sys

# Додаємо корінь проєкту в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_corpus import build_corpus, CORPUS_DIR
from src.ingestion import IngestionPipeline
from src.chunking import StructureAwareChunker, FixedSizeChunker, compare_strategies

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def run_step1():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Перевірка або створення корпусу документів
    if not CORPUS_DIR.exists() or len(list(CORPUS_DIR.glob("*.*"))) < 30:
        logger.info("📁 Створення корпусу документів...")
        build_corpus()
    else:
        logger.info(f"📁 Використовуємо наявний корпус у {CORPUS_DIR}")

    # 2. Запуск Ingestion Pipeline (читання, очищення, дедуплікація)
    logger.info("⚙️ Запуск Ingestion Pipeline (PDF / MD / HTML)...")
    pipeline = IngestionPipeline(CORPUS_DIR)
    unique_docs, manifest = pipeline.run()

    logger.info(f"📄 Оброблено документів: {len(manifest)} (з них унікальних: {len(unique_docs)})")
    duplicates = [m for m in manifest if m["is_duplicate"]]
    logger.info(f"🔍 Виявлено дублікатів: {len(duplicates)}")
    for d in duplicates:
        logger.info(f"   -> Дублікат: {d['doc_id']} (ідентичний до {d['duplicate_of']})")

    # Збереження маніфесту
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Маніфест збережено у {manifest_path}")

    # 3. Чанкінг за стратегією Structure-Aware
    logger.info("✂️ Виконання Structure-Aware Chunking...")
    chunker = StructureAwareChunker(target_max_tokens=400, overlap_tokens=50)
    all_chunks = []
    for doc in unique_docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)

    # Збереження chunks.jsonl
    chunks_path = DATA_DIR / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    logger.info(f"💾 {len(all_chunks)} чанків збережено у {chunks_path}")

    # 4. Порівняння стратегій чанкінгу
    logger.info("📊 Порівняння Structure-Aware vs Fixed-Size Chunking...")
    comparison = compare_strategies(unique_docs)

    struct_stats = comparison["structure_aware"]
    fixed_stats = comparison["fixed_size"]

    comparison_summary = {
        "structure_aware": struct_stats,
        "fixed_size": fixed_stats
    }

    comp_path = RESULTS_DIR / "chunking_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)

    # Друк порівняльної таблиці для перевіряючого
    struct_range = f"{struct_stats['min_tokens']} - {struct_stats['max_tokens']}"
    fixed_range = f"{fixed_stats['min_tokens']} - {fixed_stats['max_tokens']}"
    struct_sent = f"{struct_stats['sentence_integrity_rate']*100:.1f}%"
    fixed_sent = f"{fixed_stats['sentence_integrity_rate']*100:.1f}%"
    struct_sec = f"{struct_stats['section_context_rate']*100:.1f}%"
    fixed_sec = f"{fixed_stats['section_context_rate']*100:.1f}%"

    print("\n" + "=" * 90)
    print("📈 РЕЗУЛЬТАТИ ПОРІВНЯННЯ ЧАНКІНГ-СТРАТЕГІЙ (КРОК 1)")
    print("=" * 90)
    print(f"{'Метрика':<35} | {'Structure-Aware (Наша)':<25} | {'Fixed-Size 500 chars':<25}")
    print("-" * 90)
    print(f"{'Всього згенеровано чанків':<35} | {struct_stats['total_chunks']:<25} | {fixed_stats['total_chunks']:<25}")
    print(f"{'Середня кількість токенів':<35} | {struct_stats['mean_tokens']:<25} | {fixed_stats['mean_tokens']:<25}")
    print(f"{'Діапазон токенів (min - max)':<35} | {struct_range:<25} | {fixed_range:<25}")
    print(f"{'Збереження меж речень (Integrity)':<35} | {struct_sent:<25} | {fixed_sent:<25}")
    print(f"{'Наявність контексту розділу':<35} | {struct_sec:<25} | {fixed_sec:<25}")
    print("=" * 90 + "\n")

    logger.info("🎉 Крок 1 успішно виконано!")
    return len(manifest), len(unique_docs), len(all_chunks)


if __name__ == "__main__":
    run_step1()
