"""
Головна точка входу для запуску проєкту однією командою.
Використання:
    python main.py
    або
    make run
"""

import argparse
import logging
import sys
from pathlib import Path

from src.client import ProviderClient
from src.config import CONFIG_DETERMINISTIC
from src.runner import run_all_experiments, run_experiment, load_dataset, load_golden_set

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")


def print_table(summaries):
    """Красивий вивід зведеної таблиці результатів у термінал."""
    print("\n" + "=" * 115)
    print(f"{'Модель':<14} | {'Версія':<7} | {'Семплінг':<14} | {'Валідність':<11} | {'Точність':<10} | {'Latency p50':<12} | {'Latency p95':<12} | {'Вартість/1k':<12}")
    print("-" * 115)
    for s in summaries:
        samp_str = f"T={s['temperature']}, p={s['top_p']}"
        val_str = f"{s['valid_rate_percent']}%"
        acc_str = f"{s['overall_field_accuracy_percent']}%"
        p50_str = f"{s['latency_p50_ms']} ms"
        p95_str = f"{s['latency_p95_ms']} ms"
        cost_str = f"${s['cost_per_1000_docs_usd']:.4f}"
        print(f"{s['model']:<14} | {s['prompt_version']:<7} | {samp_str:<14} | {val_str:<11} | {acc_str:<10} | {p50_str:<12} | {p95_str:<12} | {cost_str:<12}")
    print("=" * 115 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Engineering: Document Information Extraction Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Запустити конкретний крок (1, 2, 3 або 4). За замовчуванням запускаються всі.")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    if args.step == 1:
        logger.info("=== Виконання Кроку 1: Схема + Baseline-промпт (v1) на Provider моделі ===")
        raw_docs = load_dataset()
        golden_set = load_golden_set()
        client = ProviderClient()
        res = run_experiment(client, CONFIG_DETERMINISTIC, "v1", raw_docs, golden_set, results_dir)
        print("\nРезультати Кроку 1:")
        print(f"- Оброблено документів: {len(raw_docs)}")
        print(f"- Валідність за схемою: {res['metrics']['valid_rate_percent']}%")
        print(f"- Точність на Golden Set: {res['metrics']['overall_field_accuracy_percent']}%")
        print(f"- Latency p50: {res['metrics']['latency_p50_ms']} ms")
        print(f"- Файл збережено в: results/{res['experiment_id']}.json")
        return

    logger.info("=== Запуск повної сітки експериментів (Кроки 1, 2, 3) ===")
    summaries = run_all_experiments()
    print_table(summaries)
    logger.info("Всі результати збережено в каталозі results/.")
    logger.info("Для перегляду детального аналізу та вибору моделі дивіться REPORT.md.")


if __name__ == "__main__":
    main()
