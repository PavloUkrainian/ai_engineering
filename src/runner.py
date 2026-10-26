"""
Головний модуль запуску експериментів (Orchestrator).
Виконує:
- Прогін комбінацій: Модель x Конфігурація семплінгу x Версія промпту
- Валідацію з Self-Correction для кожної відповіді
- Логування моделі, версії промпту та параметрів семплінгу до кожного результату
- Збереження результатів у каталог results/
- Розрахунок підсумкових метрик
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.client import BaseLLMClient, ProviderClient, LocalClient
from src.config import SamplingConfig, CONFIG_DETERMINISTIC, CONFIG_SAMPLING
from src.metrics import compute_metrics
from src.validator import validate_and_repair

logger = logging.getLogger(__name__)


def load_dataset() -> List[Dict[str, Any]]:
    with open("data/raw_documents.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_golden_set() -> List[Dict[str, Any]]:
    with open("data/golden_set.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt(version: str) -> str:
    path = Path("prompts") / f"{version}.txt"
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def run_experiment(
    client: BaseLLMClient,
    sampling: SamplingConfig,
    prompt_version: str,
    raw_docs: List[Dict[str, Any]],
    golden_set: List[Dict[str, Any]],
    results_dir: Path
) -> Dict[str, Any]:
    """
    Виконує один експериментальний прогін по всіх 30 текстах.
    """
    system_prompt = load_prompt(prompt_version)
    exp_id = f"{client.model_name}_{sampling.name}_{prompt_version}".replace("/", "_")
    logger.info(f"Запуск експерименту: {exp_id} ({len(raw_docs)} документів)...")

    run_records: List[Dict[str, Any]] = []

    for item in tqdm(raw_docs, desc=f"Run {exp_id}", leave=False):
        doc_id = item["id"]
        raw_text = item["raw_text"]

        # 1. Генерація через клієнт моделі
        llm_resp = client.generate(
            system_prompt=system_prompt,
            user_text=raw_text,
            sampling=sampling
        )

        # 2. Валідація з Self-Correction (Вимога Кроку 3)
        val_result = validate_and_repair(
            raw_content=llm_resp.content,
            client=client,
            system_prompt=system_prompt,
            user_text=raw_text,
            sampling=sampling
        )

        # 3. Логування метаданих (Вимога Кроку 3)
        record = {
            "doc_id": doc_id,
            "model": client.model_name,
            "prompt_version": prompt_version,
            "sampling": sampling.to_dict(),
            "latency_ms": llm_resp.latency_ms,
            "input_tokens": llm_resp.input_tokens,
            "output_tokens": llm_resp.output_tokens,
            "cost_usd": llm_resp.cost_usd,
            "is_mock": llm_resp.is_mock,
            "validation": {
                "valid": val_result.valid,
                "status": val_result.status,
                "repaired": val_result.repaired,
                "repair_attempts": val_result.repair_attempts,
                "error_message": val_result.error_message
            },
            "extracted": val_result.document.model_dump() if val_result.document else None
        }
        run_records.append(record)

    # 4. Розрахунок метрик
    metrics = compute_metrics(run_records, golden_set)

    # 5. Збереження результатів прогону у results/
    output_payload = {
        "experiment_id": exp_id,
        "model": client.model_name,
        "prompt_version": prompt_version,
        "sampling": sampling.to_dict(),
        "metrics": metrics,
        "records": run_records
    }

    out_file = results_dir / f"{exp_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    logger.info(f"Завершено {exp_id}. Валідність: {metrics.get('valid_rate_percent')}%, "
                f"Точність: {metrics.get('overall_field_accuracy_percent')}%, "
                f"Latency p50: {metrics.get('latency_p50_ms')} ms")

    return output_payload


def run_all_experiments() -> List[Dict[str, Any]]:
    """
    Запуск повної сітки експериментів згідно з вимогами ДЗ:
    Модель 1 (Provider / gpt-4o-mini) та Модель 2 (Local / llama3.2)
    x 2 конфігурації семплінгу (deterministic vs sampling)
    x 2 версії промптів (v1 baseline vs v2 improved)
    """
    raw_docs = load_dataset()
    golden_set = load_golden_set()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    provider_client = ProviderClient()
    local_client = LocalClient()

    experiment_plan = [
        # Крок 1 та Крок 2: Baseline (v1) на двох конфігураціях семплінгу
        (provider_client, CONFIG_DETERMINISTIC, "v1"),
        (provider_client, CONFIG_SAMPLING, "v1"),
        (local_client, CONFIG_DETERMINISTIC, "v1"),
        (local_client, CONFIG_SAMPLING, "v1"),

        # Крок 3: Improved prompt (v2) на найкращій конфігурації семплінгу
        (provider_client, CONFIG_DETERMINISTIC, "v2"),
        (local_client, CONFIG_DETERMINISTIC, "v2"),
    ]

    summaries = []
    for client, sampling, prompt_ver in experiment_plan:
        res = run_experiment(client, sampling, prompt_ver, raw_docs, golden_set, results_dir)
        summaries.append({
            "experiment_id": res["experiment_id"],
            "model": res["model"],
            "sampling_config": res["sampling"]["name"],
            "temperature": res["sampling"]["temperature"],
            "top_p": res["sampling"]["top_p"],
            "prompt_version": res["prompt_version"],
            "valid_rate_percent": res["metrics"]["valid_rate_percent"],
            "overall_field_accuracy_percent": res["metrics"]["overall_field_accuracy_percent"],
            "latency_p50_ms": res["metrics"]["latency_p50_ms"],
            "latency_p95_ms": res["metrics"]["latency_p95_ms"],
            "cost_per_1000_docs_usd": res["metrics"]["cost_per_1000_docs_usd"],
            "per_field_accuracy": res["metrics"]["per_field_accuracy_percent"]
        })

    summary_file = results_dir / "experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    return summaries
