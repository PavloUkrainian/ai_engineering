"""
Модуль LLM-as-a-Judge для оцінки правильності відповідей з механізмом калібрування.
Реалізує:
1. Оцінку Answer Correctness (0.0 .. 1.0) на основі Ground Truth.
2. Калібрування судді на 10+ зразках, оцінених людиною вручну (розрахунок Agreement Rate та MAE).
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.client import BaseLLMClient, ProviderClient
from src.config import SamplingConfig, CONFIG_DETERMINISTIC

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path("evals/calibration_sample.json")

JUDGE_SYSTEM_PROMPT = """Ти — безсторонній та суворий суддя якості відповідей RAG-системи (LLM-as-a-Judge).
Твоє завдання — порівняти ВІДПОВІДЬ_СИСТЕМИ з ЕТАЛОННОЮ_ВІДПОВІДДЮ (Ground Truth) на запитання користувача.

КРИТЕРІЇ ТА РУБРИКА ОЦІНЮВАННЯ (шкала 0.0 до 1.0):
- 1.0: Відповідь повністю правильна, містить усі ключові факти/цифри та не містить суперечностей чи вигадок. Для питань без відповіді (unanswerable) — надано чітку відмову ("У наданих документах немає цієї інформації").
- 0.7 - 0.9: Відповідь правильна по суті, але пропущено незначну другорядну деталь, або присутнє дрібне стилістичне спрощення.
- 0.4 - 0.6: Відповідь частково правильна, але пропущено важливу умову/поріг, або відповідь неповна.
- 0.0 - 0.3: Відповідь неправильна, суперечить еталону, містить галюцинації (вигадування фактів замість відмови на unanswerable питання) або хибне твердження на питання-пастку.

ФОРМАТ ВІДПОВІДІ (СТРОГО JSON):
{
  "score": 0.0,
  "reasoning": "Коротке обґрунтування оцінки українською мовою."
}
"""


@dataclass
class JudgeResult:
    score: float
    reasoning: str


class LLMJudge:
    """Суддя на базі LLM з підтримкою калібрування."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or ProviderClient()
        self.sampling = CONFIG_DETERMINISTIC

    def evaluate(self, question: str, ground_truth: str, candidate_answer: str) -> JudgeResult:
        """Оцінює правильність однієї відповіді."""
        user_prompt = (
            f"ЗАПИТАННЯ:\n{question}\n\n"
            f"ЕТАЛОННА ВІДПОВІДЬ (Ground Truth):\n{ground_truth}\n\n"
            f"ВІДПОВІДЬ_СИСТЕМИ:\n{candidate_answer}\n\n"
            f"Оціни відповідь строго за рубрикою та поверни результат у форматі JSON."
        )

        try:
            resp = self.llm_client.generate_text(
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                sampling=self.sampling,
                response_format={"type": "json_object"}
            )
            content = resp.content.strip()
            data = json.loads(content)
            score = float(data.get("score", 0.0))
            reasoning = str(data.get("reasoning", ""))
            return JudgeResult(score=round(score, 2), reasoning=reasoning)
        except Exception as e:
            logger.warning(f"Збій LLM-судді ({e}). Застосування евристичного фолбеку.")
            # Евристичний fallback: порівняння перекриття слів
            gt_words = set(ground_truth.lower().split())
            cand_words = set(candidate_answer.lower().split())
            overlap = len(gt_words & cand_words) / max(1, len(gt_words))
            score = round(min(1.0, overlap), 2)
            return JudgeResult(score=score, reasoning=f"Fallback scoring через помилку: {e}")

    def run_calibration(self, calibration_file: Path = CALIBRATION_FILE) -> Dict[str, Any]:
        """
        Запускає калібрування судді на 10+ розмічених людиною зразках.
        Обчислює Agreement Rate (% збігів у межах допуску +/- 0.15) та MAE.
        """
        if not calibration_file.exists():
            raise FileNotFoundError(f"Файл калібрування не знайдено: {calibration_file}")

        with open(calibration_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        judge_scores = []
        human_scores = []
        sample_results = []
        agreement_count = 0
        tolerance = 0.15  # Допуск узгодженості між суддею та людиною

        for s in samples:
            res = self.evaluate(
                question=s["question"],
                ground_truth=s["ground_truth"],
                candidate_answer=s["candidate_answer"]
            )
            h_score = float(s["human_correctness_score"])
            j_score = res.score
            diff = abs(j_score - h_score)

            is_agreed = diff <= tolerance
            if is_agreed:
                agreement_count += 1

            judge_scores.append(j_score)
            human_scores.append(h_score)
            sample_results.append({
                "id": s["id"],
                "human_score": h_score,
                "judge_score": j_score,
                "diff": round(diff, 2),
                "agreed": is_agreed,
                "judge_reasoning": res.reasoning
            })

        total = len(samples)
        mae = sum(abs(j - h) for j, h in zip(judge_scores, human_scores)) / total
        agreement_rate = (agreement_count / total) * 100.0

        # Розрахунок кореляції Пірсона (якщо дисперсія > 0)
        import numpy as np
        if np.std(judge_scores) > 1e-5 and np.std(human_scores) > 1e-5:
            corr = float(np.corrcoef(judge_scores, human_scores)[0, 1])
        else:
            corr = 1.0 if mae < 0.1 else 0.0

        return {
            "samples_count": total,
            "agreement_rate_percent": round(agreement_rate, 1),
            "mean_absolute_error": round(mae, 3),
            "pearson_correlation": round(corr, 3),
            "samples": sample_results
        }
