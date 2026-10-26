"""
Модуль валідації за схемою Pydantic та механізм Self-Correction (Вимога Кроку 3).
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from pydantic import ValidationError

from src.config import SamplingConfig
from src.schema import ExtractedDocument

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Результат валідації відповіді моделі."""
    valid: bool
    status: str                         # "success", "repaired", "failed"
    repaired: bool
    document: Optional[ExtractedDocument]
    raw_content: str
    error_message: Optional[str] = None
    repair_attempts: int = 0


def validate_and_repair(
    raw_content: str,
    client,  # BaseLLMClient
    system_prompt: str,
    user_text: str,
    sampling: SamplingConfig
) -> ValidationResult:
    """
    Валідує JSON за Pydantic ExtractedDocument.
    Якщо вихід невалідний — виконує РІВНО ОДНУ спробу виправлення (Self-Correction),
    передаючи моделі текст виявленої помилки. Якщо знову невалідний — позначає як failed.
    """
    # 1. Перша спроба валідації
    is_valid, doc, err = _parse_and_validate(raw_content)
    if is_valid and doc:
        return ValidationResult(
            valid=True,
            status="success",
            repaired=False,
            document=doc,
            raw_content=raw_content
        )

    logger.warning(f"Первинна валідація не пройшла: {err}. Запуск 1 спроби Self-Correction...")

    # 2. Рівно 1 спроба виправлення (Вимога Кроку 3)
    repair_prompt = (
        f"{system_prompt}\n\n"
        f"ВАЖЛИВО: Твоя попередня відповідь була невалідною з наступною помилкою:\n{err}\n"
        f"Будь ласка, уважно виправ помилку та поверни ТІЛЬКИ чистий валідний JSON-об'єкт "
        f"відповідно до схеми ExtractedDocument."
    )

    try:
        repair_resp = client.generate(repair_prompt, user_text, sampling)
        repaired_valid, repaired_doc, repair_err = _parse_and_validate(repair_resp.content)

        if repaired_valid and repaired_doc:
            logger.info("Успішне виправлення (repaired) з 1-ї спроби!")
            return ValidationResult(
                valid=True,
                status="repaired",
                repaired=True,
                document=repaired_doc,
                raw_content=repair_resp.content,
                repair_attempts=1
            )
        else:
            logger.error(f"Повторна спроба також не вдалася: {repair_err}. Статус: failed.")
            return ValidationResult(
                valid=False,
                status="failed",
                repaired=False,
                document=None,
                raw_content=repair_resp.content,
                error_message=f"Initial: {err} | Repair: {repair_err}",
                repair_attempts=1
            )
    except Exception as exc:
        return ValidationResult(
            valid=False,
            status="failed",
            repaired=False,
            document=None,
            raw_content=raw_content,
            error_message=f"Crash during repair: {exc}",
            repair_attempts=1
        )


def _parse_and_validate(text: str) -> Tuple[bool, Optional[ExtractedDocument], Optional[str]]:
    """Парсинг JSON та валідація через Pydantic."""
    if not text or not text.strip():
        return False, None, "Порожня відповідь від моделі"

    # Якщо модель випадково загорнула в markdown ```json ... ```
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as jde:
        return False, None, f"JSONDecodeError: {jde}"

    try:
        doc = ExtractedDocument.model_validate(data)
        return True, doc, None
    except ValidationError as ve:
        return False, None, f"Pydantic ValidationError: {ve}"
