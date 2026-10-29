"""
Конфігурація проєкту: параметри моделей, семплінгу та тарифікації.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SamplingConfig:
    """Параметри семплінгу для моделі."""
    name: str
    temperature: float
    top_p: float
    max_tokens: int = 1500

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


# Дві конфігурації семплінгу згідно з вимогою Кроку 2
CONFIG_DETERMINISTIC = SamplingConfig(
    name="deterministic",
    temperature=0.0,
    top_p=1.0
)

CONFIG_SAMPLING = SamplingConfig(
    name="sampling",
    temperature=0.7,
    top_p=0.9
)


@dataclass
class ModelCostConfig:
    """Вартість за 1M токенів (USD)."""
    input_per_million: float
    output_per_million: float


# Офіційні тарифи OpenAI gpt-4o-mini: $0.15 / 1M input, $0.60 / 1M output
PROVIDER_PRICING = {
    "gpt-4o-mini": ModelCostConfig(input_per_million=0.15, output_per_million=0.60),
    "gpt-4o": ModelCostConfig(input_per_million=2.50, output_per_million=10.00),
    "llama3.2": ModelCostConfig(input_per_million=0.0, output_per_million=0.0),  # Локальна = безкоштовно
}


@dataclass
class AppSettings:
    """Глобальні налаштування додатку."""
    # Провайдерська модель
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    provider_model: str = os.getenv("PROVIDER_MODEL", "gpt-4o-mini")

    # Локальна модель (Ollama / Local OpenAI-compatible)
    local_base_url: str = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
    local_model: str = os.getenv("LOCAL_MODEL", "llama3.2")
    local_api_key: str = os.getenv("LOCAL_API_KEY", "ollama")

    # Режим автоматичного мок-фолбеку при відсутності активного API ключа чи вимкненому Ollama
    # Mock execution mode (strictly for offline CI/smoke-tests; default is false for production benchmarking)
    use_mock_fallback: bool = os.getenv("USE_MOCK_FALLBACK", "false").lower() in ("true", "1", "yes")

    # Таймаути та повтори (Retry / Timeout)
    request_timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60.0"))
    local_request_timeout_seconds: float = float(os.getenv("LOCAL_REQUEST_TIMEOUT_SECONDS", "120.0"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))


settings = AppSettings()
