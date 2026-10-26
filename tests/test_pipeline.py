"""
Автоматизовані тести пайплайну (Unit & Regression Tests на Golden Set).
Запуск: python -m unittest discover tests
"""

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.client import ProviderClient, LocalClient, MockFallbackClient
from src.config import CONFIG_DETERMINISTIC, CONFIG_SAMPLING
from src.schema import ExtractedDocument, DocumentType, Gender
from src.validator import validate_and_repair


class TestSchemaValidation(unittest.TestCase):
    """Тестування валідації Pydantic-схеми ExtractedDocument."""

    def test_valid_document_schema(self):
        valid_data = {
            "document_type": "ID_CARD",
            "document_number": "002948172",
            "record_number": "19920514-01824",
            "holder": {
                "last_name_ukr": "КОВАЛЕНКО",
                "first_name_ukr": "ОЛЕКСІЙ",
                "patronymic_ukr": "ВАСИЛЬОВИЧ",
                "last_name_en": "KOVALENKO",
                "first_name_en": "OLEKSII"
            },
            "gender": "MALE",
            "dates": {
                "birth_date": "1992-05-14",
                "issue_date": None,
                "expiry_date": "2032-05-14"
            },
            "issuing_authority": "8011",
            "mrz_line": None,
            "ocr_quality_flags": ["GLARE"],
            "confidence_score": 0.95
        }
        doc = ExtractedDocument.model_validate(valid_data)
        self.assertEqual(doc.document_type, DocumentType.ID_CARD)
        self.assertEqual(doc.gender, Gender.MALE)
        self.assertEqual(doc.holder.last_name_ukr, "КОВАЛЕНКО")

    def test_invalid_enum_raises_error(self):
        invalid_data = {
            "document_type": "NOT_A_REAL_TYPE",
            "document_number": "123456"
        }
        with self.assertRaises(ValidationError):
            ExtractedDocument.model_validate(invalid_data)


class TestGoldenSetIntegrity(unittest.TestCase):
    """Перевірка цілісності та валідності еталонного набору (Golden Set)."""

    def test_all_golden_set_records_are_valid(self):
        golden_file = Path("data/golden_set.json")
        self.assertTrue(golden_file.exists(), "golden_set.json має існувати")

        with open(golden_file, "r", encoding="utf-8") as f:
            items = json.load(f)

        self.assertGreaterEqual(len(items), 10, "Golden set має містити щонайменше 10 розмічених документів")

        for item in items:
            self.assertIn("id", item)
            self.assertIn("expected", item)
            # Кожен запис має успішно валідуватися схемою
            doc = ExtractedDocument.model_validate(item["expected"])
            self.assertIsInstance(doc.document_type, DocumentType)


class TestSelfCorrectionRepair(unittest.TestCase):
    """Тестування механізму Self-Correction при валідації."""

    def test_repair_fixes_invalid_type_in_one_attempt(self):
        client = MockFallbackClient(model_name="llama3.2", client_type="local")
        invalid_json = '{"document_type": "INVALID_TYPE", "document_number": "BXA 839201"}'

        result = validate_and_repair(
            raw_content=invalid_json,
            client=client,
            system_prompt="Ти асистент з вилучення документів. Схема ExtractedDocument.",
            user_text="ПОСВІДЧЕННЯ ВОДІЯ BXA 839201 БОНДАРЕНКО ДМИТРО 03.11.1985",
            sampling=CONFIG_DETERMINISTIC
        )

        self.assertTrue(result.valid, "Після Self-Correction відповідь має стати валідною")
        self.assertEqual(result.status, "repaired")
        self.assertEqual(result.repair_attempts, 1)
        self.assertIsNotNone(result.document)


class TestClientInterface(unittest.TestCase):
    """Перевірка спільного інтерфейсу BaseLLMClient для обох моделей."""

    def test_clients_produce_unified_llm_response(self):
        p_client = ProviderClient()
        l_client = LocalClient()

        prompt = "Extract document"
        text = "ПАСПОРТ СЕРІЯ ЕР 492018 КРАВЧУК МИХАЙЛО"

        p_resp = p_client.generate(prompt, text, CONFIG_DETERMINISTIC)
        l_resp = l_client.generate(prompt, text, CONFIG_DETERMINISTIC)

        for resp in [p_resp, l_resp]:
            self.assertIsNotNone(resp.content)
            self.assertIsInstance(resp.latency_ms, float)
            self.assertGreater(resp.latency_ms, 0.0)
            self.assertIsInstance(resp.sampling, dict)


if __name__ == "__main__":
    unittest.main()
