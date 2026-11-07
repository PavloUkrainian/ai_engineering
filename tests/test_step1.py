"""
Unit та regression тести для Кроку 1 (Ingestion, Normalization, Deduplication, Chunking).
"""

import json
import unittest
from pathlib import Path

from src.ingestion import (
    TextCleaner,
    MarkdownParser,
    HTMLParser,
    PDFParser,
    IngestionPipeline,
    count_tokens
)
from src.chunking import StructureAwareChunker, FixedSizeChunker, compare_strategies
from src.build_corpus import CORPUS_DIR, build_corpus


class TestStep1Pipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Переконуємось, що тестовий корпус згенеровано
        if not CORPUS_DIR.exists() or len(list(CORPUS_DIR.glob("*.*"))) < 30:
            build_corpus()

    def test_text_cleaner_hyphenation(self):
        """Перевірка склеювання розірваних переносів слів."""
        dirty = "Цей доку-\n  мент містить іден-\nтифікаційні дані."
        cleaned = TextCleaner.clean(dirty)
        self.assertIn("документ", cleaned)
        self.assertIn("ідентифікаційні", cleaned)
        self.assertNotIn("доку-", cleaned)

    def test_text_cleaner_whitespace_and_pagination(self):
        """Перевірка очищення зайвих пробілів та колонтитулів."""
        dirty = "Текст\nPage 1 of 12\n\n- 2 -\n   Багато   пробілів.   \n\n\n\nКінець."
        cleaned = TextCleaner.clean(dirty)
        self.assertNotIn("Page 1 of 12", cleaned)
        self.assertNotIn("- 2 -", cleaned)
        self.assertIn("Багато пробілів.", cleaned)
        self.assertNotIn("\n\n\n", cleaned)

    def test_markdown_parser(self):
        """Перевірка коректності парсингу Markdown файлу."""
        md_file = CORPUS_DIR / "nbu_resolution_65_remote_kyc.md"
        doc = MarkdownParser.parse(md_file)
        self.assertEqual(doc.file_format, "md")
        self.assertEqual(doc.doc_type, "regulation")
        self.assertGreater(len(doc.sections), 0)
        self.assertTrue(any("Розділ II" in s.title for s in doc.sections))
        self.assertGreater(doc.clean_char_count, 100)
        self.assertEqual(len(doc.content_hash), 64)

    def test_html_parser(self):
        """Перевірка коректності парсингу семантичного HTML файлу."""
        html_file = CORPUS_DIR / "docai_image_resolution_dpi_standards.html"
        doc = HTMLParser.parse(html_file)
        self.assertEqual(doc.file_format, "html")
        self.assertEqual(doc.doc_type, "technical_guide")
        self.assertGreater(len(doc.sections), 0)
        self.assertTrue(any("300 DPI" in s.text for s in doc.sections))
        self.assertEqual(len(doc.content_hash), 64)

    def test_pdf_parser(self):
        """Перевірка коректності парсингу PDF документа."""
        pdf_file = CORPUS_DIR / "icao_doc9303_p3_td1.pdf"
        doc = PDFParser.parse(pdf_file)
        self.assertEqual(doc.file_format, "pdf")
        self.assertEqual(doc.doc_type, "standard")
        self.assertGreater(len(doc.sections), 0)
        self.assertTrue(any("ISO/IEC 7810" in s.text for s in doc.sections))
        self.assertEqual(len(doc.content_hash), 64)

    def test_ingestion_pipeline_deduplication(self):
        """Перевірка дедуплікації через контент-хеш SHA-256."""
        pipeline = IngestionPipeline(CORPUS_DIR)
        unique_docs, manifest = pipeline.run()

        # Має бути >= 30 унікальних документів
        self.assertGreaterEqual(len(unique_docs), 30)

        # Перевірка наявності хоча б одного виявленого дубліката
        duplicates = [m for m in manifest if m["is_duplicate"]]
        self.assertGreaterEqual(len(duplicates), 1)

        # Перевірка що дублікат не включено до unique_docs
        dup_ids = {d["doc_id"] for d in duplicates}
        for u in unique_docs:
            self.assertNotIn(u.doc_id, dup_ids)

    def test_structure_aware_chunker_metadata(self):
        """Перевірка повноти метаданих кожного згенерованого чанка."""
        pipeline = IngestionPipeline(CORPUS_DIR)
        unique_docs, _ = pipeline.run()

        chunker = StructureAwareChunker(target_max_tokens=400)
        chunks = chunker.chunk_document(unique_docs[0])

        self.assertGreater(len(chunks), 0)
        first_chunk = chunks[0]

        # Обов'язкові метадані за критеріями Кроку 1
        self.assertTrue(first_chunk.chunk_id)
        self.assertTrue(first_chunk.source_file)
        self.assertTrue(first_chunk.doc_title)
        self.assertTrue(first_chunk.section_title)
        self.assertTrue(first_chunk.doc_type)
        self.assertTrue(first_chunk.version_date)
        self.assertEqual(len(first_chunk.doc_hash), 64)
        self.assertEqual(len(first_chunk.chunk_hash), 64)
        self.assertGreater(first_chunk.token_count, 0)
        self.assertEqual(first_chunk.strategy, "structure_aware")

    def test_chunking_comparison(self):
        """Перевірка порівняльного аналізу стратегій чанкінгу."""
        pipeline = IngestionPipeline(CORPUS_DIR)
        unique_docs, _ = pipeline.run()

        comparison = compare_strategies(unique_docs[:5])
        struct = comparison["structure_aware"]
        fixed = comparison["fixed_size"]

        # Structure-aware має мати суттєво вищу цілісність речень та розділів
        self.assertGreater(struct["sentence_integrity_rate"], fixed["sentence_integrity_rate"])
        self.assertEqual(struct["section_context_rate"], 1.0)
        self.assertEqual(fixed["section_context_rate"], 0.0)

    def test_artifacts_exist(self):
        """Перевірка наявності сформованих файлів chunks.jsonl та manifest.json."""
        manifest_p = Path("data/manifest.json")
        chunks_p = Path("data/chunks.jsonl")

        self.assertTrue(manifest_p.exists(), "data/manifest.json має існувати")
        self.assertTrue(chunks_p.exists(), "data/chunks.jsonl має існувати")

        # Перевірка валідності JSON
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            self.assertGreaterEqual(len(manifest_data), 30)

        # Перевірка JSONL
        with open(chunks_p, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            self.assertGreaterEqual(len(lines), 30)
            first_c = json.loads(lines[0])
            self.assertIn("chunk_id", first_c)
            self.assertIn("section_title", first_c)


if __name__ == "__main__":
    unittest.main()
