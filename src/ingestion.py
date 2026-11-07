"""
Модуль завантаження, парсингу, очищення та дедуплікації документів (Ingestion Pipeline).
Підтримує формати PDF, Markdown, HTML.
Зберігає логічну структуру (заголовки, метадані, секції) та дедуплікує контент через SHA-256.
"""

import hashlib
import html
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from pypdf import PdfReader
import tiktoken

logger = logging.getLogger(__name__)

# Ініціалізація токенізатора для підрахунку токенів
try:
    _tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    _tokenizer = None


def count_tokens(text: str) -> int:
    """Підрахунок кількості токенів у тексті."""
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text))
        except Exception:
            pass
    # Наближена оцінка: ~4 символи на токен
    return max(1, len(text) // 4)


@dataclass
class DocumentSection:
    """Структурний розділ або стаття документа."""
    title: str
    text: str
    level: int = 2


@dataclass
class IngestedDocument:
    """Результат обробки та очищення вихідного документа."""
    doc_id: str
    title: str
    doc_type: str
    version_date: str
    source_file: str
    file_format: str
    content_hash: str
    clean_text: str
    sections: List[DocumentSection] = field(default_factory=list)
    raw_char_count: int = 0
    clean_char_count: int = 0
    token_count: int = 0
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sections"] = [asdict(s) for s in self.sections]
        return d


class TextCleaner:
    """Нормалізація та очищення тексту від артефактів сканування/розмітки."""

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # 1. Нормалізація нерозривних пробілів та керуючих символів
        text = text.replace("\u00a0", " ").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")

        # 2. Склеювання розірваних переносів слів на межі рядків (наприклад, "доку-\nмент" -> "документ")
        text = re.sub(r'(\b[\w\u0400-\u04FF]+)-\s*\n\s*([\w\u0400-\u04FF]+\b)', r'\1\2', text)

        # 3. Видалення типових колонтитулів та номерів сторінок (наприклад, "Page 1 of 5", "- 2 -")
        text = re.sub(r'(?i)\n\s*page\s+\d+(\s+of\s+\d+)?\s*\n', '\n', text)
        text = re.sub(r'\n\s*-\s*\d+\s*-\s*\n', '\n', text)

        # 4. Нормалізація повторюваних горизонтальних пробілів та табуляцій
        text = re.sub(r'[ \t]+', ' ', text)

        # 5. Нормалізація надлишкових порожніх рядків (не більше двох підряд)
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()


class MarkdownParser:
    """Парсер документів у форматі Markdown (.md)."""

    @classmethod
    def parse(cls, filepath: Path) -> IngestedDocument:
        raw_text = filepath.read_text(encoding="utf-8")
        clean_full = TextCleaner.clean(raw_text)

        lines = clean_full.split("\n")
        title = filepath.stem.replace("_", " ").title()
        doc_type = "document"
        version_date = "unknown"

        sections: List[DocumentSection] = []
        current_sec_title = "Вступ"
        current_sec_lines: List[str] = []

        for line in lines:
            line_str = line.strip()

            # Вилучення головного заголовка H1
            if line_str.startswith("# ") and title == filepath.stem.replace("_", " ").title():
                title = line_str[2:].strip()
                continue

            # Вилучення метаданих з блокквотів
            if line_str.startswith("> **Document Type:**"):
                m = re.search(r'\*\*Document Type:\*\*\s*([^\s\*]+)', line_str)
                if m:
                    doc_type = m.group(1).strip()
                continue
            if line_str.startswith("> **Effective Date / Version:**"):
                m = re.search(r'\*\*Effective Date / Version:\*\*\s*([^\s\*]+)', line_str)
                if m:
                    version_date = m.group(1).strip()
                continue

            # Розділи другого та третього рівнів (##, ###)
            if line_str.startswith("## ") or line_str.startswith("### "):
                if current_sec_lines:
                    sec_text = "\n".join(current_sec_lines).strip()
                    if sec_text:
                        sections.append(DocumentSection(title=current_sec_title, text=sec_text))
                    current_sec_lines = []

                if line_str.startswith("## "):
                    current_sec_title = line_str[3:].strip()
                else:
                    current_sec_title = line_str[4:].strip()
                continue

            # Пропуск розділювальних ліній
            if line_str in ("---", "***", "___"):
                continue

            current_sec_lines.append(line)

        if current_sec_lines:
            sec_text = "\n".join(current_sec_lines).strip()
            if sec_text:
                sections.append(DocumentSection(title=current_sec_title, text=sec_text))

        # Формування цільного очищеного тексту
        combined_text = "\n\n".join([f"{s.title}:\n{s.text}" for s in sections]) if sections else clean_full
        content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

        return IngestedDocument(
            doc_id=filepath.stem,
            title=title,
            doc_type=doc_type,
            version_date=version_date,
            source_file=str(filepath),
            file_format="md",
            content_hash=content_hash,
            clean_text=combined_text,
            sections=sections,
            raw_char_count=len(raw_text),
            clean_char_count=len(combined_text),
            token_count=count_tokens(combined_text)
        )


class HTMLParser:
    """Парсер семантичних документів у форматі HTML (.html)."""

    @classmethod
    def parse(cls, filepath: Path) -> IngestedDocument:
        raw_text = filepath.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_text, "html.parser")

        # Видалення непотрібних тегів
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        title = filepath.stem.replace("_", " ").title()
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        meta_type = soup.find("meta", attrs={"name": "doc-type"})
        doc_type = meta_type["content"].strip() if meta_type and meta_type.get("content") else "document"

        meta_date = soup.find("meta", attrs={"name": "version-date"})
        version_date = meta_date["content"].strip() if meta_date and meta_date.get("content") else "unknown"

        sections: List[DocumentSection] = []
        html_sections = soup.find_all(["section", "article"])
        if html_sections:
            for s in html_sections:
                h = s.find(["h2", "h3", "h4"])
                sec_title = h.get_text(strip=True) if h else "Розділ"
                # Текст секції без самого заголовка
                paragraphs = [p.get_text(strip=True) for p in s.find_all(["p", "li", "table"])]
                sec_text = TextCleaner.clean("\n".join(paragraphs))
                if sec_text:
                    sections.append(DocumentSection(title=sec_title, text=sec_text))
        else:
            # Fallback: парсинг по h2
            h2_tags = soup.find_all("h2")
            if h2_tags:
                for h2 in h2_tags:
                    sec_title = h2.get_text(strip=True)
                    text_parts = []
                    curr = h2.next_sibling
                    while curr and getattr(curr, "name", None) != "h2":
                        if getattr(curr, "name", None) in ["p", "ul", "ol", "table"]:
                            text_parts.append(curr.get_text(strip=True))
                        curr = curr.next_sibling
                    sec_text = TextCleaner.clean("\n".join(text_parts))
                    if sec_text:
                        sections.append(DocumentSection(title=sec_title, text=sec_text))
            else:
                body_text = TextCleaner.clean(soup.get_text())
                sections.append(DocumentSection(title="Основний текст", text=body_text))

        combined_text = "\n\n".join([f"{s.title}:\n{s.text}" for s in sections])
        content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

        return IngestedDocument(
            doc_id=filepath.stem,
            title=title,
            doc_type=doc_type,
            version_date=version_date,
            source_file=str(filepath),
            file_format="html",
            content_hash=content_hash,
            clean_text=combined_text,
            sections=sections,
            raw_char_count=len(raw_text),
            clean_char_count=len(combined_text),
            token_count=count_tokens(combined_text)
        )


class PDFParser:
    """Парсер документів у форматі PDF (.pdf)."""

    @classmethod
    def parse(cls, filepath: Path) -> IngestedDocument:
        reader = PdfReader(str(filepath))
        raw_pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                raw_pages.append(t)

        raw_text = "\n\n".join(raw_pages)
        clean_full = TextCleaner.clean(raw_text)

        lines = clean_full.split("\n")
        title = filepath.stem.replace("_", " ").title()
        doc_type = "standard"
        version_date = "unknown"

        # Витягування заголовка та метаданих із перших рядків
        if lines:
            title = lines[0].strip()

        for l in lines[1:5]:
            if "Type:" in l and "Date:" in l:
                m_type = re.search(r'Type:\s*([^\s\|]+)', l)
                m_date = re.search(r'Date:\s*([^\s\|]+)', l)
                if m_type:
                    doc_type = m_type.group(1).strip()
                if m_date:
                    version_date = m_date.group(1).strip()

        # Виявлення розділів за нумерацією (наприклад, "1. Scope", "2. Visual Inspection")
        sections: List[DocumentSection] = []
        current_title = "Вступ"
        current_lines: List[str] = []

        section_header_pattern = re.compile(r'^\d+\.\s+[A-Za-zА-Яа-яЇїІіЄєҐґ0-9\s,\-\(\):]+$')

        for line in lines[1:]:
            l_str = line.strip()
            if "Type:" in l_str and "Date:" in l_str:
                continue

            if section_header_pattern.match(l_str):
                if current_lines:
                    sec_text = "\n".join(current_lines).strip()
                    if sec_text:
                        sections.append(DocumentSection(title=current_title, text=sec_text))
                    current_lines = []
                current_title = l_str
                continue

            current_lines.append(line)

        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                sections.append(DocumentSection(title=current_title, text=sec_text))

        combined_text = "\n\n".join([f"{s.title}:\n{s.text}" for s in sections]) if sections else clean_full
        content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

        return IngestedDocument(
            doc_id=filepath.stem,
            title=title,
            doc_type=doc_type,
            version_date=version_date,
            source_file=str(filepath),
            file_format="pdf",
            content_hash=content_hash,
            clean_text=combined_text,
            sections=sections,
            raw_char_count=len(raw_text),
            clean_char_count=len(combined_text),
            token_count=count_tokens(combined_text)
        )


class IngestionPipeline:
    """Головний оркестратор ingestion: читання каталогу, очищення та дедуплікація."""

    def __init__(self, corpus_dir: Path):
        self.corpus_dir = Path(corpus_dir)
        self.seen_hashes: Dict[str, str] = {}  # hash -> first_doc_id

    def run(self) -> Tuple[List[IngestedDocument], List[dict]]:
        """
        Запускає ingestion по всіх файлах каталогу.
        Повертає список унікальних валідних документів та повний маніфест.
        """
        if not self.corpus_dir.exists():
            raise FileNotFoundError(f"Каталог корпусу не знайдено: {self.corpus_dir}")

        files = sorted(list(self.corpus_dir.iterdir()))
        unique_documents: List[IngestedDocument] = []
        manifest: List[dict] = []

        for f in files:
            if f.name.startswith("."):
                continue

            doc: Optional[IngestedDocument] = None
            ext = f.suffix.lower()

            try:
                if ext == ".md":
                    doc = MarkdownParser.parse(f)
                elif ext in (".html", ".htm"):
                    doc = HTMLParser.parse(f)
                elif ext == ".pdf":
                    doc = PDFParser.parse(f)
                else:
                    logger.debug(f"Пропуск непідтримуваного формату: {f.name}")
                    continue
            except Exception as e:
                logger.error(f"Помилка парсингу {f.name}: {e}")
                continue

            if doc is None:
                continue

            # Дедуплікація за SHA-256 хешем очищеного тексту
            if doc.content_hash in self.seen_hashes:
                original_id = self.seen_hashes[doc.content_hash]
                doc.is_duplicate = True
                doc.duplicate_of = original_id
                logger.info(f"⚠️ Виявлено дублікат: {doc.doc_id} (ідентичний до {original_id})")
            else:
                self.seen_hashes[doc.content_hash] = doc.doc_id
                unique_documents.append(doc)

            manifest_entry = {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "file_format": doc.file_format,
                "doc_type": doc.doc_type,
                "version_date": doc.version_date,
                "source_file": doc.source_file,
                "content_hash": doc.content_hash,
                "clean_char_count": doc.clean_char_count,
                "token_count": doc.token_count,
                "sections_count": len(doc.sections),
                "is_duplicate": doc.is_duplicate,
                "duplicate_of": doc.duplicate_of
            }
            manifest.append(manifest_entry)

        return unique_documents, manifest
