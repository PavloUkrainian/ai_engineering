"""
Модуль розбиття документів на чанки (Chunking Strategies).
Реалізує:
1. Structure-Aware (Semantic / Section-based) Chunking зі збереженням контексту розділів та метаданих.
2. Naive Fixed-Size Chunking (для експериментального порівняння).
3. Бенчмарк порівняння стратегій (втрата контексту, розриви речень, розподіл розмірів).
"""

import hashlib
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

from src.ingestion import IngestedDocument, DocumentSection, count_tokens


@dataclass
class Chunk:
    """Одиниця знань RAG-системи з повним набором метаданих."""
    chunk_id: str
    doc_id: str
    source_file: str
    doc_title: str
    section_title: str
    doc_type: str
    version_date: str
    doc_hash: str
    chunk_hash: str
    text: str
    token_count: int
    char_count: int
    chunk_index: int
    strategy: str

    def to_dict(self) -> dict:
        return asdict(self)


class StructureAwareChunker:
    """
    Семантичний чанкінг, що зберігає ієрархію розділів, статей та контекстні префікси.
    Цільовий розмір чанка: 150-400 токенів.
    """

    def __init__(self, target_max_tokens: int = 400, overlap_tokens: int = 50):
        self.target_max_tokens = target_max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, doc: IngestedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        chunk_idx = 0

        # Якщо у документа є виділені секції — чанкуємо по секціях
        sections = doc.sections if doc.sections else [DocumentSection(title="Основний текст", text=doc.clean_text)]

        for sec in sections:
            sec_text = sec.text.strip()
            if not sec_text:
                continue

            sec_tokens = count_tokens(sec_text)

            # Якщо секція компактна — залишаємо її цілісною одиницею
            if sec_tokens <= self.target_max_tokens:
                header_prefix = f"[{doc.title} > {sec.title}]\n"
                full_chunk_text = header_prefix + sec_text
                c_hash = hashlib.sha256(full_chunk_text.encode("utf-8")).hexdigest()
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}_ch{chunk_idx:03d}",
                    doc_id=doc.doc_id,
                    source_file=doc.source_file,
                    doc_title=doc.title,
                    section_title=sec.title,
                    doc_type=doc.doc_type,
                    version_date=doc.version_date,
                    doc_hash=doc.content_hash,
                    chunk_hash=c_hash,
                    text=full_chunk_text,
                    token_count=count_tokens(full_chunk_text),
                    char_count=len(full_chunk_text),
                    chunk_index=chunk_idx,
                    strategy="structure_aware"
                ))
                chunk_idx += 1
            else:
                # Розбиваємо довгу секцію по абзацах або реченнях з overlap
                sub_chunks = self._split_large_section(sec_text)
                for part_i, sub_text in enumerate(sub_chunks, 1):
                    header_prefix = f"[{doc.title} > {sec.title} (частина {part_i})]\n"
                    full_chunk_text = header_prefix + sub_text
                    c_hash = hashlib.sha256(full_chunk_text.encode("utf-8")).hexdigest()
                    chunks.append(Chunk(
                        chunk_id=f"{doc.doc_id}_ch{chunk_idx:03d}",
                        doc_id=doc.doc_id,
                        source_file=doc.source_file,
                        doc_title=doc.title,
                        section_title=f"{sec.title} (part {part_i})",
                        doc_type=doc.doc_type,
                        version_date=doc.version_date,
                        doc_hash=doc.content_hash,
                        chunk_hash=c_hash,
                        text=full_chunk_text,
                        token_count=count_tokens(full_chunk_text),
                        char_count=len(full_chunk_text),
                        chunk_index=chunk_idx,
                        strategy="structure_aware"
                    ))
                    chunk_idx += 1

        return chunks

    def _split_large_section(self, text: str) -> List[str]:
        """Розбиття великої секції по абзацах або реченнях."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        result = []
        current_part = []
        current_len = 0

        for p in paragraphs:
            p_tokens = count_tokens(p)
            if current_len + p_tokens > self.target_max_tokens and current_part:
                result.append("\n\n".join(current_part))
                # Додаємо overlap з останнього абзацу
                current_part = [current_part[-1], p] if count_tokens(current_part[-1]) <= self.overlap_tokens else [p]
                current_len = sum(count_tokens(x) for x in current_part)
            else:
                current_part.append(p)
                current_len += p_tokens

        if current_part:
            result.append("\n\n".join(current_part))

        return result if result else [text]


class FixedSizeChunker:
    """
    Наївний чанкінг фіксованого розміру (наприклад, 500 символів з перекриттям 50 символів).
    Ігнорує межі слів, заголовки та семантику. Використовується для порівняльного аналізу.
    """

    def __init__(self, chunk_size_chars: int = 500, overlap_chars: int = 50):
        self.chunk_size_chars = chunk_size_chars
        self.overlap_chars = overlap_chars

    def chunk_document(self, doc: IngestedDocument) -> List[Chunk]:
        raw_text = doc.clean_text
        chunks: List[Chunk] = []
        start = 0
        chunk_idx = 0
        step = max(1, self.chunk_size_chars - self.overlap_chars)

        while start < len(raw_text):
            end = min(start + self.chunk_size_chars, len(raw_text))
            chunk_slice = raw_text[start:end]
            c_hash = hashlib.sha256(chunk_slice.encode("utf-8")).hexdigest()

            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_fixed_{chunk_idx:03d}",
                doc_id=doc.doc_id,
                source_file=doc.source_file,
                doc_title=doc.title,
                section_title="[None - Fixed Window]",
                doc_type=doc.doc_type,
                version_date=doc.version_date,
                doc_hash=doc.content_hash,
                chunk_hash=c_hash,
                text=chunk_slice,
                token_count=count_tokens(chunk_slice),
                char_count=len(chunk_slice),
                chunk_index=chunk_idx,
                strategy="fixed_size"
            ))
            chunk_idx += 1
            if end >= len(raw_text):
                break
            start += step

        return chunks


def evaluate_chunking_strategy(chunks: List[Chunk]) -> Dict[str, Any]:
    """Розрахунок аналітичних метрик якості нарізки чанків."""
    if not chunks:
        return {}

    token_counts = [c.token_count for c in chunks]
    char_counts = [c.char_count for c in chunks]

    # Перевірка: чи закінчується чанк на повноцінне завершення речення (. ! ? або лапки)
    complete_sentence_ends = 0
    sentence_end_pattern = re.compile(r'[\.\!\?]["\'»]?\s*$')

    # Перевірка: чи містить чанк ясний контекст секції (а не фіксоване вікно)
    has_section_context = 0

    for c in chunks:
        if sentence_end_pattern.search(c.text.strip()):
            complete_sentence_ends += 1
        if c.section_title and c.section_title != "[None - Fixed Window]":
            has_section_context += 1

    mean_tokens = sum(token_counts) / len(token_counts)
    variance_tokens = sum((x - mean_tokens) ** 2 for x in token_counts) / len(token_counts)
    std_tokens = variance_tokens ** 0.5

    return {
        "total_chunks": len(chunks),
        "mean_tokens": round(mean_tokens, 1),
        "std_tokens": round(std_tokens, 1),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "mean_chars": round(sum(char_counts) / len(char_counts), 1),
        "sentence_integrity_rate": round(complete_sentence_ends / len(chunks), 3),
        "section_context_rate": round(has_section_context / len(chunks), 3)
    }


def compare_strategies(documents: List[IngestedDocument]) -> Dict[str, Any]:
    """Порівнює Structure-Aware та Fixed-Size чанкінг на наданих документах."""
    struct_chunker = StructureAwareChunker()
    fixed_chunker = FixedSizeChunker()

    all_struct_chunks: List[Chunk] = []
    all_fixed_chunks: List[Chunk] = []

    for doc in documents:
        all_struct_chunks.extend(struct_chunker.chunk_document(doc))
        all_fixed_chunks.extend(fixed_chunker.chunk_document(doc))

    eval_struct = evaluate_chunking_strategy(all_struct_chunks)
    eval_fixed = evaluate_chunking_strategy(all_fixed_chunks)

    return {
        "structure_aware": eval_struct,
        "fixed_size": eval_fixed,
        "struct_chunks": all_struct_chunks,
        "fixed_chunks": all_fixed_chunks
    }
