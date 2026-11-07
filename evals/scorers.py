"""
Детерміновані scoring-функції для автоматизованої оцінки RAG-відповідей.
Реалізує:
1. CitationScorer: точність знаходження та зазначення правильних першоджерел.
2. KeyFactsScorer: наявність обов'язкових ключових фактів, цифр та вимог.
3. RefusalScorer: точність відмов для unanswerable запитань та відсутність хибних відмов.
4. FaithfulnessScorer: обґрунтованість тверджень відповіді наданим контекстом.
"""

import re
from typing import List, Dict, Any, Tuple
from src.rag_pipeline import RAGResponse, Citation


class CitationScorer:
    """Оцінює точність цитування першоджерел."""

    @staticmethod
    def score(response: RAGResponse, expected_sources: List[str], is_unanswerable: bool) -> float:
        if is_unanswerable:
            # Для unanswerable питань цитувань не повинно бути
            return 1.0 if not response.citations or response.is_refusal else 0.0

        if not expected_sources:
            return 1.0

        if not response.citations:
            return 0.0

        # Перевірка збігу хоча б одного очікуваного джерела
        matched = 0
        for exp in expected_sources:
            exp_lower = exp.lower()
            for cit in response.citations:
                if exp_lower in cit.doc_title.lower() or exp_lower in cit.section_title.lower():
                    matched += 1
                    break

        return min(1.0, matched / max(1, len(expected_sources)))


class KeyFactsScorer:
    """Оцінює наявність ключових фактів, цифр та слів у відповіді."""

    @staticmethod
    def score(response: RAGResponse, key_facts: List[str], is_unanswerable: bool) -> float:
        if is_unanswerable:
            # Для unanswerable головний факт — наявність відмови
            return 1.0 if response.is_refusal else 0.0

        if not key_facts:
            return 1.0

        answer_lower = response.answer.lower()
        matches = 0
        for fact in key_facts:
            fact_lower = fact.lower().strip()
            if fact_lower in answer_lower:
                matches += 1

        return matches / len(key_facts)


class RefusalScorer:
    """Оцінює точність спрацьовування механізму «Не знаю»."""

    @staticmethod
    def score(response: RAGResponse, is_unanswerable: bool) -> float:
        if is_unanswerable:
            # Для запитань без відповіді має бути СТРОГО відмова
            return 1.0 if response.is_refusal else 0.0
        else:
            # Для фактологічних запитань відмови НЕ повинно бути
            return 1.0 if not response.is_refusal else 0.0


class FaithfulnessScorer:
    """
    Оцінює рівень обґрунтованості (Faithfulness) тверджень відповіді наданим контекстом.
    Перевіряє, що відповідь не містить галюцинацій та спирається на наданий контекст:
    1. Для відмов («не знаю») — 1.0 (галюцинацій немає, система коректно відмовилась).
    2. Для змістовних відповідей:
       - Наявність inline-посилань [Документ, Розділ] у тексті відповіді.
       - Валідність цитат: чи всі процитовані джерела належать до наданого контексту.
       - Числове та фактологічне заземлення: перевірка, що викладені числа, дати, пороги
         та технічні коди (найвразливіші до галюцинацій) присутні в контексті.
    """

    @staticmethod
    def score(response: RAGResponse, context_text: str) -> float:
        if response.is_refusal:
            # Якщо система чесно відмовилась відповідати — галюцинацій немає (100% faithful)
            return 1.0

        answer = response.answer.strip()
        if not answer:
            return 0.0

        # Вирізаємо підсумковий блок "Джерела:"
        clean_body = re.split(r'(?i)\n*джерела:\s*', answer)[0].strip()

        # 1. Наявність inline-посилань у тілі відповіді
        has_inline_citations = bool(re.search(r'\[([^\]\n]+?),\s*([^\]\n]+?)\]', clean_body))
        body_citation_score = 1.0 if has_inline_citations else (0.5 if response.citations else 0.0)

        # 2. Валідність цитат відносно наданого контексту
        if response.citations:
            context_lower = context_text.lower()
            valid_cits = 0
            for cit in response.citations:
                doc_norm = cit.doc_title.lower()
                # Перевіряємо, чи згадується це джерело в контексті
                if (doc_norm[:20] in context_lower or 
                    any(w in context_lower for w in re.findall(r'[\w\u0400-\u04FF]{4,}', doc_norm))):
                    valid_cits += 1
            citation_grounding = valid_cits / len(response.citations)
        else:
            citation_grounding = 0.0

        # 3. Числове та сутнісне заземлення (запобігання вигадуванню кількісних метрик, дат, порогів)
        numbers = set(re.findall(r'\b\d+(?:[\.,]\d+)?\b', clean_body))
        meaningful_numbers = {n for n in numbers if len(n) > 1 or n in ["7", "3", "1", "2"]}

        if meaningful_numbers and context_text:
            grounded_numbers = sum(1 for n in meaningful_numbers if n in context_text)
            numerical_grounding = grounded_numbers / len(meaningful_numbers)
        else:
            numerical_grounding = 1.0

        # 4. Комбінований скор faithfulness (0.4 валідність цитат + 0.4 числове заземлення + 0.2 цитування в тілі)
        score = 0.4 * citation_grounding + 0.4 * numerical_grounding + 0.2 * body_citation_score
        return round(score, 3)
