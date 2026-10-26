"""
Модуль розрахунку метрик якості та продуктивності (Вимога Кроку 2).
Метрики:
1. Точність по полях на golden set (field-level accuracy)
2. % валідних відповідей (% valid responses)
3. Latency p50 та p95
4. Вартість на 1000 документів (USD)
"""

from typing import Any, Dict, List
import numpy as np


def compute_metrics(
    all_results: List[Dict[str, Any]],
    golden_set: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Розрахунок комплексу метрик для одного прогону експерименту
    (конкретна модель x конфігурація семплінгу x версія промпту).
    """
    total_docs = len(all_results)
    if total_docs == 0:
        return {}

    # 1. % валідних відповідей
    valid_count = sum(1 for r in all_results if r.get("validation", {}).get("valid", False))
    valid_rate = (valid_count / total_docs) * 100.0

    # 2. Latency p50 / p95
    latencies = [r.get("latency_ms", 0.0) for r in all_results]
    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    mean_latency = float(np.mean(latencies))

    # 3. Вартість на 1000 документів
    total_cost = sum(r.get("cost_usd", 0.0) for r in all_results)
    avg_cost_per_doc = total_cost / total_docs
    cost_per_1000 = round(avg_cost_per_doc * 1000.0, 4)

    # 4. Точність по полях на Golden Set (10 документів)
    golden_map = {g["id"]: g["expected"] for g in golden_set}
    field_scores: Dict[str, List[int]] = {
        "document_type": [],
        "document_number": [],
        "record_number": [],
        "holder_last_name": [],
        "holder_first_name": [],
        "gender": [],
        "birth_date": [],
        "expiry_date": [],
        "issuing_authority": []
    }

    evaluated_golden_count = 0

    for r in all_results:
        doc_id = r.get("doc_id")
        if doc_id not in golden_map:
            continue

        evaluated_golden_count += 1
        expected = golden_map[doc_id]
        extracted = r.get("extracted") or {}

        # 1. document_type
        exp_type = expected.get("document_type")
        act_type = extracted.get("document_type")
        field_scores["document_type"].append(1 if exp_type == act_type else 0)

        # 2. document_number
        exp_num = _norm_str(expected.get("document_number"))
        act_num = _norm_str(extracted.get("document_number"))
        field_scores["document_number"].append(1 if exp_num == act_num else 0)

        # 3. record_number (УНЗР)
        exp_rec = _norm_str(expected.get("record_number"))
        act_rec = _norm_str(extracted.get("record_number"))
        field_scores["record_number"].append(1 if exp_rec == act_rec else 0)

        # 4. holder last_name (кирилиця або латиниця)
        exp_h = expected.get("holder", {})
        act_h = extracted.get("holder", {})
        exp_last = _norm_str(exp_h.get("last_name_ukr")) or _norm_str(exp_h.get("last_name_en"))
        act_last = _norm_str(act_h.get("last_name_ukr")) or _norm_str(act_h.get("last_name_en"))
        field_scores["holder_last_name"].append(1 if exp_last and act_last and (exp_last in act_last or act_last in exp_last) or exp_last == act_last else 0)

        # 5. holder first_name
        exp_first = _norm_str(exp_h.get("first_name_ukr")) or _norm_str(exp_h.get("first_name_en"))
        act_first = _norm_str(act_h.get("first_name_ukr")) or _norm_str(act_h.get("first_name_en"))
        field_scores["holder_first_name"].append(1 if exp_first and act_first and (exp_first in act_first or act_first in exp_first) or exp_first == act_first else 0)

        # 6. gender
        exp_g = expected.get("gender")
        act_g = extracted.get("gender")
        field_scores["gender"].append(1 if exp_g == act_g else 0)

        # 7. birth_date
        exp_d = expected.get("dates", {})
        act_d = extracted.get("dates", {})
        exp_b = _norm_str(exp_d.get("birth_date"))
        act_b = _norm_str(act_d.get("birth_date"))
        field_scores["birth_date"].append(1 if exp_b == act_b else 0)

        # 8. expiry_date
        exp_exp = _norm_str(exp_d.get("expiry_date"))
        act_exp = _norm_str(act_d.get("expiry_date"))
        field_scores["expiry_date"].append(1 if exp_exp == act_exp else 0)

        # 9. issuing_authority
        exp_auth = _norm_str(expected.get("issuing_authority"))
        act_auth = _norm_str(extracted.get("issuing_authority"))
        if not exp_auth and not act_auth:
            match_auth = 1
        elif exp_auth and act_auth and (exp_auth in act_auth or act_auth in exp_auth):
            match_auth = 1
        else:
            match_auth = 0
        field_scores["issuing_authority"].append(match_auth)

    per_field_accuracy = {
        field: round((sum(scores) / len(scores)) * 100.0, 1) if scores else 0.0
        for field, scores in field_scores.items()
    }

    all_accuracies = list(per_field_accuracy.values())
    overall_accuracy = round(sum(all_accuracies) / len(all_accuracies), 1) if all_accuracies else 0.0

    return {
        "total_documents": total_docs,
        "golden_evaluated": evaluated_golden_count,
        "valid_rate_percent": round(valid_rate, 1),
        "overall_field_accuracy_percent": overall_accuracy,
        "per_field_accuracy_percent": per_field_accuracy,
        "latency_p50_ms": round(p50_latency, 1),
        "latency_p95_ms": round(p95_latency, 1),
        "latency_mean_ms": round(mean_latency, 1),
        "total_cost_usd": round(total_cost, 6),
        "cost_per_1000_docs_usd": cost_per_1000
    }


def _norm_str(val: Any) -> str:
    """Нормалізація рядка для стійкого порівняння."""
    if val is None:
        return ""
    return str(val).strip().upper().replace(" ", "").replace("-", "").replace(".", "")
