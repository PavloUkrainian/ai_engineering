# AML/CFT Compliance Manual: Automated Sanctions and PEP Screening Procedures

> **Document Type:** policy  
> **Effective Date / Version:** 2023-10-12  

---

## 1. Mandatory Sanction Lists Repositories

FinTech automated onboarding pipelines must screen every applicant against: 1) Ukrainian National Security and Defense Council (РНБО) sanction lists; 2) US Office of Foreign Assets Control (OFAC) SDN list; 3) European Union Consolidated Financial Sanctions list; 4) United Nations Security Council Sanctions Committee database.

## 2. Fuzzy Matching Thresholds and False Positive Rules

Screening engines must utilize phonetically-aware fuzzy string matching (Jaro-Winkler, Levenshtein distance). Matches with a similarity score >= 0.85 (85%) must automatically freeze automated onboarding and route the case to Tier-2 Compliance Officer for manual investigation.
