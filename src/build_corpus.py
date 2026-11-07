"""
Генератор нормативно-технічного корпусу документів для DocAI / KYC / FinTech Compliance.
Створює >= 30 документів у 3 форматах (Markdown, HTML, PDF), повністю без персональних даних (No PII).
Включає 1 тестовий дублікат для перевірки дедуплікації.
"""

import html
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

CORPUS_DIR = Path("data/corpus")

# Опис 35 документів (теми, заголовки, розділи та зміст)
DOCUMENTS_DATA = [
    # --- 1. СТАНДАРТИ ICAO (PDF) ---
    {
        "filename": "icao_doc9303_p3_td1.pdf",
        "format": "pdf",
        "title": "ICAO Doc 9303 Part 3: Specifications for TD1 Size Machine Readable Official Travel Documents",
        "doc_type": "standard",
        "version_date": "2021-07-15",
        "sections": [
            {
                "title": "1. Scope and Dimensions of TD1 Documents",
                "text": "TD1 size machine readable cards must strictly adhere to ISO/IEC 7810 nominal dimensions: 85.60 mm in width, 53.98 mm in height, and a nominal thickness of 0.76 mm (tolerance +/- 0.08 mm). The card material must be high-durability polycarbonate (PC) or composite PET/PC capable of withstanding delamination tests up to 10 years."
            },
            {
                "title": "2. Visual Inspection Zone (VIZ) Layout",
                "text": "The Visual Inspection Zone on the front side contains nominal data elements: primary identifier (surname), secondary identifier (given names), nationality, date of birth (format DD MMM YYYY), sex (M, F, or X for unspecified), document number, and date of expiry. Font must be OCR-B or high-legibility sans-serif with a minimum height of 2.0 mm."
            },
            {
                "title": "3. Machine Readable Zone (MRZ) Structure for TD1",
                "text": "The TD1 MRZ consists of three lines of 30 characters each (total 90 characters), printed in OCR-B font at 10 characters per inch (pitch 2.54 mm). Line 1 contains document code (I, ID, IP, or IA), issuing state (3-letter ICAO code), and document number (9 chars + 1 check digit + optional 15 chars). Line 2 contains date of birth (YYMMDD + check digit), sex (M/F/<), date of expiry (YYMMDD + check digit), nationality (3-letter code), optional data, and composite check digit. Line 3 contains primary and secondary identifiers separated by double filler characters '<<'."
            }
        ]
    },
    {
        "filename": "icao_doc9303_p4_td3_passports.pdf",
        "format": "pdf",
        "title": "ICAO Doc 9303 Part 4: Specifications for Machine Readable Passports (TD3 Size)",
        "doc_type": "standard",
        "version_date": "2021-08-10",
        "sections": [
            {
                "title": "1. Physical Dimensions and Book Structure",
                "text": "A TD3 size passport booklet comprises a data page with dimensions of 125.0 mm x 88.0 mm (nominal, ISO/IEC 7810 ID-3). The biographical data page is embedded as either a flexible laminated paper sheet or an integrated rigid polycarbonate leaf fused directly into the spine binding with tamper-evident stitching."
            },
            {
                "title": "2. MRZ Structure for Passports (TD3 2x44)",
                "text": "The TD3 MRZ contains two lines of 44 characters each (total 88 characters). Line 1 format: positions 1-2 document code ('P<'), positions 3-5 issuing state, positions 6-44 holder's name (surname<<given_names). Line 2 format: positions 1-9 passport number, pos 10 check digit, positions 11-13 nationality code, positions 14-19 birth date (YYMMDD), pos 20 check digit, pos 21 sex, positions 22-27 expiry date (YYMMDD), pos 28 check digit, positions 29-42 personal/optional number, pos 43 check digit, pos 44 composite check digit."
            },
            {
                "title": "3. Security Thread and Optically Variable Devices",
                "text": "The biographical data page must incorporate at least three level-1 security features visible to the naked eye: an optically variable ink (OVI) displaying distinct colour-shift under varying angles of illumination, a holographic DOVID patch overlapping the edge of the facial portrait, and tactile laser-engraved micro-relief."
            }
        ]
    },
    {
        "filename": "icao_doc9303_check_digit_algorithms.pdf",
        "format": "pdf",
        "title": "ICAO Doc 9303 Appendix: Check Digit Calculation Rules for MRZ Zones",
        "doc_type": "technical_guide",
        "version_date": "2020-03-01",
        "sections": [
            {
                "title": "1. Weighting Factors and Modulo 10 Algorithm",
                "text": "The check digit calculation uses a repeating sequence of weighting factors: 7, 3, 1, 7, 3, 1... applied sequentially from left to right across all characters in the data element. The numerical value of each character is multiplied by its corresponding weight. The sum of all products is divided by 10; the remainder is the check digit."
            },
            {
                "title": "2. Character Value Mapping Table",
                "text": "The character set consists of digits 0-9, capital letters A-Z, and the filler character '<'. Digits 0 through 9 have values 0 through 9. Letters A through Z have values 10 through 35 respectively (A=10, B=11 ... Z=35). The filler character '<' is assigned a numeric value of 0."
            },
            {
                "title": "3. Composite Check Digit Calculation",
                "text": "The composite check digit in TD3 line 2 (position 44) verifies the combined integrity of the passport number + its check digit, the date of birth + its check digit, and the date of expiry + its check digit, plus any optional data and its check digit. A failure in the composite check indicates OCR misread or physical alteration."
            }
        ]
    },
    {
        "filename": "icao_transliteration_standards.pdf",
        "format": "pdf",
        "title": "ICAO Doc 9303 Section IV: Official Transliteration Standards for Non-Latin Scripts",
        "doc_type": "standard",
        "version_date": "2022-01-18",
        "sections": [
            {
                "title": "1. Mandatory Latin Representation in MRZ",
                "text": "Only basic Latin letters A-Z (without accents, diacritics, or cedillas) are permitted within the Machine Readable Zone. National characters must be transliterated strictly in accordance with the official ICAO conversion tables. Diacritical marks in the VIZ are stripped or converted (e.g. umlaut Ä becomes AE in German MRZ, Ukrainian Щ becomes SHCH)."
            },
            {
                "title": "2. Ukrainian Cyrillic to ICAO Latin Mapping",
                "text": "Ukrainian official travel documents use the National Transliteration Standard (Resolution No. 55): А->A, Б->B, В->V, Г->H, Ґ->G, Д->D, Е->E, Є->YE (at start of word) or E, Ж->ZH, З->Z, И->Y, І->I, Ї->YI (start) or I, Й->Y (start) or I, К->K, Л->L, М->M, Н->N, О->O, П->P, Р->R, С->S, Т->T, У->U, Ф->F, Х->KH, Ц->TS, Ч->CH, Ш->SH, Щ->SHCH, Ю->YU (start) or IU, Я->YA (start) or IA, Ь and apostrophe are omitted."
            }
        ]
    },
    {
        "filename": "icao_biometric_rfid_spec.pdf",
        "format": "pdf",
        "title": "ICAO Doc 9303 Part 10: Logical Data Structure (LDS) and Contactless IC Chip Security",
        "doc_type": "technical_guide",
        "version_date": "2021-11-20",
        "sections": [
            {
                "title": "1. Logical Data Structure (LDS1) Data Groups",
                "text": "The contactless integrated circuit embedded in e-passports and electronic ID cards contains standardized Data Groups (DG). DG1 stores the MRZ info. DG2 contains the primary facial image template (JPEG or JPEG2000 format, minimum 480x640 pixels). DG3 holds fingerprint templates (restricted access via EAC). DG14 contains Chip Authentication public keys. DG15 contains Active Authentication public keys."
            },
            {
                "title": "2. Access Control Mechanisms: BAC and PACE",
                "text": "Basic Access Control (BAC) derives encryption keys from the MRZ line (document number, date of birth, expiry date). Password Authenticated Connection Establishment (PACE / SAC) utilizes Diffie-Hellman key agreement based on the Card Access Number (CAN) or MRZ, providing robust resistance against eavesdropping."
            }
        ]
    },
    {
        "filename": "security_printing_techniques.pdf",
        "format": "pdf",
        "title": "DocAI Technical Manual: Security Printing and Physical Anti-Forgery Features",
        "doc_type": "technical_guide",
        "version_date": "2022-04-12",
        "sections": [
            {
                "title": "1. Guilloche Patterns and Iris Printing",
                "text": "Guilloche consists of intricate geometric patterns of fine interlaced continuous lines generated by high-precision mathematical algorithms. Rainbow (iris) printing involves a subtle, continuous colour blend across the substrate without discrete boundary dots. Counterfeit colour copiers inevitably introduce raster moiré dots."
            },
            {
                "title": "2. Microprinting and Tactile Laser Engraving",
                "text": "Positive and negative microprinting features characters between 0.15 mm and 0.25 mm in height, illegible without magnification (10x). Polycarbonate laser engraving produces localized carbonization inside internal card layers, creating a perceptible tactile relief on the surface that cannot be scraped off without destroying the card."
            }
        ]
    },
    {
        "filename": "optical_variable_devices_guideline.pdf",
        "format": "pdf",
        "title": "Verification Guide: Optically Variable Devices (OVD) and Holographic Overlays",
        "doc_type": "technical_guide",
        "version_date": "2022-09-05",
        "sections": [
            {
                "title": "1. DOVID and Kinegram Classifications",
                "text": "Diffractive Optically Variable Image Devices (DOVIDs), including Kinegrams and holographic patches, display dynamic kinematic effects, color reversals, and 3D visual depths under varying angles of incident light. In automated KYC video capture, dynamic liveness checks track the angular movement of high-intensity specular reflection."
            },
            {
                "title": "2. Optically Variable Ink (OVI) Characteristics",
                "text": "True OVI pigments contain microscopic multi-layer interference flakes that shift color depending on viewing angle (typically magenta to green, or gold to blue). Scanning or color photocopying produces only a single static flat tone without spectral phase shift."
            }
        ]
    },
    {
        "filename": "mrz_ocr_error_correction_patterns.pdf",
        "format": "pdf",
        "title": "DocAI OCR Engineering: Machine Learning Error Correction Patterns for MRZ",
        "doc_type": "technical_guide",
        "version_date": "2023-02-14",
        "sections": [
            {
                "title": "1. Common Optical Confusion Pairs",
                "text": "Under low-light or glare conditions, OCR engines frequently confuse optical glyphs: digit '0' and letter 'O', digit '1' and letter 'I' or 'L', digit '5' and letter 'S', digit '8' and letter 'B', digit '2' and letter 'Z'. In MRZ processing, positions are strictly typed (alpha-only vs numeric-only), enabling 100% deterministic pre-correction."
            },
            {
                "title": "2. Chevrons and Delimiter Recovery",
                "text": "Filler chevrons '<' are frequently misrecognized as '(', '[', '{', or 'K' when printed with low ink density. A specialized regex pass converts non-alphanumeric punctuation inside MRZ boundaries back to '<'. Any line with length differing from 30 (TD1) or 44 (TD3) must trigger boundary recalibration."
            }
        ]
    },
    {
        "filename": "facial_image_icao_spec.pdf",
        "format": "pdf",
        "title": "ICAO Portrait Standards: Specifications for Digital Facial Images in Travel Documents",
        "doc_type": "standard",
        "version_date": "2020-10-30",
        "sections": [
            {
                "title": "1. Pose, Illumination and Background",
                "text": "The subject portrait must be taken facing forward with a neutral facial expression, mouth closed, and eyes open and clearly visible. Background must be plain, uniform, and light-coloured (white or light grey). Shadows across face or background are unacceptable. Eyeglasses must not have tinted lenses or heavy frames obscuring pupils."
            },
            {
                "title": "2. Head Size and Geometry Ratios",
                "text": "The vertical distance from chin to crown must occupy between 70% and 80% of the total image height. The interpupillary distance must be at least 90 pixels for automated biometric matching. Compression artifacts (JPEG blocking) must not exceed standard Q=80 threshold."
            }
        ]
    },
    {
        "filename": "document_substrate_analysis.pdf",
        "format": "pdf",
        "title": "DocAI Hardware & Material Guide: Substrate Analysis and Luminescence",
        "doc_type": "technical_guide",
        "version_date": "2021-05-19",
        "sections": [
            {
                "title": "1. Optical Brightener Free Security Paper",
                "text": "Genuine security paper for passports contains no optical brightening agents (OBAs). Under ultraviolet illumination (365 nm UV-A), authentic paper remains dull dark, whereas commercial copy paper fluoresces bright blue-white due to optical whitening chemicals."
            },
            {
                "title": "2. Polycarbonate Thermal Fusion",
                "text": "Polycarbonate identity cards consist of multiple thin foil layers laminated under heat and pressure into a solid monolithic block. Once fused, individual layers cannot be separated without irreversible destruction, rendering mechanical knife-splitting counterfeiting impossible."
            }
        ]
    },

    # --- 2. РЕГУЛЯТОРНІ НОРМИ ТА KYC ПОЛІТИКИ (MARKDOWN) ---
    {
        "filename": "nbu_resolution_65_remote_kyc.md",
        "format": "md",
        "title": "Положення НБУ № 65: Дистанційна ідентифікація та верифікація клієнтів банків",
        "doc_type": "regulation",
        "version_date": "2020-05-19",
        "sections": [
            {
                "title": "Розділ II. Загальні вимоги до дистанційної ідентифікації",
                "text": "Суб'єкт первинного фінансового моніторингу (СПФМ) має право здійснювати дистанційну ідентифікацію та верифікацію клієнтів — фізичних осіб через систему BankID Національного банку України, з використанням електронного підпису Дія.Підпис, або шляхом здійснення сесії відеоверифікації в режимі реального часу."
            },
            {
                "title": "Розділ III. Правила проведення відеоверифікації",
                "text": "Відеоверифікація проводиться уповноваженим працівником банку або автоматизованим модулем із обов'язковим динамічним контролем присутності (liveness detection). Під час сеансу здійснюється: 1) фіксація геолокації та IP-адреси; 2) зчитування та візуальна перевірка захисних елементів паспорта (нахили для перевірки OVI/голограм); 3) порівняння обличчя клієнта з фотографією в документі із порогом схожості не менше 0.95."
            },
            {
                "title": "Розділ IV. Валідність та перевірка електронних копій",
                "text": "Електронна копія документа (е-паспорт у застосунку Дія) набуває юридичної сили лише після успішної перевірки її дійсності через кваліфіковану електронну печатку ДП 'Дія'. Скриншоти екрана смартфона суворо заборонено приймати як належне підтвердження особи."
            }
        ]
    },
    {
        "filename": "nbu_res65_risk_assessment_cdd.md",
        "format": "md",
        "title": "Положення НБУ № 65: Критерії ризику та належна обачність щодо клієнтів (CDD)",
        "doc_type": "regulation",
        "version_date": "2021-03-22",
        "sections": [
            {
                "title": "Стаття 14. Класифікація рівнів ризику клієнта",
                "text": "СПФМ зобов'язаний присвоювати клієнту категорію ризику: низький, середній або високий. Високий рівень ризику автоматично встановлюється для: 1) публічних діячів (PEP) та членів їх сімей; 2) осіб, чиє місце проживання або реєстрації знаходиться в державах, що не виконують рекомендації FATF; 3) клієнтів, які здійснюють нетипові складні фінансові операції на суму понад 400 000 грн."
            },
            {
                "title": "Стаття 15. Посилена належна перевірка (Enhanced Due Diligence)",
                "text": "У разі встановлення високого рівня ризику застосовуються заходи EDD: отримання дозволу керівника фінансової установи на встановлення ділових відносин; встановлення джерел походження статків (коштів) на підставі офіційних податкових декларацій; проведення постійного поглибленого моніторингу всіх фінансових транзакцій."
            }
        ]
    },
    {
        "filename": "fatf_recommendation_10_cdd.md",
        "format": "md",
        "title": "FATF Recommendation 10: Customer Due Diligence and Beneficial Ownership Standards",
        "doc_type": "standard",
        "version_date": "2023-06-10",
        "sections": [
            {
                "title": "1. Core Principles of Customer Due Diligence",
                "text": "Financial institutions must undertake CDD when: establishing business relations; carrying out occasional transactions above the designated threshold (USD/EUR 15,000); there is a suspicion of money laundering or terrorist financing; or the institution has doubts about the veracity or adequacy of previously obtained customer identification data."
            },
            {
                "title": "2. Identification and Verification of Beneficial Owners",
                "text": "Institutions must identify the ultimate beneficial owner (UBO) holding a direct or indirect ownership interest of 25% or more in a legal entity. Verification requires reliable, independently sourced documents, data or information. Reliance on self-declarations without supporting registry extracts is strictly prohibited."
            }
        ]
    },
    {
        "filename": "fatf_recommendation_12_pep.md",
        "format": "md",
        "title": "FATF Recommendation 12: Politically Exposed Persons (PEPs) Lifecycle and Monitoring",
        "doc_type": "standard",
        "version_date": "2022-11-01",
        "sections": [
            {
                "title": "1. Definition of Foreign and Domestic PEPs",
                "text": "Foreign PEPs are individuals who are or have been entrusted with prominent public functions by a foreign country (e.g. Heads of State, senior politicians, senior government or judicial officials, military officers). Domestic PEPs are individuals entrusted with prominent public functions domestically. Family members and close associates are subject to the same diligence tier."
            },
            {
                "title": "2. Risk Mitigation Measures for PEP Relationships",
                "text": "For all foreign PEPs, financial institutions must implement proactive risk management systems to determine if a customer or UBO is a PEP, obtain senior management sign-off before account opening, establish the source of wealth and funds, and conduct enhanced ongoing transaction scrutiny."
            }
        ]
    },
    {
        "filename": "fintech_acceptable_documents_policy.md",
        "format": "md",
        "title": "FinTech KYC Operational Policy: Hierarchy of Acceptable Identification Documents",
        "doc_type": "policy",
        "version_date": "2023-09-01",
        "sections": [
            {
                "title": "1. Primary Identification Documents (Tier 1)",
                "text": "Only government-issued identity documents featuring a high-resolution photograph and unambiguous legal identity are accepted as primary proof of identity: 1) Ukrainian biometric Passport for traveling abroad; 2) Ukrainian ID-card (plastic); 3) Ukrainian internal passport booklet (1994 series) with required 25/45 age photographs; 4) Foreign international biometric passport; 5) Permanent residence permit (Посвідка на постійне проживання)."
            },
            {
                "title": "2. Prohibited and Deprecated Documents",
                "text": "The following documents are NOT accepted as primary proof of identity: student cards, employee badges, military service booklets (for civilian commercial onboarding), driving licenses (except where specifically permitted as secondary proof), temporary residence certificates without biometric chips, and expired documents past their official grace periods."
            }
        ]
    },
    {
        "filename": "fintech_secondary_proof_address.md",
        "format": "md",
        "title": "FinTech KYC Policy: Proof of Address (POA) and Secondary Document Standards",
        "doc_type": "policy",
        "version_date": "2023-09-15",
        "sections": [
            {
                "title": "1. Acceptable Proof of Address Documents",
                "text": "Proof of Address (POA) is required when registering accounts with transaction limits exceeding EUR 1,000/month. Acceptable documents include: 1) Utility bill (electricity, water, natural gas, fixed broadband internet); 2) Bank account statement displaying full residential address; 3) Official tax assessment or local council registration certificate (Витяг про місце проживання). Mobile telephone bills are categorically rejected."
            },
            {
                "title": "2. Validity Period and Format Restrictions",
                "text": "All POA documents must be dated within 90 calendar days prior to the submission date. Documents must clearly display: customer's full legal name, full residential address, date of issue, and official company logo/stamp. Digital PDF statements downloaded directly from bank portals are permitted if cryptographically signed."
            }
        ]
    },
    {
        "filename": "rnocpp_tax_id_specifications.md",
        "format": "md",
        "title": "Ukrainian Tax ID (RNOCPP): Algorithmic Structure and Checksum Verification",
        "doc_type": "technical_guide",
        "version_date": "2021-08-25",
        "sections": [
            {
                "title": "1. Ten-Digit RNOCPP Structural Composition",
                "text": "The Ukrainian Registration Number of the Taxpayer's Record Card (РНОКПП) comprises exactly 10 decimal digits. The first 5 digits represent the sequential number of days elapsed from December 31, 1899 to the individual's date of birth. Digits 6-9 represent the sequential record number in the register for persons born on that day (digit 9 designates gender: odd for male, even for female). Digit 10 is the algorithmic check digit."
            },
            {
                "title": "2. Checksum Mathematical Formula",
                "text": "The 10th check digit is computed using constant weight coefficients [-1, 5, 7, 9, 4, 6, 10, 5, 7]. Sum S = sum(d[i] * w[i]) for i=0..8. Remainder R = S % 11. If R < 10, check digit = R. If R == 10, a secondary calculation with weights [3, 5, 7, 9, 4, 6, 10, 5, 7] is evaluated: S2 = sum(d[i] * w2[i]), R2 = S2 % 11, check digit = R2 % 10."
            }
        ]
    },
    {
        "filename": "ukrainian_passport_1994_booklet_rules.md",
        "format": "md",
        "title": "Document Verification Standard: Ukrainian Internal Passport (1994 Model Booklet)",
        "doc_type": "standard",
        "version_date": "2022-03-10",
        "sections": [
            {
                "title": "1. Structure and Photo Insertion Milestones",
                "text": "The 1994 booklet passport consists of 16 stitched paper pages with blue security fibers. A new photograph of the holder must be glued into the passport upon attaining 25 and 45 years of age (pages 3 and 5 respectively). The holder has a statutory window of 30 calendar days to insert the photo; during martial law, this requirement is extended until 30 days after cessation of martial law."
            },
            {
                "title": "2. Stamp and Watermark Validation",
                "text": "The dry relief seal of the State Migration Service (DMS) must visibly press into the corner of the photograph and the substrate paper. The paper must exhibit a continuous watermark depicting the National Emblem of Ukraine (Trident) when held against a transmitted light source."
            }
        ]
    },
    {
        "filename": "ukrainian_id_card_2015_technical.md",
        "format": "md",
        "title": "Technical Specification: Ukrainian Citizen ID Card (Passport Format 2015+)",
        "doc_type": "standard",
        "version_date": "2021-12-05",
        "sections": [
            {
                "title": "1. Multi-Layer Polycarbonate Construction",
                "text": "The Ukrainian ID card is a single-piece multi-layer polycarbonate card compliant with ISO/IEC 7810 format ID-1. All text elements on the surface and embedded laser-engraved images are produced via Nd:YAG laser marking inside the core polycarbonate layer without ink ribbons."
            },
            {
                "title": "2. Validity Lifespan by Age Group",
                "text": "For persons aged 14 to 18 years, the Ukrainian ID card is issued with a validity period of 4 years. For persons who have attained 18 years of age, the ID card is issued with a validity period of 10 years. Expired ID cards are legally invalid for opening bank accounts or undergoing KYC verification."
            }
        ]
    },
    {
        "filename": "driver_license_kyc_limitations.md",
        "format": "md",
        "title": "Compliance Guideline: Ukrainian Driving License Limitations in FinTech Verification",
        "doc_type": "policy",
        "version_date": "2023-04-18",
        "sections": [
            {
                "title": "1. Non-Citizenship Status of Driving Licenses",
                "text": "Under Article 5 of the Law of Ukraine on the Unified State Demographic Register, a driving license certifies special technical qualification to operate vehicles and confirms identity within the borders of Ukraine, but does NOT certify Ukrainian citizenship or tax residency status."
            },
            {
                "title": "2. Restricted Applicability in Banking Onboarding",
                "text": "Financial institutions may accept a driving license strictly as secondary identification document or for simplified tier-1 identification with monthly turnover ceiling not exceeding 5,000 UAH. Full KYC onboarding requires a primary citizenship travel document (ID card or Passport)."
            }
        ]
    },
    {
        "filename": "sanctions_pep_screening_standards.md",
        "format": "md",
        "title": "AML/CFT Compliance Manual: Automated Sanctions and PEP Screening Procedures",
        "doc_type": "policy",
        "version_date": "2023-10-12",
        "sections": [
            {
                "title": "1. Mandatory Sanction Lists Repositories",
                "text": "FinTech automated onboarding pipelines must screen every applicant against: 1) Ukrainian National Security and Defense Council (РНБО) sanction lists; 2) US Office of Foreign Assets Control (OFAC) SDN list; 3) European Union Consolidated Financial Sanctions list; 4) United Nations Security Council Sanctions Committee database."
            },
            {
                "title": "2. Fuzzy Matching Thresholds and False Positive Rules",
                "text": "Screening engines must utilize phonetically-aware fuzzy string matching (Jaro-Winkler, Levenshtein distance). Matches with a similarity score >= 0.85 (85%) must automatically freeze automated onboarding and route the case to Tier-2 Compliance Officer for manual investigation."
            }
        ]
    },
    {
        "filename": "liveness_detection_iso30107_standards.md",
        "format": "md",
        "title": "Biometric Security Guide: ISO/IEC 30107 Presentation Attack Detection (PAD)",
        "doc_type": "standard",
        "version_date": "2022-07-28",
        "sections": [
            {
                "title": "1. Presentation Attack Instruments (PAI) Taxonomy",
                "text": "Presentation attack detection categorizes spoofing attempts into: Level 1 (print attacks on 2D paper, digital photo replay on smartphone displays); Level 2 (curved 3D paper masks, high-resolution video replay on 4K tablets); Level 3 (custom silicone masks, latex prosthetics, real-time AI deepfake video injection)."
            },
            {
                "title": "2. Passive vs Active Liveness Mechanisms",
                "text": "Passive liveness analyzes skin texture, micro-reflections, pulse photoplethysmography (rPPG), and depth from a single continuous stream without requiring user commands. Active liveness prompts the user to perform randomized actions (smile, turn head 15 degrees left, blink twice within 3 seconds) to thwart static replays."
            }
        ]
    },
    {
        "filename": "gdpr_kyc_data_retention_rules.md",
        "format": "md",
        "title": "Legal Compliance: KYC Data Retention and Storage Limits under AML Law and GDPR",
        "doc_type": "regulation",
        "version_date": "2023-01-10",
        "sections": [
            {
                "title": "1. Statutory Five-Year Retention Requirement",
                "text": "Under Article 8 of the Law of Ukraine on AML/CFT, all customer identification files, video verification session recordings, document scans, and transactional records must be stored securely for a minimum of 5 years following the formal termination of the business relationship."
            },
            {
                "title": "2. Right to Erasure (GDPR Art. 17) Exceptions",
                "text": "The GDPR Right to Erasure ('Right to be Forgotten') is statutorily overridden by AML mandatory compliance obligations. A customer cannot demand the immediate deletion of their KYC verification files prior to the expiration of the mandatory 5-year anti-money laundering statutory window."
            }
        ]
    },
    {
        "filename": "legacy_kyc_rules_v1_deprecated.md",
        "format": "md",
        "title": "ARCHIVED: FinTech Operational KYC Rules v1.0 (2019 Edition - DEPRECATED)",
        "doc_type": "policy",
        "version_date": "2019-04-01",
        "sections": [
            {
                "title": "1. Archived Rule: Unverified Email Scans (SUPERSEDED)",
                "text": "WARNING: THIS POLICY IS OBSOLETE AND NO LONGER IN FORCE. In 2019, customers were permitted to submit non-certified photocopies or plain black-and-white JPEG images via email without dynamic liveness detection. This rule was completely revoked by NBU Resolution No. 65 in 2020."
            },
            {
                "title": "2. Archived Rule: Monthly Turnover Ceilings (OBSOLETE)",
                "text": "The former simplified limit of 2,000 UAH per month without secondary document verification has been replaced by the tiered risk framework established in 2021. Any system referencing this 2019 document must issue a deprecation warning."
            }
        ]
    },
    {
        "filename": "outdated_notarized_copies_memo.md",
        "format": "md",
        "title": "ARCHIVED: Guidance Memo on Paper-Based Notarized Copies (DEPRECATED 2018)",
        "doc_type": "policy",
        "version_date": "2018-06-15",
        "sections": [
            {
                "title": "1. Historic Notarized Paper Submission",
                "text": "HISTORIC REFERENCE ONLY: Under the 2018 guidance, individuals residing outside regional administrative centers were required to dispatch physical paper copies certified by a local notary through postal mail. This procedure has been totally superseded by remote digital identification via Diia and BankID."
            }
        ]
    },

    # --- 3. ТЕХНІЧНІ КЕРІВНИЦТВА ТА DOCAI СТАНДАРТИ (HTML) ---
    {
        "filename": "docai_image_resolution_dpi_standards.html",
        "format": "html",
        "title": "DocAI Computer Vision Standard: Image Resolution and Pre-processing Benchmarks",
        "doc_type": "technical_guide",
        "version_date": "2023-05-14",
        "sections": [
            {
                "title": "1. Minimum DPI Thresholds for OCR Accuracy",
                "text": "Optical Character Recognition of identity documents requires an effective resolution of at least 300 DPI across the entire bounding box of the document. For reading microprinted security text (<= 0.25 mm) or fine guilloche lines, 600 DPI is required. Images below 150 DPI suffer an exponential character error rate (CER > 18%)."
            },
            {
                "title": "2. Dynamic Range and Color Depth",
                "text": "Captured frames must feature 24-bit RGB color depth. Grayscale or aggressive binarization must be avoided during initial ingestion to preserve watermark gradients and fluorescent fiber detection under color filtering."
            }
        ]
    },
    {
        "filename": "docai_blur_glare_shadow_detection.html",
        "format": "html",
        "title": "DocAI Quality Assessment: Algorithmic Blur, Glare, and Shadow Filtering",
        "doc_type": "technical_guide",
        "version_date": "2023-07-22",
        "sections": [
            {
                "title": "1. Blur Detection using Laplacian Variance",
                "text": "Document sharpness is evaluated by convolving the grayscale image with a standard 3x3 Laplacian kernel and calculating the variance of the response map. Images exhibiting a Laplacian variance score below 100.0 are classified as blurred and immediately rejected with an in-app prompt to stabilize the mobile camera."
            },
            {
                "title": "2. Specular Glare Region Masking",
                "text": "Specular glare occurs when camera flash or ambient lighting reflects directly off glossy plastic laminate. Pixels with luminance Y > 250 in the YUV space across a contiguous area exceeding 1.5% of the MRZ or VIZ boundary mask trigger an automatic glare rejection."
            }
        ]
    },
    {
        "filename": "docai_document_boundary_corner_cropping.html",
        "format": "html",
        "title": "DocAI Computer Vision: Perspective Correction and Corner Quad Detection",
        "doc_type": "technical_guide",
        "version_date": "2022-10-18",
        "sections": [
            {
                "title": "1. Quad Corner Detection and Aspect Ratio Bounds",
                "text": "Document boundary localization uses deep segmentation networks to predict four corner coordinates (top-left, top-right, bottom-right, bottom-left). All four corners must lie within the camera viewport. The aspect ratio of the rectified quad must conform to ID-1 (1.586 +/- 0.05) or ID-3 (1.420 +/- 0.05)."
            },
            {
                "title": "2. Perspective Warp Transformation",
                "text": "A 3x3 homography matrix transforms the perspective-distorted quadrilateral into an orthogonal rectangular canvas of standard pixel dimensions (1012x638 pixels for ID-1). Truncated corners exceeding 15 pixels along any edge result in a fatal cropping defect code (ERR_DOC_CROPPED)."
            }
        ]
    },
    {
        "filename": "docai_mrz_layout_and_crop_extraction.html",
        "format": "html",
        "title": "DocAI Pipeline Specification: Automated MRZ Bounding Box Segmentation",
        "doc_type": "technical_guide",
        "version_date": "2023-03-30",
        "sections": [
            {
                "title": "1. Morphological Black-Hat Filtering",
                "text": "To localize the MRZ band on TD1 and TD3 documents, a directional morphological Black-Hat filter with a horizontal structuring element (width 25, height 3) is applied. This highlights dense, high-frequency dark character blocks against lighter substrate background."
            },
            {
                "title": "2. Horizontal Projection Profiling",
                "text": "Row-wise intensity projection profiling identifies individual text lines. The vertical distance between adjacent baseline peaks must match the standard OCR-B line spacing (tolerance 5%). Any rotational skew exceeding 3.0 degrees must be corrected via Radon transform prior to character segmentation."
            }
        ]
    },
    {
        "filename": "docai_ghost_portrait_cross_check.html",
        "format": "html",
        "title": "Fraud Detection Guide: Ghost Portrait and Secondary Image Cross-Verification",
        "doc_type": "technical_guide",
        "version_date": "2023-08-11",
        "sections": [
            {
                "title": "1. Secondary Ghost Portrait Characteristics",
                "text": "Polycarbonate ID cards and passports incorporate a secondary, smaller 'ghost' portrait laser-engraved into a clear window or printed with UV-luminescent ink. The facial geometry and feature vectors of the primary portrait and the secondary ghost portrait must yield a cosine similarity >= 0.88."
            },
            {
                "title": "2. Tamper Indicators in Portrait Swapping",
                "text": "Fraudulent document alteration commonly involves cutting out and replacing only the primary photo while leaving the ghost portrait intact. A cosine similarity divergence below 0.70 between primary and ghost portraits constitutes definitive evidence of physical document tampering."
            }
        ]
    },
    {
        "filename": "docai_font_typography_tampering_detection.html",
        "format": "html",
        "title": "Fraud Detection Guide: Font Typographical Anomaly and Kerning Verification",
        "doc_type": "technical_guide",
        "version_date": "2023-01-24",
        "sections": [
            {
                "title": "1. Strict Monospace Geometry in MRZ",
                "text": "Every character in an authentic ICAO MRZ conforms strictly to the OCR-B typeface standard with a fixed pitch of 2.54 mm (0.1 inch). Inter-character gaps and baseline alignment must remain perfectly uniform across the entire 30 or 44 character line."
            },
            {
                "title": "2. Digit Splicing and Thickness Anomalies",
                "text": "Digital alteration (e.g. changing birth year from 1999 to 1989) inevitably introduces subtle stroke-width discrepancies, antialiasing halos, or baseline vertical misalignment exceeding 2 pixels. Deep Siamese networks compare candidate glyphs against an authentic template database."
            }
        ]
    },
    {
        "filename": "diia_signature_crypto_verification.html",
        "format": "html",
        "title": "Integration Specification: Diia.Signature Cryptographic Verification Architecture",
        "doc_type": "technical_guide",
        "version_date": "2022-11-19",
        "sections": [
            {
                "title": "1. Qualified Electronic Signature (QES) Standard",
                "text": "Дія.Підпис is a Qualified Electronic Signature compliant with DSTU 4145-2002 and eIDAS standards. The private key is generated within the Secure Enclave / TEE of the user's smartphone after biometric facial verification against the Unified State Demographic Register."
            },
            {
                "title": "2. Online OCSP and CRL Validation",
                "text": "Upon receiving a signed payload, the FinTech backend must immediately query the Qualified Trust Service Provider (QTSP) via Online Certificate Status Protocol (OCSP) to verify that the signer's certificate was valid and not revoked at the exact moment of signing."
            }
        ]
    },
    {
        "filename": "bankid_nbu_integration_protocol.html",
        "format": "html",
        "title": "Integration Guide: BankID NBU OAuth2 Protocol and Scopes for FinTech",
        "doc_type": "technical_guide",
        "version_date": "2023-06-25",
        "sections": [
            {
                "title": "1. Authorization Flow and Scopes",
                "text": "BankID NBU operates over OAuth 2.0 / OpenID Connect. The client application initiates an authorization request with scopes 'customer_identity', 'passport_data', and 'tax_number'. The identifying bank cryptographically signs the returned customer payload with an X.509 certificate."
            },
            {
                "title": "2. Data Freshness and Re-verification Requirements",
                "text": "Identification data received via BankID NBU carries an issuance timestamp. Under NBU rules, customer data obtained via BankID is considered freshly verified for a maximum of 12 months, after which updated identity data must be requested."
            }
        ]
    },
    {
        "filename": "anti_money_laundering_red_flags_kyc.html",
        "format": "html",
        "title": "Compliance Manual: KYC Onboarding Red Flags and Suspicious Activity Indicators",
        "doc_type": "policy",
        "version_date": "2023-04-05",
        "sections": [
            {
                "title": "1. Typical Onboarding Anomaly Red Flags",
                "text": "Red flags that mandate manual escalation include: 1) Multiple onboarding attempts from the same device hardware ID using different identity names; 2) Virtual private network (VPN) or Tor exit nodes cloaking residential IP address; 3) IP address geographical location differing by more than 1,000 km from the declared residential city."
            },
            {
                "title": "2. Synthetic Identity Patterns",
                "text": "Synthetic identity fraud combines genuine RNOCPP tax identifiers with fabricated names or stolen passport numbers. Automated cross-referencing between the demographic registry and tax authority databases is required to detect inconsistencies before granting credit or payment limits."
            }
        ]
    },
    {
        "filename": "biometric_facial_recognition_1to1_thresholds.html",
        "format": "html",
        "title": "Biometric Engineering Standard: 1:1 Facial Verification Performance Metrics",
        "doc_type": "technical_guide",
        "version_date": "2022-08-30",
        "sections": [
            {
                "title": "1. False Match Rate (FMR) vs False Non-Match Rate (FNMR)",
                "text": "For banking KYC compliance, the 1:1 facial matching algorithm must achieve a False Match Rate (FMR) <= 0.001% (1 in 100,000) at an operational False Non-Match Rate (FNMR) <= 1.0%. The similarity threshold corresponding to this operating point must be calibrated on demographic datasets representing the target population."
            },
            {
                "title": "2. Environmental Normalization",
                "text": "Prior to vector embedding extraction (512-dimensional ArcFace/CosFace embeddings), the input face is aligned via 5-point affine transformation based on eye pupil and mouth corner coordinates, followed by histogram equalization to mitigate harsh lighting variations."
            }
        ]
    }
]


def create_pdf(filepath: Path, doc_data: dict):
    """Створює валідний PDF файл зі структурою заголовків, тексту та таблиці."""
    doc = SimpleDocTemplate(str(filepath), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1a365d'),
        spaceAfter=10
    )
    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#4a5568'),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=8,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=8
    )

    story = []
    story.append(Paragraph(html.escape(doc_data["title"]), title_style))
    story.append(Paragraph(f"<b>Type:</b> {html.escape(doc_data['doc_type'])} | <b>Date:</b> {html.escape(doc_data['version_date'])}", meta_style))
    story.append(Spacer(1, 4))

    for sec in doc_data["sections"]:
        story.append(Paragraph(html.escape(sec["title"]), h2_style))
        story.append(Paragraph(html.escape(sec["text"]), body_style))
        story.append(Spacer(1, 4))

    doc.build(story)


def create_markdown(filepath: Path, doc_data: dict):
    """Створює структурований Markdown документ."""
    lines = [
        f"# {doc_data['title']}",
        "",
        f"> **Document Type:** {doc_data['doc_type']}  ",
        f"> **Effective Date / Version:** {doc_data['version_date']}  ",
        "",
        "---",
        ""
    ]
    for sec in doc_data["sections"]:
        lines.append(f"## {sec['title']}")
        lines.append("")
        lines.append(sec["text"])
        lines.append("")
    filepath.write_text("\n".join(lines), encoding="utf-8")


def create_html(filepath: Path, doc_data: dict):
    """Створює валідний семантичний HTML5 документ."""
    sections_html = []
    for sec in doc_data["sections"]:
        sections_html.append(f"""
        <section>
            <h2>{sec['title']}</h2>
            <p>{sec['text']}</p>
        </section>""")
    
    html_content = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <title>{doc_data['title']}</title>
    <meta name="doc-type" content="{doc_data['doc_type']}">
    <meta name="version-date" content="{doc_data['version_date']}">
</head>
<body>
    <header>
        <h1>{doc_data['title']}</h1>
        <div class="metadata">
            <span class="type">Type: {doc_data['doc_type']}</span>
            <span class="date">Date: {doc_data['version_date']}</span>
        </div>
    </header>
    <main>
        {"".join(sections_html)}
    </main>
</body>
</html>"""
    filepath.write_text(html_content, encoding="utf-8")


def build_corpus() -> int:
    """Генерує повний корпус документів у data/corpus/."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for doc in DOCUMENTS_DATA:
        filepath = CORPUS_DIR / doc["filename"]
        fmt = doc["format"]
        if fmt == "pdf":
            create_pdf(filepath, doc)
        elif fmt == "md":
            create_markdown(filepath, doc)
        elif fmt == "html":
            create_html(filepath, doc)
        count += 1

    # Створюємо 1 спеціальний дублікат для демонстрації та тестування дедуплікації:
    duplicate_path = CORPUS_DIR / "duplicate_fatf_rec10.md"
    orig_path = CORPUS_DIR / "fatf_recommendation_10_cdd.md"
    duplicate_path.write_text(orig_path.read_text(encoding="utf-8"), encoding="utf-8")
    count += 1

    print(f"✅ Успішно згенеровано {count} документів (включно з 1 тестовим дублікатом) у {CORPUS_DIR}")
    return count


if __name__ == "__main__":
    build_corpus()
