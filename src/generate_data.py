"""
Скрипт генерації 30 реалістичних брудних OCR-текстів документів та 10 еталонних анотацій (Golden Set).
Усі дані повністю синтетичні та не містять персональних даних (No PII).
"""

import json
from pathlib import Path


RAW_DOCUMENTS = [
    {
        "id": "doc_01",
        "description": "ID-картка: відблиск на прізвищі, розрив стрічки MRZ, OCR сплутав O та 0",
        "raw_text": """
УКРАЇНА / UKRAINE
ПАСПОРТ ГРОМАДЯНИНА УКРАЇНИ / IDENTITY CARD
Прізвище / Surname:
К О В А Л Е Н K O  /  K O V A L E N K 0
Ім'я / Given names:
О Л Е К С І Й  /  O L E K S I I
По батькові: ВАСИЛЬОВИЧ
Стать / Sex: Ч / M   Громадянство / Nationality: УКРАЇНА / UKR
Дата народження / Date of birth: 14.05.1992
Дійсний до / Expiry date: 14.05.2032
Документ № / Document No: 002948172
Запис № / Record No: 19920514-01824
Орган, що видав / Authority: 8011
*** УВАГА: ВІДБЛИСК СПАЛАХУ НА ГОЛОГРАМІ ВНИЗУ ***
MRZ:
I<UKR0029481724<<<<<<<<<<<<<<<
9205142M3205146UKR19920514-01824<8
KOVALENKO<<OLEKSII<<<<<<<<<<<<
"""
    },
    {
        "id": "doc_02",
        "description": "Закордонний паспорт: формат дати з місяцем словом, шум у скані",
        "raw_text": """
UKRAINE / УКРАЇНА
PASSPORT / ПАСПОРТ
Type/Тип: P  Country Code/Код держави: UKR  Passport No/Номер паспорта: FN384910
Surname/Прізвище:
М Е Л Ь Н И К  /  M E L N Y K
Given names/Імена:
О К С А Н А  /  O K S A N A
Nationality/Громадянство: УКРАЇНА / UKRAINE
Date of birth/Дата народження: 22 ЛИП / JUL 1988
Personal No/Запис №: 19880722-03911
Sex/Стать: Ж / F   Place of birth: М. КИЇВ / KYIV
Date of issue/Дата видачі: 10 ВЕР / SEP 2020
Date of expiry/Дата закінчення дії: 10 ВЕР / SEP 2030
Authority/Орган, що видав: 8001
P<UKRMELNYK<<OKSANA<<<<<<<<<<<<<<<<<<<<<<<<<
FN384910<4UKR8807223F300910119880722-03911<42
"""
    },
    {
        "id": "doc_03",
        "description": "Посвідчення водія: три мови, категорії B/C, відсутність MRZ",
        "raw_text": """
ПОСВІДЧЕННЯ ВОДІЯ / DRIVING LICENCE / PERMIS DE CONDUIRE
УКРАЇНА / UKRAINE
1. БОНДАРЕНКО / BONDARENKO
2. Д М И Т Р О  І Г О Р О В И Ч / DMYTRO
3. 03.11.1985 М. ХАРКІВ
4a. 15.02.2018  4b. 15.02.2048  4c. ТСЦ 6341
5. BXA 839201
9. B, C
[PHOTO ATTACHED]  [SIGNATURE]
*** Низ картки злегка затертий, магнітна смуга ***
"""
    },
    {
        "id": "doc_04",
        "description": "Паспорт-книжечка 1994: синій паспорт, без латиниці, безстроковий, рукописний шрифт",
        "raw_text": """
ПАСПОРТ ГРОМАДЯНИНА УКРАЇНИ
Серія ЕР № 492018
Прізвище: К Р А В Ч У К
Ім'я: М И Х А Й Л О
По батькові: С Т Е П А Н О В И Ч
Стать: ЧОЛ.
Дата народження: 09 вересня 1974 року
Місце народження: с. Мошни Черкаського р-ну Черкаської обл.
Ким виданий паспорт: Придніпровським РВ УМВС України в Черкаській області
Дата видачі: 24 жовтня 1998 року
Вклеєно фотокартку при досягненні 45 років: 15.10.2019
"""
    },
    {
        "id": "doc_05",
        "description": "ID-картка зворотний бік: тільки MRZ та технічні реквізити, обрізаний край",
        "raw_text": """
<< ОРГАН ЩО ВИДАВ: 2610 >>
ДАТА ВИДАЧІ: 18.03.2019
УНЗР: 19991204-02391
НОМЕР ДОКУМЕНТА: 001837465
I<UKR0018374658<<<<<<<<<<<<<<<
9912041M2903184UKR19991204-02391<6
MOROZ<<VOLODYMYR<<<<<<<<<<<<<<
*** ЛІВИЙ КРАЙ ОБРІЗАНО ПРИ СКАНУВАННІ ***
"""
    },
    {
        "id": "doc_06",
        "description": "Закордонний паспорт: перешкоди, відблиск, пошкоджена MRZ",
        "raw_text": """
UKRAINE  PASSPORT
Type P  Code UKR  Passport No: FX102948
Surname: Т К А Ч Е Н К О  /  T K A C H E N K O
Given Names: О Л Е Н А  /  O L E N A
Sex: Ж / F   Nationality: UKRAINE
Date of birth: 19 ДЕР/DEC 1996
Date of issue: 05.02.2021   Date of expiry: 05.02.2031
Authority: 1204
Запис № / Personal No: 19961219-04820
P<UKRTKACHENKO<<OLENA<<<<<<<<<<<<<<<<<<<<<<<
FX102948<#UKR961219#F310205819961219-04820<##
*** ПОШКОДЖЕННЯ: ЗАКРЕСЛЕНО РУЧКОЮ В РАЙОНІ MRZ ***
"""
    },
    {
        "id": "doc_07",
        "description": "Посвідчення водія: пошкоджене фото, категорія A/B, одруківка в серії",
        "raw_text": """
DRIVING LICENCE / УКРАЇНА
1. САВЧУК / SAVCHUK
2. А Н Н А  В І К Т О Р І В Н А / ANNA
3. 30.08.1993 КИЇВСЬКА ОБЛ.
4a. 12.06.2015  4b. 12.06.2045  4c. ТСЦ 8042
5. CXA 194820
Стать: Жіноча
Категорії: A1, A, B
*** ПРИМІТКА: ТРІЩИНА НА ПЛАСТИКУ ПОВЕРХ ПРІЗВИЩА ***
"""
    },
    {
        "id": "doc_08",
        "description": "ID-картка лицьовий бік: чистий скан, без MRZ, повні ПІБ",
        "raw_text": """
УКРАЇНА / UKRAINE
ПАСПОРТ ГРОМАДЯНИНА УКРАЇНИ / IDENTITY CARD
Прізвище / Surname: П О Л І Щ У К / P O L I S H C H U K
Ім'я / Given Name: І Г О Р / I H O R
По батькові: В А Л Е Н Т И Н О В И Ч
Стать / Sex: Ч / M
Громадянство / Nationality: УКРАЇНА / UKR
Дата народження / Date of birth: 05.01.2001
Дійсний до / Date of expiry: 05.01.2031
Документ № / Document No: 004819203
Запис № / Record No: 20010105-01934
"""
    },
    {
        "id": "doc_09",
        "description": "Паспорт-книжечка 1994 року: потерта сторінка, штамп прописки перекриває текст",
        "raw_text": """
ПАСПОРТ ГРОМАДЯНИНА УКРАЇНИ
Серія СН номер 738192
Прізвище: П А В Л Е Н К О
Ім'я: Н А Т А Л І Я
По батькові: О Л Е Г І В Н А
Стать: ЖІН.
Дата народження: 17 травня 1982 року
Місце народження: м. Одеса
Паспорт виданий: Приморським РВ ОМУ УМВС України в Одеській області
Дата видачі: 20 червня 1998 р.
[ШТАМП РЕЄСТРАЦІЇ МІСЦЯ ПРОЖИВАННЯ: м. Одеса, вул. Дерибасівська...]
"""
    },
    {
        "id": "doc_10",
        "description": "Закордонний паспорт: подовжений термін дії під час воєнного стану (штамп)",
        "raw_text": """
UKRAINE / ПАСПОРТ ГРОМАДЯНИНА УКРАЇНИ ДЛЯ ВИЇЗДУ ЗА КОРДОН
Type/Тип: P   Code: UKR   No: FF729104
Surname: Л И С Е Н К О  /  L Y S E N K O
Given names: В А Д И М  /  V A D Y M
Nationality: УКРАЇНА
Date of birth: 08 КВІ / APR 1979
Sex: Ч / M   Place of birth: ПОЛТАВСЬКА ОБЛ.
Date of issue: 14 ТРА / MAY 2013
Date of expiry: 14 ТРА / MAY 2023
Authority: 5301  Personal No: 19790408-01120
*** ВІДМІТКА: ТЕРМІН ДІЇ ПРОДОВЖЕНО ДО 14.05.2028 ШТАМПОМ ДМС ***
P<UKRLYSENKO<<VADYM<<<<<<<<<<<<<<<<<<<<<<<<<
FF729104<2UKR7904085M230514819790408-01120<14
"""
    },
    # Документи 11-30 для загального датасету (тестування стійкості моделі)
    {
        "id": "doc_11",
        "raw_text": "UKRAINE IDENTITY CARD. Surname: ROMANENKO, Name: YULIIA. Doc No: 003819284. DOB: 1995-11-20. Sex: F. Record: 19951120-09381. Authority: 8013. Expiry: 2030-11-20. I<UKR0038192842<<<<<<<<<<<<<<<"
    },
    {
        "id": "doc_12",
        "raw_text": "ПАСПОРТ КНИЖЕЧКА СЕРИЯ ВА 192837. ВАСИЛЬЧУК РОМАН ПЕТРОВИЧ. ЧОЛ. 12.03.1968 Р.Н. ВИДАНИЙ ЛУЦЬКИМ МВ УМВС. 15.04.1996."
    },
    {
        "id": "doc_13",
        "raw_text": "DRIVING LICENCE УКРАЇНА. ПРІЗВИЩЕ: ГРИЦЕНКО / HRYTSENKO. ІМ'Я: ТАРАС / TARAS. НОМЕР: АХВ 991823. ДАТА ВИДАЧІ: 2019-08-10. ДІЙСНЕ ДО: 2049-08-10. ТСЦ 1241. КАТЕГОРІЯ B."
    },
    {
        "id": "doc_14",
        "raw_text": "PASSPORT UKRAINE. SURNAME: SYDORENKO / СИДОРЕНКО. GIVEN NAME: MARIIA / МАРІЯ. PASSPORT NO: FG992817. SEX: F. DOB: 02.09.1991. ISSUE: 11.11.2018. EXPIRY: 11.11.2028. P<UKRSYDORENKO<<MARIIA<<<<<<<<<<<<<<"
    },
    {
        "id": "doc_15",
        "raw_text": "ID-КАРТКА 001928374. КУЗЬМЕНКО ОЛЕНА СЕРГІЇВНА / KUZMENKO OLENA. СТАТЬ Ж / F. Д.Н. 28.02.1989. ДІЙСНИЙ ДО: 28.02.2029. УНЗР: 19890228-09182. ОРГАН 5612."
    },
    {
        "id": "doc_16",
        "raw_text": "ПАСПОРТ УКРАЇНА МТ 391029. МАРЧЕНКО ДАРИНА МИКОЛАЇВНА. ЖІН. 19.12.1980 МІСЦЕ НАРОДЖЕННЯ М. ЧЕРНІГІВ. ДАТА ВИДАЧІ: 10 СІЧНЯ 1997."
    },
    {
        "id": "doc_17",
        "raw_text": "ПОСВІДЧЕННЯ ВОДІЯ. ЯКОВЕНКО СЕРГІЙ / YAKOVENKO SERHII. НОМЕР: ВАЕ 381920. НАРОДЖ. 14.07.1977. ТЕРМІН ДІЇ: 2011-04-20 ДО 2041-04-20. ТСЦ 0541."
    },
    {
        "id": "doc_18",
        "raw_text": "UKR PASSPORT TRAVEL. NO: FJ192831. SURNAME: PETRENKO. GIVEN NAME: VOLODYMYR. SEX: M. DOB: 1984-06-15. EXP: 2032-06-15. P<UKRPETRENKO<<VOLODYMYR<<<<<<<<<<<<<"
    },
    {
        "id": "doc_19",
        "raw_text": "ID КАРТКА УНЗР: 20030412-08123. ПАСПОРТ: 005918237. БІЛОУС АНДРІЙ РОМАНОВИЧ / BILOUS ANDRII. Ч / M. 12.04.2003. ДІЙСНИЙ ДО: 12.04.2033. ОРГАН: 7110."
    },
    {
        "id": "doc_20",
        "raw_text": "СИНИЙ ПАСПОРТ КНИЖКА. СЕРИЯ НВ 849201. ФЕДОРЕНКО ВІТАЛІЙ ІВАНОВИЧ. ЧОЛ. 03.10.1971. ВИДАНИЙ КИЇВСЬКИМ РВ М. ХАРКОВА 1995 Р."
    },
    {
        "id": "doc_21",
        "raw_text": "PERMIS DE CONDUIRE / UKRAINE. ВОДІЙСЬКЕ: АВВ 728193. МАТВІЄНКО КАТЕРИНА / MATVIIENKO KATERYNA. 1990-10-05. ДАТА ВИДАЧІ: 2016-03-12. ДІЙСНЕ ДО: 2046-03-12. КАТЕГОРІЯ B1, B."
    },
    {
        "id": "doc_22",
        "raw_text": "PASSPORT OF UKRAINE. DOC: FR391024. SURNAME: STEPANENKO. NAME: INNA. SEX: F. DOB: 1993-01-25. EXPIRES: 2029-08-14. AUTHORITY: 3201. P<UKRSTEPANENKO<<INNA<<<<<<<<<<<<<<<<"
    },
    {
        "id": "doc_23",
        "raw_text": "ID CARD UKRAINE. NO: 003910283. SURNAME: DANYLENKO. NAME: MAXTM (OCR ERROR, SHOULD BE MAKSYM). SEX: M. DOB: 1997-04-30. RECORD: 19970430-01928. EXPIRES: 2027-04-30."
    },
    {
        "id": "doc_24",
        "raw_text": "ПАСПОРТ УКРАЇНИ КНИЖЕЧКА. СЕРІЯ СС 291039. ОНОПРІЄНКО ГАЛИНА ПЕТРІВНА. ЖІН. 1965-08-12. ВИДАНИЙ ДНІПРОВСЬКИМ РВ КИЄВА 1996-12-01."
    },
    {
        "id": "doc_25",
        "raw_text": "DRIVING LICENCE 1. ЛИТВИН / LYTVYN 2. ОЛЕКСАНДР 3. 1982-03-19 4a. 2020-01-15 4b. 2050-01-15 5. ВСА 391820. CATEGORY: C, CE."
    },
    {
        "id": "doc_26",
        "raw_text": "PASSPORT FOR TRAVEL ABROAD. NO: FL819203. HOLUB NATALIIA / ГОЛУБ НАТАЛІЯ. FEMALE. DOB: 1998-09-09. ISSUED: 2022-05-10. EXPIRES: 2032-05-10. P<UKRHOLUB<<NATALIIA<<<<<<<<<<<<<<<<<<<<"
    },
    {
        "id": "doc_27",
        "raw_text": "ID КАРТА УКРАЇНИ. №002819034. ВЛАСНИК: ВОЛОШИН АРТЕМ ДМИТРОВИЧ / VOLOSHYN ARTEM. СТАТЬ: Ч. НАР: 1994-12-14. ДІЙСНА ДО: 2034-12-14. ЗАПИС: 19941214-03912. ОРГАН: 6811."
    },
    {
        "id": "doc_28",
        "raw_text": "СЕРІЯ АМ № 591023. ІВАНЧУК ВАСИЛЬ МИХАЙЛОВИЧ. ЧОЛ. 1959 РОКУ НАРОДЖЕННЯ. ВИДАНИЙ УЖГОРОДСЬКИМ МВ 1997 РОКУ."
    },
    {
        "id": "doc_29",
        "raw_text": "ВОДІЙСЬКЕ ПОСВІДЧЕННЯ СХЕ 928301. ДОВЖЕНКО ВАЛЕРІЙ. ДАТА: 1991-07-22. ВИДАНО: 2017-09-01. ДІЙСНЕ ДО: 2047-09-01. КАТ. B."
    },
    {
        "id": "doc_30",
        "raw_text": "UKRAINE PASSPORT. FB482910. KOZLOV YEVHEN / КОЗЛОВ ЄВГЕН. MALE. BORN: 1987-03-03. RECORD NO: 19870303-01829. P<UKRKOZLOV<<YEVHEN<<<<<<<<<<<<<<<<<<<<<"
    }
]


# Еталонна розмітка (Golden Set) для перших 10 документів
GOLDEN_SET = [
    {
        "id": "doc_01",
        "expected": {
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
            "mrz_line": "I<UKR0029481724<<<<<<<<<<<<<<< 9205142M3205146UKR19920514-01824<8 KOVALENKO<<OLEKSII<<<<<<<<<<<<",
            "ocr_quality_flags": ["GLARE", "NOISY_SYMBOLS"],
            "confidence_score": 0.95
        }
    },
    {
        "id": "doc_02",
        "expected": {
            "document_type": "FOREIGN_PASSPORT",
            "document_number": "FN384910",
            "record_number": "19880722-03911",
            "holder": {
                "last_name_ukr": "МЕЛЬНИК",
                "first_name_ukr": "ОКСАНА",
                "patronymic_ukr": None,
                "last_name_en": "MELNYK",
                "first_name_en": "OKSANA"
            },
            "gender": "FEMALE",
            "dates": {
                "birth_date": "1988-07-22",
                "issue_date": "2020-09-10",
                "expiry_date": "2030-09-10"
            },
            "issuing_authority": "8001",
            "mrz_line": "P<UKRMELNYK<<OKSANA<<<<<<<<<<<<<<<<<<<<<<<<< FN384910<4UKR8807223F300910119880722-03911<42",
            "ocr_quality_flags": [],
            "confidence_score": 1.0
        }
    },
    {
        "id": "doc_03",
        "expected": {
            "document_type": "DRIVERS_LICENSE",
            "document_number": "BXA 839201",
            "record_number": None,
            "holder": {
                "last_name_ukr": "БОНДАРЕНКО",
                "first_name_ukr": "ДМИТРО",
                "patronymic_ukr": "ІГОРОВИЧ",
                "last_name_en": "BONDARENKO",
                "first_name_en": "DMYTRO"
            },
            "gender": "MALE",
            "dates": {
                "birth_date": "1985-11-03",
                "issue_date": "2018-02-15",
                "expiry_date": "2048-02-15"
            },
            "issuing_authority": "ТСЦ 6341",
            "mrz_line": None,
            "ocr_quality_flags": ["BLURRY"],
            "confidence_score": 0.9
        }
    },
    {
        "id": "doc_04",
        "expected": {
            "document_type": "PASSPORT_BOOKLET",
            "document_number": "ЕР 492018",
            "record_number": None,
            "holder": {
                "last_name_ukr": "КРАВЧУК",
                "first_name_ukr": "МИХАЙЛО",
                "patronymic_ukr": "СТЕПАНОВИЧ",
                "last_name_en": None,
                "first_name_en": None
            },
            "gender": "MALE",
            "dates": {
                "birth_date": "1974-09-09",
                "issue_date": "1998-10-24",
                "expiry_date": None
            },
            "issuing_authority": "Придніпровським РВ УМВС України в Черкаській області",
            "mrz_line": None,
            "ocr_quality_flags": [],
            "confidence_score": 0.95
        }
    },
    {
        "id": "doc_05",
        "expected": {
            "document_type": "ID_CARD",
            "document_number": "001837465",
            "record_number": "19991204-02391",
            "holder": {
                "last_name_ukr": None,
                "first_name_ukr": None,
                "patronymic_ukr": None,
                "last_name_en": "MOROZ",
                "first_name_en": "VOLODYMYR"
            },
            "gender": "MALE",
            "dates": {
                "birth_date": "1999-12-04",
                "issue_date": "2019-03-18",
                "expiry_date": "2029-03-18"
            },
            "issuing_authority": "2610",
            "mrz_line": "I<UKR0018374658<<<<<<<<<<<<<<< 9912041M2903184UKR19991204-02391<6 MOROZ<<VOLODYMYR<<<<<<<<<<<<<<",
            "ocr_quality_flags": ["CROPPED_TEXT"],
            "confidence_score": 0.85
        }
    },
    {
        "id": "doc_06",
        "expected": {
            "document_type": "FOREIGN_PASSPORT",
            "document_number": "FX102948",
            "record_number": "19961219-04820",
            "holder": {
                "last_name_ukr": "ТКАЧЕНКО",
                "first_name_ukr": "ОЛЕНА",
                "patronymic_ukr": None,
                "last_name_en": "TKACHENKO",
                "first_name_en": "OLENA"
            },
            "gender": "FEMALE",
            "dates": {
                "birth_date": "1996-12-19",
                "issue_date": "2021-02-05",
                "expiry_date": "2031-02-05"
            },
            "issuing_authority": "1204",
            "mrz_line": "P<UKRTKACHENKO<<OLENA<<<<<<<<<<<<<<<<<<<<<<< FX102948<#UKR961219#F310205819961219-04820<##",
            "ocr_quality_flags": ["NOISY_SYMBOLS"],
            "confidence_score": 0.85
        }
    },
    {
        "id": "doc_07",
        "expected": {
            "document_type": "DRIVERS_LICENSE",
            "document_number": "CXA 194820",
            "record_number": None,
            "holder": {
                "last_name_ukr": "САВЧУК",
                "first_name_ukr": "АННА",
                "patronymic_ukr": "ВІКТОРІВНА",
                "last_name_en": "SAVCHUK",
                "first_name_en": "ANNA"
            },
            "gender": "FEMALE",
            "dates": {
                "birth_date": "1993-08-30",
                "issue_date": "2015-06-12",
                "expiry_date": "2045-06-12"
            },
            "issuing_authority": "ТСЦ 8042",
            "mrz_line": None,
            "ocr_quality_flags": ["NOISY_SYMBOLS"],
            "confidence_score": 0.9
        }
    },
    {
        "id": "doc_08",
        "expected": {
            "document_type": "ID_CARD",
            "document_number": "004819203",
            "record_number": "20010105-01934",
            "holder": {
                "last_name_ukr": "ПОЛІЩУК",
                "first_name_ukr": "ІГОР",
                "patronymic_ukr": "ВАЛЕНТИНОВИЧ",
                "last_name_en": "POLISHCHUK",
                "first_name_en": "IHOR"
            },
            "gender": "MALE",
            "dates": {
                "birth_date": "2001-01-05",
                "issue_date": None,
                "expiry_date": "2031-01-05"
            },
            "issuing_authority": None,
            "mrz_line": None,
            "ocr_quality_flags": [],
            "confidence_score": 0.95
        }
    },
    {
        "id": "doc_09",
        "expected": {
            "document_type": "PASSPORT_BOOKLET",
            "document_number": "СН 738192",
            "record_number": None,
            "holder": {
                "last_name_ukr": "ПАВЛЕНКО",
                "first_name_ukr": "НАТАЛІЯ",
                "patronymic_ukr": "ОЛЕГІВНА",
                "last_name_en": None,
                "first_name_en": None
            },
            "gender": "FEMALE",
            "dates": {
                "birth_date": "1982-05-17",
                "issue_date": "1998-06-20",
                "expiry_date": None
            },
            "issuing_authority": "Приморським РВ ОМУ УМВС України в Одеській області",
            "mrz_line": None,
            "ocr_quality_flags": ["NOISY_SYMBOLS"],
            "confidence_score": 0.9
        }
    },
    {
        "id": "doc_10",
        "expected": {
            "document_type": "FOREIGN_PASSPORT",
            "document_number": "FF729104",
            "record_number": "19790408-01120",
            "holder": {
                "last_name_ukr": "ЛИСЕНКО",
                "first_name_ukr": "ВАДИМ",
                "patronymic_ukr": None,
                "last_name_en": "LYSENKO",
                "first_name_en": "VADYM"
            },
            "gender": "MALE",
            "dates": {
                "birth_date": "1979-04-08",
                "issue_date": "2013-05-14",
                "expiry_date": "2028-05-14"
            },
            "issuing_authority": "5301",
            "mrz_line": "P<UKRLYSENKO<<VADYM<<<<<<<<<<<<<<<<<<<<<<<<< FF729104<2UKR7904085M230514819790408-01120<14",
            "ocr_quality_flags": [],
            "confidence_score": 0.9
        }
    }
]


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    raw_path = data_dir / "raw_documents.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(RAW_DOCUMENTS, f, ensure_ascii=False, indent=2)
    print(f"Created {raw_path} with {len(RAW_DOCUMENTS)} documents.")

    golden_path = data_dir / "golden_set.json"
    with open(golden_path, "w", encoding="utf-8") as f:
        json.dump(GOLDEN_SET, f, ensure_ascii=False, indent=2)
    print(f"Created {golden_path} with {len(GOLDEN_SET)} golden annotations.")


if __name__ == "__main__":
    main()
