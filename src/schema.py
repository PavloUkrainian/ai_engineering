from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Тип документа, що посвідчує особу."""
    ID_CARD = "ID_CARD"                      # Паспорт громадянина України у формі ID-картки
    PASSPORT_BOOKLET = "PASSPORT_BOOKLET"    # Паспорт зразка 1994 року (синя книжечка)
    FOREIGN_PASSPORT = "FOREIGN_PASSPORT"    # Паспорт громадянина України для виїзду за кордон
    DRIVERS_LICENSE = "DRIVERS_LICENSE"      # Посвідчення водія
    UNKNOWN = "UNKNOWN"                      # Не вдалося ідентифікувати


class Gender(str, Enum):
    """Стать особи."""
    MALE = "MALE"          # Чоловіча (Ч / M)
    FEMALE = "FEMALE"      # Жіноча (Ж / F)
    UNKNOWN = "UNKNOWN"    # Не вказано або не розпізнано


class PersonName(BaseModel):
    """ПІБ власника документа (вкладений об'єкт)."""
    last_name_ukr: Optional[str] = Field(
        default=None,
        description="Прізвище українською мовою (наприклад: Шевченко)"
    )
    first_name_ukr: Optional[str] = Field(
        default=None,
        description="Ім'я українською мовою (наприклад: Тарас)"
    )
    patronymic_ukr: Optional[str] = Field(
        default=None,
        description="По батькові українською мовою (для книжечок та ID-карток, якщо наявне)"
    )
    last_name_en: Optional[str] = Field(
        default=None,
        description="Прізвище латиницею (наприклад: Shevchenko)"
    )
    first_name_en: Optional[str] = Field(
        default=None,
        description="Ім'я латиницею (наприклад: Taras)"
    )


class DocumentDates(BaseModel):
    """Ключові дати документа у форматі YYYY-MM-DD (вкладений об'єкт)."""
    birth_date: Optional[str] = Field(
        default=None,
        description="Дата народження особи у форматі РРРР-ММ-ДД (наприклад: 1990-05-14)"
    )
    issue_date: Optional[str] = Field(
        default=None,
        description="Дата видачі документа у форматі РРРР-ММ-ДД"
    )
    expiry_date: Optional[str] = Field(
        default=None,
        description="Дата закінчення терміну дії (null для безстрокових паспортів-книжечок)"
    )


class ExtractedDocument(BaseModel):
    """
    Головна схема вилученого документа з сирого OCR-тексту.
    Містить:
    - 2 Enums: DocumentType, Gender
    - 2 вкладені моделі: PersonName, DocumentDates
    - список дефектів OCR: ocr_quality_flags (List[str])
    - скалярні поля: document_number, record_number, issuing_authority, mrz_line, confidence_score
    Разом: 9 полів верхнього рівня (вимога ДЗ >= 6).
    """
    document_type: DocumentType = Field(
        description="Тип документа (ID_CARD, PASSPORT_BOOKLET, FOREIGN_PASSPORT, DRIVERS_LICENSE, UNKNOWN)"
    )
    document_number: Optional[str] = Field(
        default=None,
        description="Номер документа (наприклад: 001234567 для ID-картки, ЕР 123456 для книжечки)"
    )
    record_number: Optional[str] = Field(
        default=None,
        description="УНЗР (Унікальний номер запису в реєстрі) у форматі РРРРММДД-XXXXX (лише для біометричних)"
    )
    holder: PersonName = Field(
        default_factory=PersonName,
        description="ПІБ власника документа (кирилиця та латиниця)"
    )
    gender: Gender = Field(
        default=Gender.UNKNOWN,
        description="Стать особи: MALE, FEMALE, UNKNOWN"
    )
    dates: DocumentDates = Field(
        default_factory=DocumentDates,
        description="Дати народження, видачі та закінчення дії документа"
    )
    issuing_authority: Optional[str] = Field(
        default=None,
        description="Орган, що видав документ (наприклад: 8012 або назва підрозділу ДМС)"
    )
    mrz_line: Optional[str] = Field(
        default=None,
        description="Машинозчитувані стрічки (MRZ), якщо виявлені в тексті"
    )
    ocr_quality_flags: List[str] = Field(
        default_factory=list,
        description="Виявлені проблеми якості скану: GLARE, CROPPED_TEXT, BLURRY, NOISY_SYMBOLS тощо"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Оцінка впевненості моделі у вилучених даних від 0.0 до 1.0"
    )
