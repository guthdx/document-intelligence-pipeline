"""Entity IR models for extracted structured data."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Optional, Union
from uuid import UUID

from pydantic import Field

from .base import BaseIRModel, BoundingBox, ConfidenceLevel, Provenance


class EntityType(str, Enum):
    """Types of entities that can be extracted."""

    # Dates and times
    DATE = "date"
    DATE_RANGE = "date_range"
    TIME = "time"
    DATETIME = "datetime"

    # Medical/clinical
    LAB_VALUE = "lab_value"
    VITAL_SIGN = "vital_sign"
    MEDICATION = "medication"
    DOSAGE = "dosage"
    DIAGNOSIS = "diagnosis"
    PROCEDURE = "procedure"
    ALLERGY = "allergy"
    SYMPTOM = "symptom"

    # Identifiers/codes
    ICD_CODE = "icd_code"
    CPT_CODE = "cpt_code"
    NDC_CODE = "ndc_code"
    LOINC_CODE = "loinc_code"
    MRN = "mrn"  # Medical Record Number

    # People and organizations
    PROVIDER_NAME = "provider_name"
    PATIENT_NAME = "patient_name"
    ORGANIZATION = "organization"
    FACILITY = "facility"

    # Quantities
    NUMERIC_VALUE = "numeric_value"
    MEASUREMENT = "measurement"
    PERCENTAGE = "percentage"
    MONEY = "money"

    # Other
    PHONE = "phone"
    ADDRESS = "address"
    EMAIL = "email"
    REFERENCE = "reference"  # Reference to other document/section
    OTHER = "other"


class Entity(BaseIRModel):
    """
    Extracted structured entity with provenance.

    Entities are values extracted from text blocks that can be
    queried and validated. Each entity links back to its source.
    """

    # Parent references
    document_id: UUID
    page_id: UUID
    block_id: UUID
    page_number: int = Field(..., ge=1)

    # Entity classification
    entity_type: EntityType
    sub_type: Optional[str] = Field(
        None, description="More specific classification within type"
    )

    # Extracted value
    raw_text: str = Field(..., description="Original text as it appears in source")
    normalized_value: Optional[str] = Field(
        None, description="Normalized/parsed value"
    )
    parsed_value: Optional[Any] = Field(
        None, description="Parsed value in native type (date, float, etc.)"
    )

    # Source location
    bbox: BoundingBox
    char_span: Optional[tuple[int, int]] = Field(
        None, description="Character offsets in block text [start, end)"
    )

    # Confidence
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM)
    extraction_model: str = Field(..., description="Model used for extraction")

    # Relationships
    related_entity_ids: list[UUID] = Field(
        default_factory=list,
        description="Related entities (e.g., medication linked to dosage)",
    )

    # Validation
    is_validated: bool = Field(default=False)
    validation_errors: list[str] = Field(default_factory=list)
    needs_audit: bool = Field(default=False)

    def get_provenance(self) -> Provenance:
        """Get provenance object for this entity."""
        return Provenance(
            document_id=self.document_id,
            page_number=self.page_number,
            block_id=self.block_id,
            bbox=self.bbox,
            confidence=self.confidence,
            stage="entity_extraction",
            model_name=self.extraction_model,
        )


class LabValue(Entity):
    """Specialized entity for lab results."""

    entity_type: EntityType = Field(default=EntityType.LAB_VALUE)

    # Lab-specific fields
    test_name: str = Field(..., description="Name of the lab test")
    test_code: Optional[str] = Field(None, description="LOINC or lab code")
    value: Union[float, str] = Field(..., description="Test result value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    reference_range: Optional[str] = Field(None, description="Normal reference range")
    flag: Optional[str] = Field(
        None, description="Abnormal flag: H, L, HH, LL, A, etc."
    )
    collection_date: Optional[date] = None
    result_date: Optional[date] = None

    @property
    def is_abnormal(self) -> bool:
        """Check if value is flagged as abnormal."""
        return self.flag is not None and self.flag.upper() in [
            "H", "L", "HH", "LL", "A", "C", "HIGH", "LOW", "ABNORMAL", "CRITICAL"
        ]


class Medication(Entity):
    """Specialized entity for medications."""

    entity_type: EntityType = Field(default=EntityType.MEDICATION)

    # Medication-specific fields
    drug_name: str = Field(..., description="Medication name (brand or generic)")
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    ndc_code: Optional[str] = Field(None, description="National Drug Code")
    rxnorm_code: Optional[str] = None
    dosage: Optional[str] = Field(None, description="Dosage amount")
    dosage_unit: Optional[str] = None
    route: Optional[str] = Field(None, description="Route of administration")
    frequency: Optional[str] = Field(None, description="How often taken")
    instructions: Optional[str] = Field(None, description="Special instructions")
    prescriber: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    is_prn: bool = Field(default=False, description="As needed")


class DateEntity(Entity):
    """Specialized entity for dates."""

    entity_type: EntityType = Field(default=EntityType.DATE)

    # Date-specific fields
    parsed_date: Optional[date] = None
    date_format: Optional[str] = Field(
        None, description="Detected format: MM/DD/YYYY, YYYY-MM-DD, etc."
    )
    is_approximate: bool = Field(
        default=False, description="True if date is approximate (e.g., 'around March')"
    )
    date_context: Optional[str] = Field(
        None, description="Context: admission_date, discharge_date, procedure_date, etc."
    )


class DiagnosisEntity(Entity):
    """Specialized entity for diagnoses."""

    entity_type: EntityType = Field(default=EntityType.DIAGNOSIS)

    # Diagnosis-specific fields
    diagnosis_text: str
    icd10_code: Optional[str] = None
    icd9_code: Optional[str] = None
    snomed_code: Optional[str] = None
    is_primary: bool = Field(default=False)
    is_rule_out: bool = Field(default=False, description="Rule out diagnosis")
    diagnosis_date: Optional[date] = None
    diagnosing_provider: Optional[str] = None
