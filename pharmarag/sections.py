"""LOINC section codes kept from SPL labels (spec section 4). Contract file.

Ingestion keeps only these sections. `analyze` maps a question to target codes from
this table. Verify against downloaded labels during M1 and request changes via report.
"""

PRESCRIPTION_LABEL_LOINC = "34391-3"  # SPL document type: human prescription drug label

SECTIONS: dict[str, str] = {
    "34066-1": "Boxed warning",
    "34067-9": "Indications and usage",
    "34068-7": "Dosage and administration",
    "43678-2": "Dosage forms and strengths",
    "34070-3": "Contraindications",
    "43685-7": "Warnings and precautions",
    "34071-1": "Warnings",
    "42232-9": "Precautions",
    "34084-4": "Adverse reactions",
    "34073-7": "Drug interactions",
    "43684-0": "Use in specific populations",
    "34088-5": "Overdosage",
    "34090-1": "Clinical pharmacology",
    "34069-5": "How supplied, storage and handling",
}
