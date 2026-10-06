from datetime import date

import pytest

from pharmarag.ingest.audit import flags

NEW = date(2025, 1, 1)


@pytest.mark.parametrize(
    "title",
    [
        "PAXIL (PAROXETINE HYDROCHLORIDE) TABLET, FILM COATED [APOTEX CORP.]",
        "NEURONTIN (GABAPENTIN) CAPSULE NEURONTIN (GABAPENTIN) TABLET, FILM COATED [VIATRIS]",
        "ZIPPY (DRUGX) CAPSULE, LIQUID FILLED [ACME]",
        "HYDROXYZINE PAMOATE CAPSULE [AMNEAL PHARMACEUTICALS OF NEW YORK LLC]",
        "ZOLOFT (SERTRALINE HYDROCHLORIDE) TABLET, FILM COATED [VIATRIS SPECIALTY LLC]",
    ],
)
def test_ordinary_picks_are_not_flagged(title):
    assert flags(title, NEW) == []


@pytest.mark.parametrize(
    "title, reason",
    [
        ("BRISDELLE (PAROXETINE) CAPSULE [LEGACY PHARMA USA, INC.]", "special-indication product"),
        ("GRALISE (GABAPENTIN) TABLET, FILM COATED [ALMATICA PHARMA LLC]", "special-indication"),
        ("DEPAKOTESPRINKLES (DIVALPROEX SODIUM) CAPSULE [ABBVIE INC.]", "special-indication"),
        ("OZURDEX (DEXAMETHASONE) IMPLANT [ALLERGAN, INC.]", "non-oral-solid form"),
        ("ESTRING (ESTRADIOL) SYSTEM [PFIZER LABORATORIES DIV PFIZER INC]", "non-oral-solid form"),
        ("BELBUCA (BUPRENORPHINE) FILM, SOLUBLE [BDSI]", "non-oral-solid form"),
        ("TEGRETOL (CARBAMAZEPINE) SUSPENSION [NOVARTIS]", "niche dose form"),
        ("OMEPRAZOLE CAPSULE [STAT RX USA LLC]", "repackager"),
    ],
)
def test_special_picks_are_flagged_with_a_reason(title, reason):
    assert any(reason in f for f in flags(title, NEW)), flags(title, NEW)


def test_old_labels_are_flagged():
    assert flags("TRANDATE (LABETALOL) TABLET [PROMETHEUS]", date(2009, 12, 14)) == [
        "old label (2009)"
    ]
    assert flags("TRANDATE (LABETALOL) TABLET [PROMETHEUS]", None) == []
