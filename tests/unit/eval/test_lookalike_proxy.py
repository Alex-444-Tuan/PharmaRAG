"""The baseline proxy (unfiltered retrieval) on a two-drug database, so every count is known."""

import json
import uuid

import pytest

from pharmarag.eval import lookalike as la

pytestmark = pytest.mark.pg

HYDROXYZINE_KEY, HYDRALAZINE_KEY = "5553", "5470"

CHUNKS = {
    HYDROXYZINE_KEY: [
        "hydroxyzine dose for anxiety in adults is 50 mg",
        "hydroxyzine is contraindicated in early pregnancy",
    ],
    HYDRALAZINE_KEY: [
        "hydralazine dose for hypertension in adults is 10 mg",
        "hydralazine may cause a lupus like syndrome",
    ],
}


@pytest.fixture
async def two_drugs(pg_conn, fake_embedder):
    for key, texts in CHUNKS.items():
        setid = uuid.uuid4()
        await pg_conn.execute(
            "INSERT INTO label (setid, spl_version, title, ingredient_key, ingredient_rxcuis, "
            "product_rxcuis) VALUES (%s, 1, %s, %s, %s, %s)",
            (setid, f"label {key}", key, [key], []),
        )
        for i, (text, vec) in enumerate(
            zip(texts, fake_embedder.embed_documents(texts), strict=True)
        ):
            await pg_conn.execute(
                "INSERT INTO chunk (setid, spl_version, ingredient_key, section_loinc, "
                "section_title, ordinal, text, spans, embedding) "
                "VALUES (%s, 1, %s, '34068-7', 'Dosage', %s, %s, %s::jsonb, %s)",
                (setid, key, i, text, json.dumps([{"i": 0, "start": 0, "end": len(text)}]), vec),
            )
    await pg_conn.commit()
    return pg_conn


def row(question, gold):
    return la.Row(
        question=question,
        mention="x",
        gold_ingredient_key=gold,
        gold_name="x",
        partner_name="y",
        kind="misspelling",
    )


async def test_proxy_counts_top1_and_top8_separately(two_drugs, fake_embedder, thresholds):
    rows = [
        # The best chunk is the gold drug's, but the other drug's chunks are in the top 8 too.
        row("what is the hydroxyzine dose for anxiety in adults", HYDROXYZINE_KEY),
        # The question is really about hydralazine: with gold hydroxyzine, the top chunk is wrong.
        row("hydralazine dose for hypertension in adults", HYDROXYZINE_KEY),
    ]
    out = await la.run_baseline_proxy(rows, two_drugs, fake_embedder, thresholds.retrieval)

    assert out[0].top1_key == HYDROXYZINE_KEY
    assert not out[0].top1_wrong
    assert out[0].any_top8_wrong  # unfiltered: hydralazine chunks are in the evidence list
    assert {HYDROXYZINE_KEY, HYDRALAZINE_KEY} <= set(out[0].top8_keys)

    assert out[1].top1_key == HYDRALAZINE_KEY
    assert out[1].top1_wrong and out[1].any_top8_wrong

    assert la.proxy_counts(out) == {
        "total": 2,
        "top1_wrong": 1,
        "any_top8_wrong": 2,
        "no_evidence": 0,
    }


async def test_a_question_with_no_matching_text_counts_as_no_evidence_not_wrong(
    two_drugs, fake_embedder, thresholds
):
    # Words absent from every chunk: the lexical branch is empty. Dense still ranks something,
    # so use a retrieval depth of zero to force "nothing returned".
    none = thresholds.retrieval.model_copy(update={"top_k_per_branch": 0})
    (out,) = await la.run_baseline_proxy(
        [row("zzz qqq", HYDROXYZINE_KEY)], two_drugs, fake_embedder, none
    )
    assert out.top1_key is None
    assert not out.top1_wrong and not out.any_top8_wrong
    assert la.proxy_counts([out])["no_evidence"] == 1
