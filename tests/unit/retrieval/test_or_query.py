from pharmarag.retrieval.lexical import MAX_QUERY_TERMS, or_query


def test_or_query_joins_tokens_with_or_and_dedupes():
    assert or_query("Hydroxyzine dosage, hydroxyzine ANXIETY adults") == (
        "hydroxyzine | dosage | anxiety | adults"
    )


def test_or_query_keeps_exact_tokens_like_cyp3a4_and_doses():
    assert or_query("CYP3A4 10mg") == "cyp3a4 | 10mg"


def test_or_query_cannot_inject_tsquery_operators():
    q = or_query("a & b | !c <-> d:* ') OR (1=1 --")
    assert q == "a | b | c | d | or | 1"
    assert all(ch not in q for ch in "&!<>:*'()=-")


def test_or_query_empty_and_capped():
    assert or_query("?! --") == ""
    assert len(or_query(" ".join(f"w{i}" for i in range(100))).split(" | ")) == MAX_QUERY_TERMS
