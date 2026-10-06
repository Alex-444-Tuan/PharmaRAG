from pharmarag.retrieval.lexical import or_query, strip_drug_terms


def test_strips_mention_and_resolved_name_case_insensitively_per_token():
    q = "Hydroxyzin max dose for HYDROXYZINE in adults"
    assert strip_drug_terms(q, ["hydroxyzin", "Hydroxyzine"]) == "max dose for in adults"


def test_multi_word_terms_strip_each_token():
    assert strip_drug_terms("atorvastatin calcium myopathy", ["atorvastatin calcium"]) == "myopathy"


def test_tokens_are_whole_words_not_substrings():
    # "hydroxyzin" must not eat the front of an unrelated word or of the longer name.
    assert strip_drug_terms("hydroxyzine hydroxyzin", ["hydroxyzin"]) == "hydroxyzine"


def test_nothing_to_strip_keeps_the_query_tokens():
    assert strip_drug_terms("CYP3A4 10mg", []) == "CYP3A4 10mg"
    assert strip_drug_terms("CYP3A4 10mg", ["", "  "]) == "CYP3A4 10mg"


def test_only_the_drug_name_left_is_an_empty_query():
    assert strip_drug_terms("Hydroxyzine, hydroxyzin!", ["hydroxyzin", "hydroxyzine"]) == ""
    assert or_query("") == ""  # and an empty query is what lexical_search skips on


def test_result_still_cannot_inject_tsquery_operators():
    q = strip_drug_terms("a & b | !c <-> d:*", ["a"])
    assert all(ch not in or_query(q) for ch in "&!<>:*")
