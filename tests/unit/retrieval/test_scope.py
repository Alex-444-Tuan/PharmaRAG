import pytest

from pharmarag.retrieval.scope import FilterRequiredError, scopes


def test_keys_become_scopes_one_per_drug():
    assert scopes({"ingredient_keys": ["1", "2+3"]}) == ["1", "2+3"]


def test_no_mention_and_no_keys_is_the_unfiltered_path():
    assert scopes({}) == [None]
    assert scopes({"mentions": [], "ingredient_keys": []}) == [None]


def test_mention_without_keys_refuses_unfiltered_ranking():
    with pytest.raises(FilterRequiredError):
        scopes({"mentions": ["hydroxyzin"], "ingredient_keys": []})
    with pytest.raises(FilterRequiredError):
        scopes({"mentions": ["hydroxyzin"]})
