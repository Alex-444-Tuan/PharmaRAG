"""Pinned label file: round trip, ordering, and missing file."""

from pharmarag.ingest.pins import Pins


def test_missing_file_is_empty(tmp_path):
    pins = Pins.load(tmp_path / "none.csv")
    assert pins.get("17767") == []


def test_round_trip_sorted_numerically_by_rxcui(tmp_path):
    path = tmp_path / "data" / "label_pins.csv"
    pins = Pins(path)
    pins.set("17767", "amlodipine", ["AAAAAAAA-0000-0000-0000-000000000001"])
    pins.set(
        "2599",
        "clonidine",
        ["bbbbbbbb-0000-0000-0000-000000000002", "cccccccc-0000-0000-0000-000000000003"],
    )
    pins.save()

    assert path.read_text().splitlines() == [
        "ingredient_rxcui,name,setid",
        "2599,clonidine,bbbbbbbb-0000-0000-0000-000000000002",
        "2599,clonidine,cccccccc-0000-0000-0000-000000000003",
        "17767,amlodipine,aaaaaaaa-0000-0000-0000-000000000001",
    ]
    again = Pins.load(path)
    assert again.get("2599") == [
        "bbbbbbbb-0000-0000-0000-000000000002",
        "cccccccc-0000-0000-0000-000000000003",
    ]
    assert again.get("17767") == ["aaaaaaaa-0000-0000-0000-000000000001"]  # lowercased


def test_set_replaces_the_previous_pin_and_save_leaves_no_temp_file(tmp_path):
    path = tmp_path / "pins.csv"
    pins = Pins(path)
    pins.set("1", "x", ["a"])
    pins.set("1", "x", ["b"])
    pins.save()
    assert Pins.load(path).get("1") == ["b"]
    assert not list(tmp_path.glob("*.part"))


def test_pins_without_a_path_do_not_write():
    pins = Pins()
    pins.set("1", "x", ["a"])
    pins.save()  # no error, nothing written
    assert pins.get("1") == ["a"]
