"""CLI helpers: seed file reading and the mapping-file cache."""

import os
import time

import respx

from pharmarag.ingest.dailymed import RXNORM_MAPPINGS_URL, DailyMedClient
from pharmarag.ingest.run import MAPPINGS_MAX_AGE_S, ensure_mappings, read_seed


def test_read_seed_returns_name_and_group_and_skips_blank_names(tmp_path):
    f = tmp_path / "seed.csv"
    f.write_text("name,lookalike_group\nhydralazine,hydralazine-hydroxyzine\n  lisinopril ,\n,\n")
    assert read_seed(f) == [("hydralazine", "hydralazine-hydroxyzine"), ("lisinopril", "")]


def test_read_seed_tolerates_a_missing_group_column(tmp_path):
    f = tmp_path / "seed.csv"
    f.write_text("name\nwarfarin\n")
    assert read_seed(f) == [("warfarin", "")]


@respx.mock
async def test_mapping_file_is_downloaded_when_missing_and_reused_when_fresh(tmp_path):
    route = respx.get(RXNORM_MAPPINGS_URL).respond(content=b"zip")
    client = DailyMedClient("https://dm.test/v2")
    dest = tmp_path / "cache" / "rxnorm_mappings.zip"

    await ensure_mappings(client, dest, refresh=False)
    await ensure_mappings(client, dest, refresh=False)
    assert route.call_count == 1

    await ensure_mappings(client, dest, refresh=True)
    assert route.call_count == 2


@respx.mock
async def test_stale_mapping_file_is_downloaded_again(tmp_path):
    route = respx.get(RXNORM_MAPPINGS_URL).respond(content=b"new")
    dest = tmp_path / "rxnorm_mappings.zip"
    dest.write_bytes(b"old")
    old = time.time() - MAPPINGS_MAX_AGE_S - 60
    os.utime(dest, (old, old))

    await ensure_mappings(DailyMedClient("https://dm.test/v2"), dest, refresh=False)

    assert route.call_count == 1 and dest.read_bytes() == b"new"
