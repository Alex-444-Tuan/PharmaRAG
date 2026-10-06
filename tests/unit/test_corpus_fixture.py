import pytest


@pytest.mark.corpus
async def test_corpus_is_read_only_and_populated(corpus_conn):
    cur = await corpus_conn.execute("SELECT count(*) FROM label")
    assert (await cur.fetchone())[0] > 0
    with pytest.raises(Exception, match="read-only"):
        await corpus_conn.execute("DELETE FROM corpus_meta")
