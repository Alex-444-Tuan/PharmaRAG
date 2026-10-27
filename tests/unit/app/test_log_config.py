"""NFR-5: every log line is JSON, including uvicorn's and tracebacks."""

import json
import logging
import logging.config

import pytest

from pharmarag.app.observability import LOG_CONFIG_PATH


@pytest.fixture
def restore_logging():
    """dictConfig is process-global: put the loggers back the way the other tests expect."""
    names = ("", "uvicorn", "uvicorn.error", "uvicorn.access")
    saved = {
        n: (list(lg.handlers), lg.level, lg.propagate)
        for n in names
        for lg in [logging.getLogger(n)]
    }
    yield
    for n, (handlers, level, propagate) in saved.items():
        lg = logging.getLogger(n)
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate


def json_lines(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def test_the_log_config_file_is_valid_json_for_dictconfig(restore_logging):
    config = json.loads(LOG_CONFIG_PATH.read_text())
    logging.config.dictConfig(config)  # raises on a bad formatter path or handler


def test_uvicorn_loggers_and_tracebacks_come_out_as_json(restore_logging, capsys):
    logging.config.dictConfig(json.loads(LOG_CONFIG_PATH.read_text()))
    logging.getLogger("uvicorn.error").info("Started server process [1]")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "10.0.0.1:5", "GET", "/healthz", "1.1", 200
    )
    try:
        raise ValueError("boom")
    except ValueError:
        # What uvicorn logs when the app raises (after Starlette sent the 500).
        logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
    logging.getLogger("pharmarag.app.main").warning("dependency failed mid-stream")

    lines = json_lines(capsys.readouterr().out)
    assert [ln["logger"] for ln in lines] == [
        "uvicorn.error",
        "uvicorn.access",
        "uvicorn.error",
        "pharmarag.app.main",
    ]
    assert lines[0]["message"] == "Started server process [1]"
    assert "GET /healthz" in lines[1]["message"]
    assert lines[2]["level"] == "ERROR" and "ValueError: boom" in lines[2]["exc"]
    assert "Traceback" in lines[2]["exc"]  # the traceback is a field, not loose lines on stdout


def test_a_message_with_newlines_stays_one_json_line(restore_logging, capsys):
    logging.config.dictConfig(json.loads(LOG_CONFIG_PATH.read_text()))
    logging.getLogger("uvicorn.error").warning("first\nsecond")
    (line,) = json_lines(capsys.readouterr().out)
    assert line["message"] == "first\nsecond"
