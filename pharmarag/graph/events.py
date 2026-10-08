"""Server-sent-event plumbing for graph nodes (spec section 10, "Errors and streaming").

Nodes publish events with LangGraph custom streaming. A request opts in to SSE through
`config["configurable"]["sse"]`, which `run_ask_stream` sets. This is per-run config, not graph
state, so the `RAGState` contract is untouched and a concurrent non-SSE request is unaffected.

Outside a graph run (a node function called directly in a unit test) there is no config, and
everything here degrades to "not SSE, drop the event".
"""

from collections.abc import Callable

from langgraph.config import get_config, get_stream_writer

Emit = Callable[[dict], None]


def _drop(_: dict) -> None:
    return None


def is_sse() -> bool:
    try:
        return bool(get_config().get("configurable", {}).get("sse"))
    except RuntimeError:
        return False


def emitter() -> Emit:
    """A function that publishes one custom event, or a no-op when nothing is listening."""
    if not is_sse():
        return _drop
    try:
        return get_stream_writer()
    except (RuntimeError, KeyError):
        return _drop
