"""Per-request health facts that are not graph state: did RxNav fail during this request?

`RAGState` is a contract file, and these facts are about the request, not the question, so they
travel in the run config like the SSE flag (graph/events.py). `run_ask` and `run_ask_stream` put
one `RequestHealth` in `config["configurable"]`; `fetch_rx_facts` writes to it and `finalize`
reads it into `meta.facts_unavailable`. Outside a graph run (a node called directly in a unit
test) there is no config and everything degrades to "nothing recorded".
"""

from dataclasses import dataclass

from langgraph.config import get_config

KEY = "health"


@dataclass
class RequestHealth:
    # The RxNorm facts branch returned nothing because RxNav was down (not because no facts
    # exist), so the answer was written from label text alone.
    facts_unavailable: bool = False


def request_health() -> RequestHealth | None:
    try:
        return get_config().get("configurable", {}).get(KEY)
    except RuntimeError:
        return None
