"""Pinned reference labels: `data/label_pins.csv`.

SPEC: the spec ranks by "latest effective date", and repackagers republish constantly, so a
fresh selection can differ from run to run. That would change `corpus_version` and prune the
previous label every week (NFR-2 wants reproducible corpora). A pin records the setid chosen
for an ingredient. Later runs keep the pinned setid while it still exists and passes the
checks, and take only its newest version, so `corpus_version` changes only when that label is
really updated. `--repin` ignores the pins and selects afresh.

CSV columns: ingredient_rxcui, name, setid. One row per pinned setid.
"""

from __future__ import annotations

import csv
from pathlib import Path

FIELDS = ["ingredient_rxcui", "name", "setid"]


class Pins:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._names: dict[str, str] = {}
        self._setids: dict[str, list[str]] = {}

    @classmethod
    def load(cls, path: Path) -> Pins:
        pins = cls(path)
        if path.exists():
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    rxcui, setid = row["ingredient_rxcui"].strip(), row["setid"].strip().lower()
                    pins._names[rxcui] = row["name"]
                    pins._setids.setdefault(rxcui, []).append(setid)
        return pins

    def get(self, rxcui: str) -> list[str]:
        return list(self._setids.get(rxcui, []))

    def set(self, rxcui: str, name: str, setids: list[str]) -> None:
        self._names[rxcui] = name
        self._setids[rxcui] = [s.lower() for s in setids]

    def save(self) -> None:
        """Write the file, sorted by RxCUI, via a temp file so a crash cannot truncate it."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".part")
        with tmp.open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(FIELDS)
            for rxcui in sorted(self._setids, key=int):
                for setid in self._setids[rxcui]:
                    writer.writerow([rxcui, self._names[rxcui], setid])
        tmp.replace(self.path)
