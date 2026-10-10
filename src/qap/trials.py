"""Append-only trial log: every signal/model/spec evaluated counts toward the deflated-Sharpe trial count.

Re-running the same (name, spec) appends a record but counts once: `trial_count` is the number of
UNIQUE (name, spec) pairs, so reruns never inflate N.
"""
import json
import pathlib
import time


def log_trial(name: str, spec: dict, metrics: dict, path: str = "data/trials.jsonl") -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps({"ts": time.time(), "name": name, "spec": spec, "metrics": metrics}, default=str) + "\n")


def trial_count(path: str = "data/trials.jsonl") -> int:
    p = pathlib.Path(path)
    if not p.exists():
        return 0
    seen = set()
    for line in p.open():
        if line.strip():
            r = json.loads(line)
            seen.add((r["name"], json.dumps(r["spec"], sort_keys=True, default=str)))
    return len(seen)
