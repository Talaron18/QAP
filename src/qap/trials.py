"""Append-only trial log: every signal/model/spec evaluated counts toward the deflated-Sharpe trial count."""
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
    return sum(1 for _ in p.open()) if p.exists() else 0
