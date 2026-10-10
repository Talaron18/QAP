"""Lazy WRDS connection. Nothing connects at import time; credentials come from the environment."""
import os


def connect():
    """Open a WRDS connection using WRDS_USERNAME (password via ~/.pgpass or the library prompt)."""
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        raise RuntimeError("Set WRDS_USERNAME (and keep credentials out of git).")
    import wrds  # optional dependency: pip install -e .[wrds]
    return wrds.Connection(wrds_username=user)
