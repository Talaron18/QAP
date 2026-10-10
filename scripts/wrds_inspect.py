"""Step 1 of extraction: list which candidate tables exist and their columns.

Run where your WRDS credentials live:  WRDS_USERNAME=you python scripts/wrds_inspect.py
Writes data/wrds_schema.txt (git-ignored). Paste it back so table/column config can be
filled in from the real schema instead of guesses.
"""
import pathlib

from qap.config import CANDIDATE_TABLES
from qap.wrds_client import connect


def main() -> None:
    db = connect()
    lines = []
    for key, candidates in CANDIDATE_TABLES.items():
        for full in candidates:
            schema, table = full.split(".")
            cols = db.raw_sql(
                "select column_name, data_type from information_schema.columns "
                "where table_schema = %(s)s and table_name = %(t)s order by ordinal_position",
                params={"s": schema, "t": table},
            )
            if cols.empty:
                lines.append(f"[{key}] {full}: NOT FOUND / no access")
            else:
                lines.append(f"[{key}] {full}: " + ", ".join(f"{r.column_name}:{r.data_type}" for r in cols.itertuples()))
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True)
    (out / "wrds_schema.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
