"""Step 1 of extraction: record the real WRDS schema so loaders are not written from guesses.

Run where your WRDS credentials live:   WRDS_USERNAME=you python scripts/wrds_inspect.py
(takes a few minutes). Writes to git-ignored data/:
  wrds_schema.txt       every table and its columns in the relevant libraries
  wrds_meta_<lib>_<table>.csv   rows of CRSP meta* tables (SIZ->CIZ mapping, delisting flag codes)
Paste wrds_schema.txt back; it holds schema and metadata, not market data.
"""
import pathlib

from qap.wrds_client import connect

# Substring match on library names (crsp*, comp*, ibes*, ff*, totalq, iid, beta suite, links, 13F)
LIB_HINTS = ("crsp", "comp", "ibes", "ff", "totalq", "iid", "beta", "wrdsapps", "factors", "tr_13f", "tfn", "snap")
META_ROW_LIMIT = 5000


def main() -> None:
    db = connect()
    out = pathlib.Path("data")
    out.mkdir(exist_ok=True)
    lines = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        lines.append(msg)

    libs = sorted(l for l in db.list_libraries() if any(h in l.lower() for h in LIB_HINTS))
    log(f"# libraries: {', '.join(libs)}")
    for lib in libs:
        try:
            tables = sorted(db.list_tables(library=lib))
        except Exception as e:  # permission errors must not stop the run
            log(f"[{lib}] list_tables failed: {e!r}")
            continue
        log(f"## {lib}: {len(tables)} tables")
        for t in tables:
            try:
                d = db.describe_table(library=lib, table=t)
                log(f"{lib}.{t}: " + ", ".join(f"{r.name}:{r.type}" for r in d.itertuples()))
            except Exception as e:
                log(f"{lib}.{t}: describe failed: {e!r}")
            if lib.lower().startswith("crsp") and t.lower().startswith("meta"):
                try:
                    rows = db.raw_sql(f"select * from {lib}.{t} limit {META_ROW_LIMIT}")
                    rows.to_csv(out / f"wrds_meta_{lib}_{t}.csv", index=False)
                    log(f"  -> dumped {len(rows)} rows of {lib}.{t}")
                except Exception as e:
                    log(f"  -> meta dump failed: {e!r}")
    (out / "wrds_schema.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
