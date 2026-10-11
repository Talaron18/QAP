"""Regenerate src/qap/jkp_themes.py from the JKP Global Factor Data Documentation PDF (Table 9).

    pdftotext -layout Documentation.pdf doc.txt && python scripts/build_jkp_themes.py doc.txt
Notes: the PDF text layer loses underscores (names are rebuilt with "_"), and one citation ("Chordia, Subrahmanyam, and
Anshuman") wraps onto a second line that looks like a record; it is merged into dolvol_var_126d. The result must have 153
unique names in 13 themes; anything else aborts.
"""
import re
import sys

THEMES = ["Accruals", "Debt Issuance", "Investment", "Low Leverage", "Low Risk", "Momentum", "Profit Growth",
          "Profitability", "Quality", "Seasonality", "Size", "Short-Term Reversal", "Value"]
TAIL = re.compile(r"(?:^|\s)(-?1)\s+([01])\s*$")


def parse(path):
    lines = open(path).read().split("\n")
    start = next(i for i, l in enumerate(lines) if i > 2000 and "Factor Details and Citations" in l and l.strip().startswith("10"))
    end = next(i for i, l in enumerate(lines) if i > start and l.strip() == "Other Factors")
    theme, recs, cur = None, [], None
    for l in lines[start:end]:
        s = l.strip()
        if s in THEMES:
            theme, cur = s, None
            continue
        if theme is None or not s or s.startswith(("Table 9", "Variable", "Description")) or re.fullmatch(r"\d+", s):
            continue
        fields = [f for f in re.split(r"\s{2,}", s) if f]
        if len(fields) >= 2 and re.match(r"^[a-z][a-z0-9 ]*", fields[1]) and not fields[1].startswith(("and ", "et ")):
            name = re.match(r"^[a-z][a-z0-9 ]*", fields[1]).group(0).strip().replace(" ", "_")
            cur = {"theme": theme, "name": name, "sign": None}
            recs.append(cur)
        m = TAIL.search(l)
        if m and cur is not None and cur["sign"] is None:
            cur["sign"] = int(m.group(1))
    recs = [r for r in recs if r["name"] != "shuman"]
    for r in recs:
        if r["name"] == "dolvol_var_126d":
            r["sign"] = -1
    assert len(recs) == 153 and len({r["name"] for r in recs}) == 153 and all(r["sign"] in (-1, 1) for r in recs), "parse check failed"
    return recs


if __name__ == "__main__":
    recs = parse(sys.argv[1])
    print({t: sum(r["theme"] == t for r in recs) for t in THEMES})
