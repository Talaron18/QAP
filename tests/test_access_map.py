import importlib.util
import pathlib

SPEC = importlib.util.spec_from_file_location("wrds_access_map", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "wrds_access_map.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_option_like_pattern_hits_option_vol_index_borrow_tables_and_not_the_research_tables():
    hit = ["cboe cboe", "cboe ivlisted_2019", "optionm opprcd2019", "cboe ivborrowrate", "cboe_indexes vix", "cboe eqhvol", "x volatility_surface"]
    miss = ["crsp msf_v2", "comp funda", "contrib_general arc", "comp secm", "ff factors_daily", "crsp dsedist", "contrib_general liva"]
    assert all(mod.STRONG.search(s) for s in hit)
    assert not any(mod.STRONG.search(s) for s in miss)


def test_classify_separates_permission_denied_from_other_errors_and_success():
    import pandas as pd
    assert mod.classify(pd.DataFrame({"a": [1]})) == "OK"
    assert mod.classify("ProgrammingError: permission denied for schema cboe_eod") == "DENIED"
    assert mod.classify("OperationalError: statement timeout") == "ERROR"
