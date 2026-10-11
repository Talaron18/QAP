import re

from qap.jkp_themes import ALL_CHARACTERISTICS, THEMES


def test_153_unique_characteristics_in_13_themes():
    assert len(THEMES) == 13 and len(ALL_CHARACTERISTICS) == 153 and len(set(ALL_CHARACTERISTICS)) == 153
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", n) for n in ALL_CHARACTERISTICS)
    assert all(s in (-1, 1) for m in THEMES.values() for _, s in m)


def test_known_members_and_signs():
    d = {n: (t, s) for t, m in THEMES.items() for n, s in m}
    assert d["gp_at"] == ("Quality", 1) and d["at_gr1"] == ("Investment", -1) and d["be_me"] == ("Value", 1)
    assert d["ret_12_1"] == ("Momentum", 1) and d["ret_1_0"] == ("Short-Term Reversal", -1)
    assert d["beta_60m"] == ("Low Risk", -1) and d["market_equity"] == ("Size", -1) and d["chcsho_12m"] == ("Value", -1)
    assert d["f_score"] == ("Profitability", 1) and d["oaccruals_at"] == ("Accruals", -1)
    assert d["dolvol_var_126d"] == ("Profitability", -1)
