from qap.trials import log_trial, trial_count


def test_reruns_do_not_inflate_count(tmp_path):
    p = str(tmp_path / "t.jsonl")
    for _ in range(3):
        log_trial("gpa", {"w": "VW"}, {"x": 1}, p)
    log_trial("gpa", {"w": "EW"}, {"x": 2}, p)
    log_trial("bm", {"w": "VW"}, {"x": 3}, p)
    assert trial_count(p) == 3
    assert trial_count(str(tmp_path / "none.jsonl")) == 0
