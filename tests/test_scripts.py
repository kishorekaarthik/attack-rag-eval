import sys

from copilot import config

sys.path.insert(0, str(config.ROOT / "scripts"))


def summary(n):
    return {"n": n, "hits": 1, "hit_rate": 1 / n, "hit_rate_ci": (0.0, 1.0), "mrr": 0.5, "mrr_ci": (0.0, 1.0),
            "hits_parent": 1, "hit_rate_parent": 1 / n}


def test_eval_summary_prints_sets_it_has_no_label_for(capsys):
    from eval_retrieval import print_summary

    print_summary({"overall": summary(32), "recent": summary(32)})
    out = capsys.readouterr().out
    assert "recent" in out and "1/32" in out
