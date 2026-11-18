import pytest

from copilot.evals.metrics import (bootstrap_ci, hit_at_k, ranked_technique_ids, reciprocal_rank, score_question,
                                   summarise)
from copilot.index import Hit


def hits(*technique_ids):
    return [Hit(chunk_id=f"{t}:description:{i}", technique_id=t, name=t, section="description", text="", score=1.0 - i / 100)
            for i, t in enumerate(technique_ids)]


def test_ranked_ids_dedupes_keeping_first_position():
    assert ranked_technique_ids(hits("T1003.001", "T1003.001", "T1059", "T1003.001", "T1486")) == ["T1003.001", "T1059", "T1486"]


def test_hit_at_k_counts_unique_techniques_not_chunks():
    # T1486 is the 6th chunk but the 2nd distinct technique.
    retrieved = hits("T1001", "T1001", "T1001", "T1001", "T1001", "T1486")
    assert hit_at_k(retrieved, "T1486", k=5) is True


def test_hit_at_k_is_false_beyond_the_kth_unique_technique():
    retrieved = hits("T1001", "T1002", "T1003", "T1004", "T1005", "T1486")
    assert hit_at_k(retrieved, "T1486", k=5) is False
    assert hit_at_k(retrieved, "T1486", k=6) is True


def test_reciprocal_rank_uses_deduplicated_technique_rank():
    assert reciprocal_rank(hits("T1059", "T1059", "T1059", "T1486"), "T1486") == 0.5


def test_reciprocal_rank_is_zero_when_absent():
    assert reciprocal_rank(hits("T1059", "T1003"), "T1486") == 0.0


def test_reciprocal_rank_is_zero_beyond_the_kth_unique_technique():
    retrieved = hits("T1001", "T1002", "T1003", "T1004", "T1005", "T1486")
    assert reciprocal_rank(retrieved, "T1486", k=5) == 0.0
    assert reciprocal_rank(retrieved, "T1486", k=6) == 1 / 6


def test_reciprocal_rank_is_one_at_top():
    assert reciprocal_rank(hits("T1486", "T1003"), "T1486") == 1.0


def test_exact_scoring_treats_parent_and_subtechnique_as_different():
    assert hit_at_k(hits("T1059.001"), "T1059", k=5) is False
    assert hit_at_k(hits("T1059"), "T1059.001", k=5) is False
    assert reciprocal_rank(hits("T1059.001"), "T1059") == 0.0


def test_parent_level_scoring_matches_within_the_same_family():
    assert hit_at_k(hits("T1059.001"), "T1059", k=5, parent_level=True) is True
    assert hit_at_k(hits("T1059"), "T1059.001", k=5, parent_level=True) is True
    assert hit_at_k(hits("T1059.003"), "T1059.001", k=5, parent_level=True) is True
    assert hit_at_k(hits("T1058.001"), "T1059.001", k=5, parent_level=True) is False


def test_parent_level_rank_dedupes_by_family():
    assert reciprocal_rank(hits("T1003.001", "T1003.002", "T1059.003"), "T1059.001", parent_level=True) == 0.5


def test_score_question_reports_all_four_numbers():
    assert score_question(hits("T1003", "T1059.003", "T1059.001"), "T1059.001", k=5) == {
        "hit": True, "rr": 1 / 3, "hit_parent": True, "rr_parent": 0.5,
    }


def test_bootstrap_ci_brackets_the_mean_and_is_repeatable():
    values = [1.0] * 40 + [0.0] * 60
    lo, hi = bootstrap_ci(values)
    assert lo < 0.4 < hi
    assert 0.28 < lo < 0.33 and 0.47 < hi < 0.52  # ~ +/- 1.96 * sqrt(.4*.6/100)
    assert bootstrap_ci(values) == (lo, hi)


def test_bootstrap_ci_collapses_when_every_value_is_the_same():
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_summarise_reports_raw_counts_rates_and_intervals():
    rows = [{"hit": True, "rr": 1.0, "hit_parent": True, "rr_parent": 1.0}] * 3 + \
           [{"hit": False, "rr": 0.0, "hit_parent": True, "rr_parent": 0.5}]
    s = summarise(rows)
    assert (s["n"], s["hits"], s["hits_parent"]) == (4, 3, 4)
    assert s["hit_rate"] == 0.75 and s["mrr"] == 0.75
    assert s["hit_rate_parent"] == 1.0 and s["mrr_parent"] == 0.875
    assert s["hit_rate_ci"][0] <= 0.75 <= s["hit_rate_ci"][1]
    assert s["mrr_ci"][0] <= 0.75 <= s["mrr_ci"][1]


def test_summarise_of_nothing_is_just_a_zero_count():
    assert summarise([]) == {"n": 0}


# --- paired comparison ---------------------------------------------------------------------

def test_mcnemar_exact_matches_the_binomial_tail():
    from copilot.evals.metrics import mcnemar_exact

    # 10 discordant pairs split 8/2: two-sided p = 2 * P(X <= 2), X ~ Bin(10, 0.5) = 2 * 56/1024
    assert mcnemar_exact(8, 2) == pytest.approx(112 / 1024)
    assert mcnemar_exact(2, 8) == mcnemar_exact(8, 2)


def test_mcnemar_is_one_when_there_is_no_disagreement_or_an_even_split():
    from copilot.evals.metrics import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0


def test_paired_comparison_counts_fixed_and_broke_and_the_difference():
    from copilot.evals.metrics import paired_comparison

    before = [True] * 50 + [False] * 50
    after = [True] * 45 + [False] * 5 + [True] * 20 + [False] * 30   # broke 5, fixed 20
    r = paired_comparison(before, after)
    assert (r["n"], r["fixed"], r["broke"]) == (100, 20, 5)
    assert (r["hits_before"], r["hits_after"]) == (50, 65)
    assert r["diff"] == pytest.approx(0.15)
    assert r["diff_ci"][0] < 0.15 < r["diff_ci"][1]
    assert r["diff_ci"][0] > 0          # 20 vs 5 discordant is a clear improvement
    assert r["p_value"] < 0.01


def test_paired_ci_comes_only_from_discordant_questions():
    from copilot.evals.metrics import paired_comparison

    same = [True, False] * 50
    r = paired_comparison(same, same)
    assert (r["fixed"], r["broke"], r["diff"], r["diff_ci"], r["p_value"]) == (0, 0, 0.0, (0.0, 0.0), 1.0)


def test_paired_comparison_rejects_unpaired_inputs():
    from copilot.evals.metrics import paired_comparison

    with pytest.raises(ValueError, match="same questions"):
        paired_comparison([True, False], [True])
