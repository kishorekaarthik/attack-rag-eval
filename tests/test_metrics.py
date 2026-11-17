from copilot.evals.metrics import hit_at_k, ranked_technique_ids, reciprocal_rank, score_question
from copilot.index import Hit


def hits(*technique_ids):
    return [Hit(chunk_id=f"{t}:description:{i}", technique_id=t, name=t, section="description", text="", score=1.0 - i / 100)
            for i, t in enumerate(technique_ids)]


def test_ranked_ids_dedupes_keeping_first_position():
    assert ranked_technique_ids(hits("T1003.001", "T1003.001", "T1059", "T1003.001", "T1486")) == ["T1003.001", "T1059", "T1486"]


def test_hit_at_k_looks_only_at_first_k_chunks():
    retrieved = hits("T1", "T1", "T1", "T1", "T1", "T1486")
    assert hit_at_k(retrieved, "T1486", k=5) is False
    assert hit_at_k(retrieved, "T1486", k=6) is True


def test_reciprocal_rank_uses_deduplicated_technique_rank():
    # T1486 is the 4th chunk but the 2nd distinct technique.
    assert reciprocal_rank(hits("T1059", "T1059", "T1059", "T1486"), "T1486") == 0.5


def test_reciprocal_rank_is_zero_when_absent():
    assert reciprocal_rank(hits("T1059", "T1003"), "T1486") == 0.0


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
    # Families in order: T1003, T1059 -> gold family T1059 is rank 2.
    assert reciprocal_rank(hits("T1003.001", "T1003.002", "T1059.003"), "T1059.001", parent_level=True) == 0.5


def test_score_question_reports_all_four_numbers():
    assert score_question(hits("T1003", "T1059.003", "T1059.001"), "T1059.001", k=5) == {
        "hit": True, "rr": 1 / 3, "hit_parent": True, "rr_parent": 0.5,
    }
