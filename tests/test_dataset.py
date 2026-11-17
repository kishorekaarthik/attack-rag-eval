import json

import pytest

from copilot.attack import Technique
from copilot.evals.dataset import (DatasetError, Question, assign_splits, leaks, load_questions, read_csv,
                                   save_questions, validate)


def tech(tid, name):
    return Technique(id=tid, name=name, description="d", tactics=["execution"], detection="",
                     is_subtechnique="." in tid, created="2020-01-01T00:00:00.000Z")


TECHS = [tech("T1059.001", "PowerShell"), tech("T1486", "Data Encrypted for Impact"), tech("T1053.002", "At")]


def q(i, tid="T1486", source="independent", text="Files were scrambled and a ransom note appeared.", split=None):
    return Question(id=f"{source[0]}{i:03d}", question=text, technique_id=tid, source=source, split=split)


# --- leak check -----------------------------------------------------------

def test_leak_when_question_contains_technique_name_any_case():
    assert leaks("They ran powershell scripts.", TECHS[0]) is True


def test_leak_when_question_contains_technique_id():
    assert leaks("Classic T1059.001 behaviour.", TECHS[0]) is True


def test_no_leak_for_a_paraphrase():
    assert leaks("A hidden console ran an encoded command.", TECHS[0]) is False


def test_name_must_match_as_whole_words():
    assert leaks("They used data encrypted for impactful effect.", TECHS[1]) is False


def test_very_short_names_are_not_treated_as_leaks():
    # "At" is an ordinary English word; flagging it would reject almost every sentence.
    assert leaks("At 9 am the job ran.", TECHS[2]) is False


# --- validation -----------------------------------------------------------

def test_valid_set_passes():
    validate([q(1), q(2, "T1059.001", text="A hidden console ran an encoded command.")], TECHS, retired={}, successors={})


def test_revoked_answer_key_fails_and_names_the_successor():
    with pytest.raises(DatasetError, match=r"i001.*T1070\.001.*revoked.*T1685\.005"):
        validate([q(1, "T1070.001")], TECHS, retired={"T1070.001": "revoked"}, successors={"T1070.001": "T1685.005"})


def test_deprecated_answer_key_fails():
    with pytest.raises(DatasetError, match=r"T1002.*deprecated"):
        validate([q(1, "T1002")], TECHS, retired={"T1002": "deprecated"}, successors={})


def test_unknown_answer_key_fails():
    with pytest.raises(DatasetError, match=r"T9999.*not in ATT&CK"):
        validate([q(1, "T9999")], TECHS, retired={}, successors={})


def test_leaking_question_fails():
    with pytest.raises(DatasetError, match="leaks"):
        validate([q(1, "T1059.001", text="PowerShell was used.")], TECHS, retired={}, successors={})


def test_duplicate_question_ids_fail():
    with pytest.raises(DatasetError, match="duplicate"):
        validate([q(1), q(1)], TECHS, retired={}, successors={})


def test_all_problems_are_reported_together():
    with pytest.raises(DatasetError) as err:
        validate([q(1, "T9999"), q(2, "T1002")], TECHS, retired={"T1002": "deprecated"}, successors={})
    assert "T9999" in str(err.value) and "T1002" in str(err.value)


# --- splits ---------------------------------------------------------------

def hundred():
    return [q(i, source="independent") for i in range(25)] + [q(i, source="generated") for i in range(75)]


def test_split_is_70_dev_30_test():
    out = assign_splits(hundred())
    assert sum(x.split == "dev" for x in out) == 70
    assert sum(x.split == "test" for x in out) == 30


def test_split_is_stratified_by_source():
    out = assign_splits(hundred())
    independent_test = sum(x.split == "test" for x in out if x.source == "independent")
    generated_test = sum(x.split == "test" for x in out if x.source == "generated")
    assert independent_test in (7, 8) and generated_test in (22, 23)


def test_split_is_stable_for_a_seed_and_changes_with_it():
    a = [x.split for x in assign_splits(hundred(), seed=13)]
    assert a == [x.split for x in assign_splits(hundred(), seed=13)]
    assert a != [x.split for x in assign_splits(hundred(), seed=14)]


def test_split_does_not_depend_on_input_order():
    forward = {x.id: x.split for x in assign_splits(hundred())}
    backward = {x.id: x.split for x in assign_splits(list(reversed(hundred())))}
    assert forward == backward


# --- files ----------------------------------------------------------------

def test_read_csv_assigns_ids_and_source_and_keeps_note(tmp_path):
    path = tmp_path / "o.csv"
    path.write_text('question,technique_id,note\n"Logs were wiped, then nothing.",T1685.005,originally T1070.001\n"Second.",T1486,\n',
                    encoding="utf-8")
    first, second = read_csv(path, source="independent")
    assert (first.id, first.technique_id, first.source, first.note) == ("i001", "T1685.005", "independent", "originally T1070.001")
    assert (second.id, second.note, second.split) == ("i002", "", None)


def test_save_then_load_round_trips_and_filters(tmp_path):
    path = tmp_path / "q.jsonl"
    qs = assign_splits(hundred())
    save_questions(qs, path)
    assert load_questions(path) == sorted(qs, key=lambda x: x.id)
    assert len(load_questions(path, split="test")) == 30
    assert len(load_questions(path, source="independent")) == 25
    assert {x.source for x in load_questions(path, split="dev", source="generated")} == {"generated"}
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["id"]
