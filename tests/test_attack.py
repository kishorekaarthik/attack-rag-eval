from copilot.attack import load_techniques
from tests.bundles import attack_pattern, detection, write_bundle


def load(tmp_path, objects):
    return load_techniques(write_bundle(tmp_path / "b.json", objects))


def test_reads_id_name_tactics_from_attack_pattern(tmp_path):
    [t] = load(tmp_path, [attack_pattern("T1059.001", "PowerShell", tactics=("execution", "defense-evasion"))])
    assert (t.id, t.name) == ("T1059.001", "PowerShell")
    assert t.tactics == ["execution", "defense-evasion"]
    assert t.is_subtechnique is True
    assert t.created == "2020-01-01T00:00:00.000Z"


def test_drops_revoked_and_deprecated(tmp_path):
    ts = load(tmp_path, [
        attack_pattern("T1001", "Kept"),
        attack_pattern("T1002", "Revoked", revoked=True),
        attack_pattern("T1003", "Deprecated", x_mitre_deprecated=True),
    ])
    assert [t.id for t in ts] == ["T1001"]


def test_ignores_non_technique_objects(tmp_path):
    ts = load(tmp_path, [{"type": "malware", "id": "malware--1", "name": "x"}, attack_pattern("T1001", "Kept")])
    assert [t.id for t in ts] == ["T1001"]


def test_strips_citations_and_code_tags_from_description(tmp_path):
    [t] = load(tmp_path, [attack_pattern(
        "T1001", "X",
        description="Adversaries use <code>cmd.exe</code>.(Citation: Some Report 2020) See [PowerShell](https://attack.mitre.org/techniques/T1059/001).")])
    assert t.description == "Adversaries use cmd.exe. See PowerShell."


def test_detection_comes_from_linked_detection_strategy(tmp_path):
    objs = [attack_pattern("T1001", "X")] + detection(
        "T1001", "DET1", "Detect odd DNS", [("Linux", "Watch auditd.(Citation: A)"), ("Windows", "Watch Sysmon 22.")])
    [t] = load(tmp_path, objs)
    assert t.detection == "Detect odd DNS\n[Linux] Watch auditd.\n[Windows] Watch Sysmon 22."


def test_legacy_detection_field_is_used_when_present(tmp_path):
    [t] = load(tmp_path, [attack_pattern("T1001", "X", x_mitre_detection="Monitor process creation.")])
    assert t.detection == "Monitor process creation."


def test_detection_empty_when_nothing_linked(tmp_path):
    [t] = load(tmp_path, [attack_pattern("T1001", "X")])
    assert t.detection == ""


def test_techniques_sorted_by_id(tmp_path):
    ts = load(tmp_path, [attack_pattern("T1059.001", "B"), attack_pattern("T1003", "A")])
    assert [t.id for t in ts] == ["T1003", "T1059.001"]


def revoked_by(old, new):
    return {"type": "relationship", "id": f"relationship--{old}-{new}", "relationship_type": "revoked-by",
            "source_ref": f"attack-pattern--{old}", "target_ref": f"attack-pattern--{new}"}


def test_revoked_successors_maps_old_id_to_active_replacement(tmp_path):
    from copilot.attack import revoked_successors

    path = write_bundle(tmp_path / "b.json", [
        attack_pattern("T1070.001", "Old", revoked=True),
        attack_pattern("T1685.005", "New"),
        revoked_by("T1070.001", "T1685.005"),
    ])
    assert revoked_successors(path) == {"T1070.001": "T1685.005"}


def test_revoked_successors_follows_chains_to_the_final_active_id(tmp_path):
    from copilot.attack import revoked_successors

    path = write_bundle(tmp_path / "b.json", [
        attack_pattern("T1001", "A", revoked=True),
        attack_pattern("T1002", "B", revoked=True),
        attack_pattern("T1003", "C"),
        revoked_by("T1001", "T1002"),
        revoked_by("T1002", "T1003"),
    ])
    assert revoked_successors(path) == {"T1001": "T1003", "T1002": "T1003"}


def test_retired_ids_lists_revoked_and_deprecated_but_not_active(tmp_path):
    from copilot.attack import retired_ids

    path = write_bundle(tmp_path / "b.json", [
        attack_pattern("T1001", "Active"),
        attack_pattern("T1002", "Revoked", revoked=True),
        attack_pattern("T1003", "Deprecated", x_mitre_deprecated=True),
    ])
    assert retired_ids(path) == {"T1002": "revoked", "T1003": "deprecated"}


def uses(source_id, tid, description, **extra):
    rel = {"type": "relationship", "id": f"relationship--{source_id}-{tid}", "relationship_type": "uses",
           "source_ref": source_id, "target_ref": f"attack-pattern--{tid}", "description": description}
    rel.update(extra)
    return rel


def test_procedure_examples_come_from_uses_relationships(tmp_path):
    [t] = load(tmp_path, [
        attack_pattern("T1059.001", "PowerShell"),
        {"type": "malware", "id": "malware--1", "name": "Emotet"},
        {"type": "intrusion-set", "id": "intrusion-set--1", "name": "APT29"},
        uses("malware--1", "T1059.001", "[Emotet](https://attack.mitre.org/software/S0367) has used PowerShell to fetch payloads.(Citation: X)"),
        uses("intrusion-set--1", "T1059.001", "[APT29](https://attack.mitre.org/groups/G0016) ran encoded PowerShell commands."),
    ])
    assert t.procedures == ["APT29 ran encoded PowerShell commands.", "Emotet has used PowerShell to fetch payloads."]


def test_procedure_examples_skip_revoked_empty_and_non_technique_targets(tmp_path):
    [t] = load(tmp_path, [
        attack_pattern("T1059.001", "PowerShell"),
        {"type": "malware", "id": "malware--1", "name": "Emotet"},
        {"type": "malware", "id": "malware--2", "name": "Old", "revoked": True},
        uses("malware--1", "T1059.001", "Revoked relationship.", revoked=True),
        uses("malware--2", "T1059.001", "From revoked software."),
        uses("malware--1", "T1059.001", ""),
        {"type": "relationship", "id": "relationship--x", "relationship_type": "uses",
         "source_ref": "intrusion-set--1", "target_ref": "malware--1", "description": "Group uses software."},
    ])
    assert t.procedures == []
