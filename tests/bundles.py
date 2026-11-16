"""Hand-made miniature STIX bundles for tests. Shapes mirror ATT&CK v20."""
import json
from pathlib import Path


def attack_pattern(tid, name, description="Some description.", tactics=("execution",), **extra):
    obj = {
        "type": "attack-pattern",
        "id": f"attack-pattern--{tid}",
        "name": name,
        "description": description,
        "created": "2020-01-01T00:00:00.000Z",
        "external_references": [
            {"source_name": "other", "external_id": "X-1"},
            {"source_name": "mitre-attack", "external_id": tid, "url": f"https://attack.mitre.org/techniques/{tid}"},
        ],
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": t} for t in tactics],
        "x_mitre_is_subtechnique": "." in tid,
    }
    obj.update(extra)
    return obj


def detection(tid, det_id, strategy_name, analytics):
    """Detection strategy + analytics + the `detects` relationship, as in v18+.

    analytics: list of (platform, text)
    """
    objs, refs = [], []
    for i, (platform, text) in enumerate(analytics):
        aid = f"x-mitre-analytic--{det_id}-{i}"
        refs.append(aid)
        objs.append({"type": "x-mitre-analytic", "id": aid, "name": f"Analytic {i}",
                     "description": text, "x_mitre_platforms": [platform]})
    sid = f"x-mitre-detection-strategy--{det_id}"
    objs.append({"type": "x-mitre-detection-strategy", "id": sid, "name": strategy_name,
                 "x_mitre_analytic_refs": refs})
    objs.append({"type": "relationship", "id": f"relationship--{det_id}", "relationship_type": "detects",
                 "source_ref": sid, "target_ref": f"attack-pattern--{tid}"})
    return objs


def write_bundle(path: Path, objects) -> Path:
    path.write_text(json.dumps({"type": "bundle", "id": "bundle--test", "objects": objects}), encoding="utf-8")
    return path
