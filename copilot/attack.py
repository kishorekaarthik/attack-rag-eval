"""Download a pinned MITRE ATT&CK enterprise bundle and parse techniques out of it.

Where detection text lives (checked against v20.1): since ATT&CK v18 the
`x_mitre_detection` field on techniques is empty. Detection guidance moved to
`x-mitre-detection-strategy` objects, linked to a technique by a `detects`
relationship; each strategy lists `x-mitre-analytic` objects whose
`description` holds the per-platform detection text. We read the new
structure and fall back to the legacy field so older bundles still parse.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from copilot import config


@dataclass(frozen=True)
class Technique:
    id: str  # "T1059" or "T1059.001"
    name: str
    description: str
    tactics: list[str]
    detection: str  # "" if none
    is_subtechnique: bool
    created: str


def bundle_path(version: str = config.ATTACK_VERSION) -> Path:
    """Path to the bundle for `version`, downloading it once if absent."""
    path = config.RAW_DIR / f"enterprise-attack-{version}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(config.ATTACK_URL.format(version=version), timeout=300)
        resp.raise_for_status()
        path.write_bytes(resp.content)
    return path


_CITATION = re.compile(r"\(Citation: [^)]*\)")
_TAG = re.compile(r"</?code>")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://)[^)]*\)")


def clean(text: str) -> str:
    """Strip markup that adds tokens but no meaning for retrieval."""
    text = _CITATION.sub("", text)
    text = _TAG.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    return text.strip()


def _is_active(obj: dict) -> bool:
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


def _attack_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def _detections(objects: list[dict]) -> dict[str, str]:
    """Map attack-pattern STIX id -> detection text built from strategies and analytics."""
    by_id = {o["id"]: o for o in objects}
    out: dict[str, list[str]] = {}
    for rel in objects:
        if rel.get("type") != "relationship" or rel.get("relationship_type") != "detects" or not _is_active(rel):
            continue
        strategy = by_id.get(rel["source_ref"])
        if not strategy or not _is_active(strategy):
            continue
        lines = [strategy.get("name", "").strip()]
        for ref in strategy.get("x_mitre_analytic_refs", []):
            analytic = by_id.get(ref)
            if not analytic or not _is_active(analytic):
                continue
            text = clean(analytic.get("description", ""))
            if text:
                platforms = "/".join(analytic.get("x_mitre_platforms", []))
                lines.append(f"[{platforms}] {text}" if platforms else text)
        out.setdefault(rel["target_ref"], []).append("\n".join(line for line in lines if line))
    return {k: "\n\n".join(v) for k, v in out.items()}


def load_techniques(path: Path) -> list[Technique]:
    """Active (not revoked, not deprecated) techniques and sub-techniques, sorted by ID."""
    objects = json.loads(Path(path).read_text(encoding="utf-8"))["objects"]
    detections = _detections(objects)
    techniques = []
    for obj in objects:
        if obj.get("type") != "attack-pattern" or not _is_active(obj):
            continue
        tid = _attack_id(obj)
        if not tid:
            continue
        techniques.append(Technique(
            id=tid,
            name=obj["name"],
            description=clean(obj.get("description", "")),
            tactics=[p["phase_name"] for p in obj.get("kill_chain_phases", [])
                     if p.get("kill_chain_name") == "mitre-attack"],
            detection=detections.get(obj["id"]) or clean(obj.get("x_mitre_detection", "")),
            is_subtechnique=bool(obj.get("x_mitre_is_subtechnique")),
            created=obj.get("created", ""),
        ))
    return sorted(techniques, key=lambda t: t.id)


def _attack_patterns(path: Path) -> tuple[list[dict], list[dict]]:
    objects = json.loads(Path(path).read_text(encoding="utf-8"))["objects"]
    return [o for o in objects if o.get("type") == "attack-pattern"], objects


def retired_ids(path: Path) -> dict[str, str]:
    """Technique IDs that exist in the bundle but must not be used: id -> "revoked" | "deprecated"."""
    patterns, _ = _attack_patterns(path)
    out = {}
    for obj in patterns:
        tid = _attack_id(obj)
        if tid and not _is_active(obj):
            out[tid] = "revoked" if obj.get("revoked") else "deprecated"
    return out


def revoked_successors(path: Path) -> dict[str, str]:
    """Revoked technique ID -> the active ID that replaced it, following `revoked-by` chains."""
    patterns, objects = _attack_patterns(path)
    by_stix = {o["id"]: o for o in patterns}
    nxt = {r["source_ref"]: r["target_ref"] for r in objects
           if r.get("type") == "relationship" and r.get("relationship_type") == "revoked-by"}
    out = {}
    for stix_id, obj in by_stix.items():
        if stix_id not in nxt:
            continue
        cur, seen = stix_id, set()
        while cur in nxt and cur not in seen:  # `seen` guards against a cycle in bad data
            seen.add(cur)
            cur = nxt[cur]
        target = by_stix.get(cur)
        if target and _is_active(target) and _attack_id(obj) and _attack_id(target):
            out[_attack_id(obj)] = _attack_id(target)
    return out
