"""Settings in one place. Secrets come from the environment (.env), never from code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Pinned ATT&CK releases. Pinning makes every eval number reproducible: a new
# release adds/revokes techniques and would silently change the answer key.
ATTACK_VERSION = "20.1"
# Last release of the previous major version; diffed against ATTACK_VERSION to
# find "techniques added in the latest release" (Phase 5).
ATTACK_PREV_VERSION = "19.2"
ATTACK_URL = (
    "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/"
    "enterprise-attack/enterprise-attack-{version}.json"
)

RAW_DIR = ROOT / "data" / "raw"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "attack_techniques"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Exact model IDs are pinned so eval results name the model that produced them.
# gemini-3.1-flash-lite: cheapest Gemini Flash-family model open to new API
# keys on 2026-11-17 ($0.25 in / $1.50 out per 1M tokens). The cheaper
# gemini-2.5-flash-lite is closed to new users. Google lists a shutdown date
# of 2027-05-07 for 3.1, with gemini-3.5-flash-lite as the replacement.
GEMINI_MODEL = "gemini-3.1-flash-lite"
CLAUDE_MODEL = "claude-haiku-5-5"
