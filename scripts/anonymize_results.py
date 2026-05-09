"""Redact deanonymizing strings from result JSONs prior to public release.

Walks results/ and rewrites JSON metadata to remove user, hostname, absolute
path, and W&B entity identifiers. Operates in place; idempotent.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REDACTIONS = [
    (re.compile(r"/gpfs/home\d+/scur\d+"), "/redacted/home"),
    (re.compile(r"/gpfs/home\d+"), "/redacted/home"),
    (re.compile(r"gcn\d+\.local\.snellius\.surf\.nl"), "redacted-host"),
    (re.compile(r"snellius\.surf\.nl"), "redacted-host"),
    (re.compile(r"scur\d+"), "anon-user"),
    (re.compile(r"foundationmodels"), "anon-entity"),
]


def redact_string(s: str) -> str:
    for pat, repl in REDACTIONS:
        s = pat.sub(repl, s)
    return s


def redact(value):
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def main(root: Path) -> int:
    changed = 0
    scanned = 0
    for path in sorted(root.rglob("*.json")):
        scanned += 1
        original = path.read_text(encoding="utf-8")
        try:
            data = json.loads(original)
        except json.JSONDecodeError:
            print(f"skip (not JSON): {path}", file=sys.stderr)
            continue
        scrubbed = redact(data)
        rewritten = json.dumps(scrubbed, indent=2, ensure_ascii=False)
        if rewritten.strip() != original.strip():
            path.write_text(rewritten + "\n", encoding="utf-8")
            changed += 1
    print(f"scanned={scanned} changed={changed}")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
    raise SystemExit(main(target))
