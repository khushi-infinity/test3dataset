#!/usr/bin/env python3
"""
Starter Kit Schema Inspector & Adapter
Run this inside your cloned Secret-Wards-ToolKit directory to:
  1. Inspect the exact JSON schema of official dev/truth.json and dev/attacker*.json.
  2. Emit a schema diff report so you can map dev_unseen_domain/truth.json 1:1 to check_dev.py.

Usage:
  python3 tools/sync_schema_with_kit.py --kit-dev /path/to/Secret-Wards-ToolKit/dev
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path


def summarize_schema(obj, depth: int = 0) -> str:
    pad = "  " * depth
    if isinstance(obj, dict):
        lines = [f"{pad}dict ({len(obj)} keys):"]
        for k, v in list(obj.items())[:15]:
            if isinstance(v, (dict, list)) and depth < 2:
                lines.append(f"{pad}  {k!r}:")
                lines.append(summarize_schema(v, depth + 2))
            else:
                lines.append(f"{pad}  {k!r}: {type(v).__name__} = {str(v)[:100]}")
        return "\n".join(lines)
    if isinstance(obj, list):
        head = summarize_schema(obj[0], depth + 1) if obj else f"{pad}  <empty>"
        return f"{pad}list (len={len(obj)}), sample[0]:\n{head}"
    return f"{pad}{type(obj).__name__}: {str(obj)[:100]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-dev", required=True, type=Path, help="Path to Secret-Wards-ToolKit/dev folder")
    args = ap.parse_args()

    print("=== OFFICIAL STARTER KIT dev/ FILE TREE ===")
    for p in sorted(args.kit_dev.rglob("*")):
        print(" ", p.relative_to(args.kit_dev).as_posix())

    for jf in sorted(args.kit_dev.glob("*.json")):
        print(f"\n=== SCHEMA OF {jf.name} ===")
        data = json.loads(jf.read_text(encoding="utf-8"))
        print(summarize_schema(data))


if __name__ == "__main__":
    main()
