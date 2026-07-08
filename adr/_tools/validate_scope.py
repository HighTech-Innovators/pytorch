#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


ROW_RE = re.compile(
    r"^\|\s*`(?P<directory>[^`]+)`\s*\|\s*(?P<status>COVERED|EXCLUDED|PENDING)\s*\|\s*(?P<detail>.*?)\s*\|$"
)


def load_scope(scope_path: Path):
    rows = []
    for line_no, line in enumerate(scope_path.read_text().splitlines(), start=1):
        match = ROW_RE.match(line.strip())
        if match:
            row = match.groupdict()
            row["line"] = line_no
            rows.append(row)
    return rows


def ancestors(path: str):
    parts = path.split("/")
    for index in range(len(parts) - 1, 0, -1):
        yield "/".join(parts[:index])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    src_root = repo_root / "src"
    scope_path = src_root / "adr-scope.md"

    result = {
        "scope_exists": scope_path.exists(),
        "missing_depth1": [],
        "missing_other": [],
        "pending_rows": [],
        "explicit_rows": [],
    }

    if not scope_path.exists():
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    rows = load_scope(scope_path)
    by_dir = {row["directory"]: row for row in rows}
    covered = {row["directory"] for row in rows if row["status"] == "COVERED"}
    excluded = {row["directory"] for row in rows if row["status"] == "EXCLUDED"}
    pending = [row for row in rows if row["status"] == "PENDING"]

    result["pending_rows"] = pending
    result["explicit_rows"] = rows

    all_dirs = []
    for path in src_root.rglob("*"):
        if not path.is_dir():
            continue
        rel_parts = path.relative_to(src_root).parts
        if not rel_parts or rel_parts[0] == "adr":
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        all_dirs.append(path.relative_to(src_root).as_posix())
    all_dirs.sort()

    for rel_dir in all_dirs:
        if rel_dir == ".":
            continue
        depth = len(rel_dir.split("/"))
        explicit = by_dir.get(rel_dir)
        if depth == 1:
            if explicit is None:
                result["missing_depth1"].append(rel_dir)
            continue
        if explicit is not None:
            continue
        if any(ancestor in excluded for ancestor in ancestors(rel_dir)):
            continue
        if any(ancestor in covered for ancestor in ancestors(rel_dir)):
            continue
        result["missing_other"].append(rel_dir)

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
