#!/usr/bin/env python3
"""Check 1: Scope map completeness."""
import re, subprocess, sys
from pathlib import Path

REPO = Path("/home/github-runner/actions-runner/_work/CodeWeave-PyTorch/CodeWeave-PyTorch/src")
SCOPE_FILE = REPO / "adr-scope.md"

scope_text = SCOPE_FILE.read_text()

covered = set()
excluded = set()
all_scope_dirs = set()

for line in scope_text.splitlines():
    if not line.startswith("|"):
        continue
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 4:
        continue
    d = parts[1].strip()
    status = parts[3].strip()
    if not d.startswith("./") or d == "./":
        continue
    d = d.lstrip("./")
    all_scope_dirs.add(d)
    if status == "COVERED":
        covered.add(d)
    elif status == "EXCLUDED":
        excluded.add(d)

# Get all directories
result = subprocess.run(
    ["find", ".", "-type", "d", "-not", "-path", "*/.*"],
    cwd=REPO, capture_output=True, text=True
)
dirs = sorted(r.lstrip("./") for r in result.stdout.strip().splitlines() if r != ".")
dirs = [d for d in dirs if d]

missing = []
for d in dirs:
    parts = d.split("/")
    depth = len(parts)
    if depth == 1:
        if d not in all_scope_dirs:
            missing.append((d, "depth-1 missing"))
        continue
    # Check if any ancestor is EXCLUDED
    excluded_ancestor = any("/".join(parts[:i]) in excluded for i in range(1, depth))
    if excluded_ancestor:
        continue
    # Check if any ancestor is COVERED
    covered_ancestor = any("/".join(parts[:i]) in covered for i in range(1, depth))
    if covered_ancestor:
        continue
    # Must appear explicitly
    if d not in all_scope_dirs:
        missing.append((d, "not covered by scope or ancestor rule"))

print(f"Total dirs scanned: {len(dirs)}")
print(f"COVERED entries: {len(covered)}")
print(f"EXCLUDED entries: {len(excluded)}")
print(f"Missing from scope: {len(missing)}")
if missing:
    for d, reason in missing[:30]:
        print(f"  MISSING: ./{d} ({reason})")
else:
    print("  All directories covered by scope rules.")
