#!/usr/bin/env python3
"""Check 1: Scope map coverage checker."""
import os, re, sys

SRC = "/home/github-runner/actions-runner/_work/CodeWeave-PyTorch/CodeWeave-PyTorch/src"
SCOPE_FILE = os.path.join(SRC, "adr-scope.md")

# Parse adr-scope.md
covered = set()
excluded = set()
with open(SCOPE_FILE) as f:
    for line in f:
        m = re.search(r'`([^`]+)`.*\|\s*(COVERED|EXCLUDED)', line)
        if m:
            d, status = m.group(1), m.group(2)
            if status == "COVERED":
                covered.add(d)
            else:
                excluded.add(d)

all_in_scope = covered | excluded

# Get all directories under src
all_dirs = []
for root, dirs, files in os.walk(SRC):
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for d in dirs:
        full = os.path.join(root, d)
        rel = os.path.relpath(full, SRC)
        all_dirs.append(rel)

all_dirs.sort()

missing = []
for d in all_dirs:
    parts = d.split(os.sep)
    depth = len(parts)
    
    if depth == 1:
        # Must be explicitly in scope
        if d not in all_in_scope:
            missing.append((d, "depth-1 not in adr-scope.md"))
        continue
    
    # Check if any ancestor is EXCLUDED
    anc_excluded = False
    for i in range(1, depth):
        anc = os.path.join(*parts[:i])
        if anc in excluded:
            anc_excluded = True
            break
    if anc_excluded:
        continue
    
    # Check if any ancestor is COVERED
    anc_covered = False
    for i in range(1, depth):
        anc = os.path.join(*parts[:i])
        if anc in covered:
            anc_covered = True
            break
    if anc_covered:
        continue
    
    # Must be explicitly in scope
    if d not in all_in_scope:
        missing.append((d, "not covered by ancestor or explicit entry"))

print(f"COVERED entries: {sorted(covered)}")
print(f"EXCLUDED entries: {sorted(excluded)}")
print(f"\nTotal dirs checked: {len(all_dirs)}")
if missing:
    print(f"\nMISSING ({len(missing)}):")
    for m, reason in missing:
        print(f"  {m}: {reason}")
else:
    print("\nAll directories covered (explicitly or by ancestor rule)")
