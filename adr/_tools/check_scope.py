#!/usr/bin/env python3
"""Check 1: Scope map coverage validation."""
import re, sys, os
from pathlib import Path

ROOT = Path("src")
scope_file = ROOT / "adr-scope.md"

if not scope_file.exists():
    print("FAIL: adr-scope.md does not exist")
    sys.exit(1)

# Parse scope map
excluded = set()
covered = set()
pending = []
rows = []

for line in scope_file.read_text().splitlines():
    m = re.match(r'\|\s*`([^`]+)`.*\|\s*(COVERED|EXCLUDED|PENDING)', line)
    if m:
        dirpath = m.group(1).strip('/')
        status = m.group(2)
        rows.append((dirpath, status))
        if status == 'EXCLUDED':
            excluded.add(dirpath)
        elif status == 'COVERED':
            covered.add(dirpath)
        elif status == 'PENDING':
            pending.append(dirpath)

all_dirs_in_scope = {r[0] for r in rows}

# Get all directories from the src repo
all_dirs = []
for d in sorted(ROOT.rglob('*')):
    if d.is_dir() and not any(part.startswith('.') for part in d.parts):
        rel = str(d.relative_to(ROOT))
        all_dirs.append(rel)

# Check depth-1 dirs
depth1 = [d for d in all_dirs if '/' not in d]
missing_depth1 = [d for d in depth1 if d not in all_dirs_in_scope]

# For deeper dirs, check implicit rules
def has_excluded_ancestor(d):
    parts = d.split('/')
    for i in range(1, len(parts)):
        ancestor = '/'.join(parts[:i])
        if ancestor in excluded:
            return True
    return False

def has_covered_ancestor(d):
    parts = d.split('/')
    for i in range(1, len(parts)):
        ancestor = '/'.join(parts[:i])
        if ancestor in covered:
            return True
    return False

missing_others = []
for d in all_dirs:
    if '/' not in d:
        continue  # depth-1 handled above
    if d in all_dirs_in_scope:
        continue
    if has_excluded_ancestor(d):
        continue
    if has_covered_ancestor(d):
        continue
    missing_others.append(d)

print(f"PENDING entries ({len(pending)}): {pending}")
print(f"Missing depth-1 dirs: {missing_depth1}")
print(f"Missing other dirs (not implicitly covered): {missing_others[:20]}")
print(f"COVERED entries: {sorted(covered)}")
print(f"EXCLUDED entries: {sorted(excluded)}")

# Check 1 result
fails = []
if pending:
    fails.append(f"PENDING entries: {', '.join(pending)}")
if missing_depth1:
    fails.append(f"Missing depth-1: {', '.join(missing_depth1)}")
if missing_others:
    fails.append(f"Missing dirs: {', '.join(missing_others[:10])}")

if fails:
    print(f"\nCheck 1: FAIL")
    for f in fails:
        print(f"  - {f}")
else:
    print("\nCheck 1: PASS")
