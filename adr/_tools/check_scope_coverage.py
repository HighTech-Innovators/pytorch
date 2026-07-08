#!/usr/bin/env python3
"""Check 1: Verify all directories are covered by adr-scope.md."""

import os
import re
import sys

SCOPE_FILE = "src/adr-scope.md"
SRC_DIR = "src"

# Parse adr-scope.md
covered = set()
excluded = set()

with open(SCOPE_FILE) as f:
    for line in f:
        m = re.match(r'\|\s*`([^`]+)`\s*\|\s*(COVERED|EXCLUDED)\s*\|', line)
        if m:
            path, status = m.group(1), m.group(2)
            if status == 'COVERED':
                covered.add(path)
            elif status == 'EXCLUDED':
                excluded.add(path)

print(f"COVERED entries: {len(covered)}")
print(f"EXCLUDED entries: {len(excluded)}")

# Get all directories under src/
all_dirs = []
for root, dirs, files in os.walk(SRC_DIR):
    # Skip hidden dirs
    dirs[:] = [d for d in dirs if not d.startswith('.')]
    for d in sorted(dirs):
        full = os.path.join(root, d)
        rel = os.path.relpath(full, SRC_DIR)
        all_dirs.append(rel)

all_dirs.sort()

def depth(path):
    return len(path.split(os.sep))

def is_excluded_ancestor(path):
    parts = path.split(os.sep)
    for i in range(1, len(parts)):
        ancestor = os.sep.join(parts[:i])
        if ancestor in excluded:
            return True
    return False

def is_covered_ancestor(path):
    parts = path.split(os.sep)
    for i in range(1, len(parts)):
        ancestor = os.sep.join(parts[:i])
        if ancestor in covered:
            return True
    return False

all_scope = covered | excluded
missing = []

for d in all_dirs:
    d_unix = d.replace(os.sep, '/')
    if depth(d) == 1:
        # Depth 1: must appear explicitly
        if d_unix not in all_scope:
            missing.append(f"DEPTH-1 MISSING: {d_unix}")
    else:
        if d_unix in all_scope:
            pass  # Explicitly classified
        elif is_excluded_ancestor(d_unix):
            pass  # Implicitly excluded
        elif is_covered_ancestor(d_unix):
            pass  # Implicitly covered
        else:
            missing.append(f"UNCOVERED: {d_unix}")

if missing:
    print(f"\nMISSING ({len(missing)}):")
    for m in missing:
        print(f"  {m}")
else:
    print("\nAll directories are classified.")
