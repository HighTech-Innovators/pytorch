#!/usr/bin/env python3
"""Check 1: Verify adr-scope.md covers all directories in src repo.

Run from the outer repo root (parent of src/).
"""
import re
import sys
import os
from pathlib import Path

SRC = Path("src")
SCOPE_FILE = SRC / "adr-scope.md"

def parse_scope(scope_file):
    excluded = set()
    covered = set()
    pending = []
    rows = []
    with open(scope_file) as f:
        for line in f:
            # Match table rows: | ./dir | ... | STATUS | ... |
            m = re.match(r'\|\s*(\./[^|]+?)\s*\|[^|]*\|\s*(EXCLUDED|COVERED|PENDING)\s*\|', line)
            if m:
                d = m.group(1).strip()
                status = m.group(2).strip()
                # normalize: strip leading ./
                norm = d.lstrip('.').lstrip('/')
                rows.append((norm, status))
                if status == 'EXCLUDED':
                    excluded.add(norm)
                elif status == 'COVERED':
                    covered.add(norm)
                elif status == 'PENDING':
                    pending.append(norm)
    return excluded, covered, pending, rows

def get_depth(path_str):
    # depth relative to src root — number of path components
    parts = [p for p in path_str.split('/') if p]
    return len(parts)

def has_excluded_ancestor(path_str, excluded):
    parts = path_str.split('/')
    for i in range(1, len(parts)):
        ancestor = '/'.join(parts[:i])
        if ancestor in excluded:
            return True
    return False

def has_covered_ancestor(path_str, covered):
    parts = path_str.split('/')
    for i in range(1, len(parts)):
        ancestor = '/'.join(parts[:i])
        if ancestor in covered:
            return True
    return False

def main():
    excluded, covered, pending, rows = parse_scope(SCOPE_FILE)
    
    all_scope_dirs = excluded | covered | set(d for d, s in rows)
    
    # Get all directories from src
    src_dirs = []
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for d in dirs:
            full = os.path.join(root, d)
            rel = os.path.relpath(full, SRC)
            rel = rel.replace('\\', '/')
            src_dirs.append(rel)
    src_dirs.sort()
    
    missing = []
    for d in src_dirs:
        depth = get_depth(d)
        if depth == 1:
            # Must appear explicitly in scope
            if d not in all_scope_dirs:
                missing.append((d, 'depth-1, missing from scope'))
        else:
            if d in all_scope_dirs:
                continue  # explicitly listed, OK
            if has_excluded_ancestor(d, excluded):
                continue  # implicitly excluded
            if has_covered_ancestor(d, covered):
                continue  # implicitly covered
            missing.append((d, 'not covered by explicit or implicit rule'))
    
    print("=== Check 1 Results ===")
    print(f"COVERED count: {len(covered)}")
    print(f"EXCLUDED count: {len(excluded)}")
    print(f"PENDING count: {len(pending)}")
    if pending:
        print("PENDING dirs:", pending)
    if missing:
        print(f"\nMISSING/UNCOVERED dirs ({len(missing)}):")
        for d, reason in missing:
            print(f"  {d}: {reason}")
    else:
        print("\nAll directories classified or implicitly covered.")
    
    status = "PASS"
    if pending:
        status = "FAIL (PENDING entries)"
    if missing:
        status = f"FAIL (missing dirs)"
    print(f"\nCheck 1: {status}")

if __name__ == '__main__':
    os.chdir('/home/github-runner/actions-runner/_work/CodeWeave-PyTorch/CodeWeave-PyTorch')
    main()
