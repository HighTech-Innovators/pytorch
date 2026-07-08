#!/usr/bin/env python3
"""ADR validation script for CodeWeave-PyTorch.
Runs all 5 checks from work/2-validate-adrs.md.
Run from the outer repo root (parent of src/).
"""

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # outer repo root
SRC = REPO_ROOT / "src"
BOOK = REPO_ROOT / "book"
ADR_SCOPE = SRC / "adr-scope.md"

VALID_EXCLUSION_REASONS = {
    "Auto-generated code",
    "Build/config only",
    "Vendored/third-party",
    "Test data only",
    "Test suite",
    "Empty or stub",
    "Leaf with no architectural boundary",
}

LEAF_EXCLUSION = "Leaf with no architectural boundary"
EMPTY_EXCLUSION = "Empty or stub"
BUILD_EXCLUSION = "Build/config only"

issues = []

def fail(check, msg):
    issues.append((check, msg))

# ── Parse adr-scope.md ──────────────────────────────────────────────────────

def parse_scope(path):
    covered = {}   # dir -> notes
    excluded = {}  # dir -> reason
    pending = []
    with open(path) as f:
        text = f.read()
    # Parse table rows  | `dir` | STATUS | ... |
    for line in text.splitlines():
        m = re.match(r'\|\s*`([^`]+)`\s*\|\s*(COVERED|EXCLUDED|PENDING)\s*\|(.*)$', line)
        if not m:
            continue
        d, status, rest = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        if status == "COVERED":
            covered[d] = rest
        elif status == "EXCLUDED":
            # extract reason from rest (second column in EXCLUDED table)
            cols = [c.strip() for c in rest.split('|') if c.strip()]
            reason = cols[0] if cols else ""
            excluded[d] = reason
        elif status == "PENDING":
            pending.append(d)
    return covered, excluded, pending

covered, excluded, pending = parse_scope(ADR_SCOPE)
all_scope_dirs = set(covered.keys()) | set(excluded.keys())

print(f"Parsed adr-scope.md: {len(covered)} COVERED, {len(excluded)} EXCLUDED, {len(pending)} PENDING")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 1 — Scope map current
# ═══════════════════════════════════════════════════════════════════════════
print("\n── CHECK 1: Scope map current ──")

# FAIL if PENDING entries
if pending:
    for d in pending:
        fail(1, f"PENDING entry: {d}")
    print(f"  FAIL: {len(pending)} PENDING entries")
else:
    print("  No PENDING entries")

# Get all directories under src (excluding hidden)
actual_dirs = set()
for dirpath, dirnames, _ in os.walk(SRC):
    dirnames[:] = [d for d in dirnames if not d.startswith('.')]
    rel = Path(dirpath).relative_to(SRC)
    if str(rel) != '.':
        actual_dirs.add(str(rel))

# Apply coverage rules
missing = []
for d in sorted(actual_dirs):
    parts = Path(d).parts
    depth = len(parts)
    
    # Is it explicitly in scope?
    if d in all_scope_dirs:
        continue
    
    # Depth-1: must be explicit, no exemptions
    if depth == 1:
        missing.append((d, "depth-1, not in adr-scope.md"))
        continue
    
    # Check EXCLUDED ancestor
    excluded_ancestor = False
    for i in range(1, depth):
        ancestor = str(Path(*parts[:i]))
        if ancestor in excluded:
            excluded_ancestor = True
            break
    if excluded_ancestor:
        continue
    
    # Check COVERED ancestor
    covered_ancestor = False
    for i in range(1, depth):
        ancestor = str(Path(*parts[:i]))
        if ancestor in covered:
            covered_ancestor = True
            break
    if covered_ancestor:
        continue
    
    # Not covered by any rule
    missing.append((d, "not in adr-scope.md and no ancestor coverage"))

if missing:
    for d, reason in missing:
        fail(1, f"Directory not in scope: {d} ({reason})")
    print(f"  FAIL: {len(missing)} directories missing from adr-scope.md")
    for d, r in missing[:20]:
        print(f"    {d}: {r}")
    if len(missing) > 20:
        print(f"    ... and {len(missing)-20} more")
else:
    print("  All directories covered by scope rules")

check1_status = "FAIL" if any(c == 1 for c, _ in issues) else "PASS"
print(f"  CHECK 1: {check1_status}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 2 — Actual ADR files match COVERED entries
# ═══════════════════════════════════════════════════════════════════════════
print("\n── CHECK 2: Files match COVERED entries ──")

# Double-nesting check
wrong_depth = list((SRC / "src").rglob("ADR.md")) if (SRC / "src").exists() else []
if wrong_depth:
    for f in wrong_depth:
        fail(2, f"Wrong nesting depth: {f}")
    print(f"  FAIL: {len(wrong_depth)} ADRs at wrong depth (./src/src/)")
else:
    print("  No double-nesting found")

# All actual ADR.md files
actual_adrs = sorted(SRC.rglob("ADR.md"))
actual_adr_dirs = set()
for a in actual_adrs:
    rel = a.parent.relative_to(SRC)
    actual_adr_dirs.add(str(rel))

print(f"  Found {len(actual_adrs)} ADR.md files")
print(f"  COVERED entries: {len(covered)}")

# For every COVERED dir: verify ADR.md exists at exactly ./src/<dir>/ADR.md
missing_adrs = []
for d in sorted(covered.keys()):
    expected = SRC / d / "ADR.md"
    if not expected.exists():
        missing_adrs.append(d)
        fail(2, f"COVERED dir missing ADR.md: {d}")

if missing_adrs:
    print(f"  FAIL: {len(missing_adrs)} COVERED dirs missing ADR.md")
    for d in missing_adrs:
        print(f"    {d}")
else:
    print("  All COVERED dirs have ADR.md")

# Orphan ADRs: ADR.md exists but not in COVERED
orphan_adrs = []
for a in actual_adrs:
    rel = str(a.parent.relative_to(SRC))
    if rel not in covered:
        orphan_adrs.append(str(a))
        fail(2, f"Orphan ADR.md (not in COVERED): {a}")

if orphan_adrs:
    print(f"  FAIL: {len(orphan_adrs)} orphan ADR.md files")
    for o in orphan_adrs:
        print(f"    {o}")
else:
    print("  No orphan ADR.md files")

# Count check
if len(actual_adrs) != len(covered):
    fail(2, f"ADR count mismatch: {len(actual_adrs)} files vs {len(covered)} COVERED entries")
    print(f"  FAIL: count mismatch ({len(actual_adrs)} actual vs {len(covered)} COVERED)")
else:
    print(f"  Count matches: {len(actual_adrs)}")

check2_status = "FAIL" if any(c == 2 for c, _ in issues) else "PASS"
print(f"  CHECK 2: {check2_status}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 3 — Exclusion justifications valid
# ═══════════════════════════════════════════════════════════════════════════
print("\n── CHECK 3: Exclusion justifications ──")

# Load book chapter text for name-checking
book_text = ""
if BOOK.exists():
    for chf in BOOK.glob("*.md"):
        try:
            book_text += chf.read_text(errors='replace') + "\n"
        except Exception:
            pass

invalid_exclusions = []
for d, reason in sorted(excluded.items()):
    if reason not in VALID_EXCLUSION_REASONS:
        fail(3, f"Invalid exclusion reason for {d}: '{reason}'")
        invalid_exclusions.append((d, reason))

if invalid_exclusions:
    print(f"  FAIL: {len(invalid_exclusions)} invalid exclusion reasons")
    for d, r in invalid_exclusions:
        print(f"    {d}: '{r}'")
else:
    print("  All exclusion reasons are valid")

# Check if EXCLUDED dirs are named in book as distinct architectural units
# (basic heuristic: dir basename mentioned as distinct unit)
book_violations = []
for d, reason in sorted(excluded.items()):
    parts = Path(d).parts
    name = parts[-1]  # basename
    depth = len(parts)
    # Only check if not already obviously non-architectural
    if reason == LEAF_EXCLUSION and depth <= 2:
        # Check if named in book as distinct architectural unit
        # Simple check: look for the name as a section heading or bold/emphasized term
        if re.search(rf'\b{re.escape(name)}\b', book_text, re.IGNORECASE):
            # More specific: check if used as an architectural unit (not just a passing mention)
            # Look for headings or prominent mentions
            if re.search(rf'##.*{re.escape(name)}|`{re.escape(name)}`', book_text, re.IGNORECASE):
                book_violations.append((d, reason, name))

# Line-count checks for non-exempt reasons
line_count_violations = []
for d, reason in sorted(excluded.items()):
    src_dir = SRC / d
    if not src_dir.exists():
        continue
    
    if reason in ("Test data only", "Test suite", "Vendored/third-party", "Auto-generated code"):
        continue  # exempt
    
    # Count lines of source files at maxdepth 1
    exts = ('*.py', '*.cpp', '*.h', '*.cu', '*.cc', '*.cxx', '*.hpp')
    lines = 0
    for ext in exts:
        for f in src_dir.glob(ext):
            try:
                lines += len(f.read_text(errors='replace').splitlines())
            except Exception:
                pass
    
    if reason == BUILD_EXCLUSION and lines > 2000:
        fail(3, f"Build/config-only dir {d} has {lines} lines (>2000)")
        line_count_violations.append((d, reason, lines, 2000))
    elif reason == EMPTY_EXCLUSION and lines > 50:
        fail(3, f"Empty/stub dir {d} has {lines} lines (>50)")
        line_count_violations.append((d, reason, lines, 50))
    elif reason == LEAF_EXCLUSION and lines > 200:
        fail(3, f"Leaf dir {d} has {lines} lines (>200)")
        line_count_violations.append((d, reason, lines, 200))

if line_count_violations:
    print(f"  FAIL: {len(line_count_violations)} line-count violations")
    for d, r, lc, limit in line_count_violations:
        print(f"    {d}: {r}, {lc} lines (limit {limit})")
else:
    print("  All line-count checks pass")

check3_status = "FAIL" if any(c == 3 for c, _ in issues) else "PASS"
print(f"  CHECK 3: {check3_status}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 4 — ADR content non-stub
# ═══════════════════════════════════════════════════════════════════════════
print("\n── CHECK 4: ADR content non-stub ──")

REQUIRED_SECTIONS = ["Role", "Key Files", "Public Interface", "Dependencies",
                     "Runtime Behaviour", "Performance Profile", "Design Rationale"]

stub_issues = []

def count_sentences(text):
    # Simple sentence counter
    return len(re.findall(r'[.!?]+', text))

for d in sorted(covered.keys()):
    adr_path = SRC / d / "ADR.md"
    if not adr_path.exists():
        continue  # Already flagged in Check 2
    
    try:
        content = adr_path.read_text(errors='replace')
    except Exception as e:
        fail(4, f"Cannot read {adr_path}: {e}")
        continue
    
    lines = content.splitlines()
    non_empty = [l for l in lines if l.strip()]
    
    adr_issues = []
    
    # Title heading check
    if not non_empty or not non_empty[0].startswith('# '):
        adr_issues.append("does not begin with level-1 heading")
        fail(4, f"{d}/ADR.md: missing level-1 title heading")
    else:
        # Check heading contains backtick-wrapped directory path
        title = non_empty[0]
        if '`' not in title:
            adr_issues.append("title heading missing backtick-wrapped dir path")
            fail(4, f"{d}/ADR.md: title heading missing backtick dir path")
    
    # Section index: bare bullet list immediately after title
    # Find lines after title heading
    title_idx = next((i for i, l in enumerate(lines) if l.strip() and l.startswith('# ')), None)
    if title_idx is not None:
        # Lines after title, skip blanks
        after_title = []
        for l in lines[title_idx+1:]:
            if l.strip():
                after_title.append(l)
            # Stop at first non-empty line that isn't a bullet
            # We collect until we hit a non-bullet non-empty line
            if l.strip() and not l.startswith('-') and not l.startswith('*'):
                break
        
        # Check for section index bullets
        section_bullets = [l for l in after_title if l.startswith('-') or l.startswith('*')]
        if len(section_bullets) < 3:
            adr_issues.append("missing section-index bullet list after title")
            fail(4, f"{d}/ADR.md: missing section-index bullet list after title heading")
    
    # Key Files section appears exactly once and is a table
    kf_matches = list(re.finditer(r'^##\s+Key Files', content, re.MULTILINE))
    if len(kf_matches) == 0:
        adr_issues.append("missing ## Key Files section")
        fail(4, f"{d}/ADR.md: missing ## Key Files section")
    elif len(kf_matches) > 1:
        adr_issues.append("## Key Files appears more than once")
        fail(4, f"{d}/ADR.md: ## Key Files duplicated")
    else:
        # Check it's a table (not bullet list)
        kf_start = kf_matches[0].end()
        kf_section = content[kf_start:kf_start+2000]
        # Find first non-blank content
        kf_lines = [l for l in kf_section.splitlines() if l.strip()][:10]
        has_table = any('|' in l for l in kf_lines[:5])
        has_bullets = any(l.startswith('-') for l in kf_lines[:5])
        if has_bullets and not has_table:
            adr_issues.append("Key Files section is bullet list, not table")
            fail(4, f"{d}/ADR.md: Key Files is bullet list, not table")
        # Check for placeholder rows
        if has_table:
            table_rows = [l for l in kf_lines if '|' in l and not re.match(r'\s*\|[-|:\s]+\|', l)]
            # Remove header row
            data_rows = table_rows[1:] if len(table_rows) > 1 else []
            if not data_rows:
                adr_issues.append("Key Files table has no data rows")
                fail(4, f"{d}/ADR.md: Key Files table has no rows")
            else:
                # Check for placeholder rows
                placeholders = [r for r in data_rows if re.search(r'placeholder|TODO|TBD|N/A', r, re.I)]
                real_rows = [r for r in data_rows if r not in placeholders]
                if not real_rows:
                    adr_issues.append("Key Files table has only placeholder rows")
                    fail(4, f"{d}/ADR.md: Key Files table has only placeholders")
    
    # Dependencies section
    dep_match = re.search(r'^##\s+Dependencies', content, re.MULTILINE)
    if dep_match:
        dep_section = content[dep_match.end():dep_match.end()+2000]
        dep_lines = [l.strip() for l in dep_section.splitlines() if l.strip()][:15]
        has_table = any('|' in l for l in dep_lines[:8])
        has_none_stmt = any(
            re.search(r'no\s+(?:notable|external|src[-\s]local|direct|further|additional|internal)', l, re.I)
            or re.search(r'no\s+\S+\s+dep', l, re.I)
            or re.search(r'none', l, re.I)
            for l in dep_lines[:8]
        )
        if not has_table and not has_none_stmt:
            adr_issues.append("Dependencies section has no table and no 'no notable dependencies' statement")
            fail(4, f"{d}/ADR.md: Dependencies section missing table or no-dep statement")
    
    # Runtime Behaviour: at least 2 sentences
    rb_match = re.search(r'^##\s+Runtime Behaviour', content, re.MULTILINE)
    if rb_match:
        next_section = re.search(r'^##\s+', content[rb_match.end():], re.MULTILINE)
        rb_text = content[rb_match.end():rb_match.end() + (next_section.start() if next_section else 2000)]
        rb_sentences = count_sentences(rb_text)
        if rb_sentences < 2:
            adr_issues.append(f"Runtime Behaviour has <2 sentences ({rb_sentences})")
            fail(4, f"{d}/ADR.md: Runtime Behaviour has fewer than 2 sentences")
    else:
        adr_issues.append("missing ## Runtime Behaviour section")
        fail(4, f"{d}/ADR.md: missing Runtime Behaviour section")
    
    # Performance Profile: at least 2 sentences
    pp_match = re.search(r'^##\s+Performance Profile', content, re.MULTILINE)
    if pp_match:
        next_section = re.search(r'^##\s+', content[pp_match.end():], re.MULTILINE)
        pp_text = content[pp_match.end():pp_match.end() + (next_section.start() if next_section else 2000)]
        pp_sentences = count_sentences(pp_text)
        if pp_sentences < 2:
            adr_issues.append(f"Performance Profile has <2 sentences ({pp_sentences})")
            fail(4, f"{d}/ADR.md: Performance Profile has fewer than 2 sentences")
    else:
        adr_issues.append("missing ## Performance Profile section")
        fail(4, f"{d}/ADR.md: missing Performance Profile section")
    
    if adr_issues:
        stub_issues.append((d, adr_issues))

# Check for relative paths (../) in ADR dependency links
print("  Checking for relative path violations (../)...")
rel_path_violations = []
for adr_file in actual_adrs:
    content = adr_file.read_text(errors='replace')
    for i, line in enumerate(content.splitlines(), 1):
        # Find markdown link targets: ](target) — only flag if the target starts with ../
        for m in re.finditer(r'\]\(([^)]+)\)', line):
            target = m.group(1)
            if target.startswith('../') or '/../' in target:
                rel = adr_file.relative_to(SRC)
                rel_path_violations.append((str(rel), i, line.strip()))
                fail(4, f"Relative path in {rel} line {i}: {line.strip()}")
                break

if rel_path_violations:
    print(f"  FAIL: {len(rel_path_violations)} relative path violations")
    for f, l, line in rel_path_violations:
        print(f"    {f}:{l}: {line[:80]}")
else:
    print("  No relative path (../) violations")

# Check for broken ADR dependency links
print("  Checking for broken ADR dependency links...")
broken_links = []
for adr_file in actual_adrs:
    content = adr_file.read_text(errors='replace')
    for i, line in enumerate(content.splitlines(), 1):
        for m in re.finditer(r'\]\(([^)]*ADR\.md)\)', line):
            link = m.group(1)
            if link.startswith('http'):
                continue
            # Strip leading ./ or /
            link_clean = link.lstrip('./')
            target = SRC / link_clean
            if not target.exists():
                rel = adr_file.relative_to(SRC)
                broken_links.append((str(rel), i, link))
                fail(4, f"Broken link in {rel} line {i}: '{link}' not found at {target}")

if broken_links:
    print(f"  FAIL: {len(broken_links)} broken ADR dependency links")
    for f, l, link in broken_links:
        print(f"    {f}:{l}: '{link}'")
else:
    print("  No broken ADR dependency links")

if stub_issues:
    print(f"  FAIL: {len(stub_issues)} ADRs have content issues")
    for d, iss in stub_issues[:10]:
        print(f"    {d}: {'; '.join(iss)}")
    if len(stub_issues) > 10:
        print(f"    ... and {len(stub_issues)-10} more")
else:
    print("  All ADR content checks pass")

check4_status = "FAIL" if any(c == 4 for c, _ in issues) else "PASS"
print(f"  CHECK 4: {check4_status}")

# ═══════════════════════════════════════════════════════════════════════════
# CHECK 5 — Book subsystem cross-reference
# ═══════════════════════════════════════════════════════════════════════════
print("\n── CHECK 5: Book subsystem cross-reference ──")

# Read book _generated/architecture-map.md for subsystems
generated_dir = BOOK / "_generated"
book_subsystems = []  # list of (src_rel_path, source)

if generated_dir.exists():
    arch_map = generated_dir / "architecture-map.md"
    if arch_map.exists():
        try:
            text = arch_map.read_text(errors='replace')
            # Extract table rows like | `./src/c10/core` | ... |
            for m in re.finditer(r'\|\s*`\./src/([^`]+)`\s*\|', text):
                path = m.group(1).rstrip('/')
                # Skip if it's a file (has a file extension)
                if '.' not in Path(path).name:
                    book_subsystems.append((path, "architecture-map.md"))
        except Exception as e:
            print(f"  Warning: could not read architecture-map.md: {e}")

# Also scan chapter files for named subsystems using backtick paths
for chf in sorted(BOOK.glob("*.md")):
    if chf.name in ("BOOK-INDEX.md", "BOOK-VALIDATION.md", "manuscript-complete.md"):
        continue
    try:
        text = chf.read_text(errors='replace')
        # Find backtick-wrapped directory paths
        for m in re.finditer(r'`((?:torch|aten|c10|caffe2|functorch|torchgen|tools|test|third_party|benchmarks|cmake|docs|scripts|android|binaries)[^`]*)`', text):
            name = m.group(1).strip().rstrip('/')
            # Filter to directory-like paths (no spaces, no file-extension-only name)
            if ' ' not in name and '.' not in Path(name).name:
                book_subsystems.append((name, chf.name))
    except Exception:
        pass

# Deduplicate
seen = {}
for path, source in book_subsystems:
    if path not in seen:
        seen[path] = source
book_subsystems_dedup = list(seen.items())

print(f"  Found {len(book_subsystems_dedup)} unique directory-like subsystems in book")

check5_fails = []
check5_results = []
for name, source in sorted(book_subsystems_dedup):
    # Is this name COVERED or has a COVERED ancestor?
    is_covered = name in covered
    has_covered_ancestor = False
    parts = Path(name).parts
    for i in range(1, len(parts)):
        anc = str(Path(*parts[:i]))
        if anc in covered:
            has_covered_ancestor = True
            break
    
    # Is it EXCLUDED with no covered ancestor?
    is_excluded = name in excluded
    
    if not is_covered and not has_covered_ancestor:
        if is_excluded:
            fail(5, f"Book-named subsystem '{name}' is EXCLUDED but has no COVERED ancestor")
            check5_fails.append((name, "EXCLUDED, no COVERED ancestor"))
            check5_results.append((name, "EXCLUDED (no COVERED ancestor)"))
        else:
            # Check if the directory even exists
            if (SRC / name).exists():
                fail(5, f"Book-named subsystem '{name}' exists but has no coverage in adr-scope.md")
                check5_fails.append((name, "exists but no coverage"))
                check5_results.append((name, "NOT COVERED (directory exists)"))
            else:
                check5_results.append((name, "not found in src (may be renamed/merged)"))
    else:
        check5_results.append((name, "COVERED" if is_covered else f"covered via ancestor"))

if check5_fails:
    print(f"  FAIL: {len(check5_fails)} book-named subsystems without coverage")
    for name, reason in check5_fails:
        print(f"    {name}: {reason}")
else:
    print("  All book-named subsystems are covered")

check5_status = "FAIL" if any(c == 5 for c, _ in issues) else "PASS"
print(f"  CHECK 5: {check5_status}")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n══════════════════════════════════════════")
print("SUMMARY")
print(f"  Check 1 (Scope map current):     {check1_status}")
print(f"  Check 2 (Files match COVERED):   {check2_status}")
print(f"  Check 3 (Exclusion reasons):     {check3_status}")
print(f"  Check 4 (ADR content non-stub):  {check4_status}")
print(f"  Check 5 (Book cross-reference):  {check5_status}")
print()

all_statuses = [check1_status, check2_status, check3_status, check4_status, check5_status]
overall = "PASS" if all(s == "PASS" for s in all_statuses) else "FAIL"
print(f"  OVERALL: {overall}")
print("══════════════════════════════════════════")

if issues:
    print(f"\nTotal issues: {len(issues)}")
    by_check = {}
    for c, msg in issues:
        by_check.setdefault(c, []).append(msg)
    for c in sorted(by_check.keys()):
        print(f"  Check {c}: {len(by_check[c])} issues")

# Return structured data for report generation
results = {
    "check1": {"status": check1_status, "missing_dirs": missing, "pending": pending},
    "check2": {"status": check2_status, "missing_adrs": missing_adrs, "orphan_adrs": orphan_adrs,
               "wrong_depth": wrong_depth, "count_actual": len(actual_adrs), "count_covered": len(covered)},
    "check3": {"status": check3_status, "invalid": invalid_exclusions, "line_violations": line_count_violations},
    "check4": {"status": check4_status, "stub_issues": stub_issues,
               "rel_path_violations": rel_path_violations, "broken_links": broken_links},
    "check5": {"status": check5_status, "fails": check5_fails, "results": check5_results},
    "overall": overall,
    "issues": issues,
    "covered": covered,
    "excluded": excluded,
}

import json
out_path = Path(__file__).parent / "validation_results.json"
with open(out_path, 'w') as f:
    # Not all objects are JSON-serializable (Path), convert
    def to_str(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
