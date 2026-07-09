#!/usr/bin/env python3
"""Comprehensive ADR validation — all 5 checks from work/2-validate-adrs.md."""
import re, os, subprocess, sys
from pathlib import Path

REPO = Path("/home/github-runner/actions-runner/_work/CodeWeave-PyTorch/CodeWeave-PyTorch/src")
BOOK = Path("/home/github-runner/actions-runner/_work/CodeWeave-PyTorch/CodeWeave-PyTorch/book")
SCOPE_FILE = REPO / "adr-scope.md"

# ─── Parse adr-scope.md ───────────────────────────────────────────────────────
scope_text = SCOPE_FILE.read_text()
covered, excluded, all_scope = set(), set(), set()
for line in scope_text.splitlines():
    m = re.match(r'\|\s*\.(\/[^\|]+?)\s*\|\s*[^\|]+\|\s*(COVERED|EXCLUDED)\s*\|', line)
    if not m:
        continue
    path = m.group(1).strip().lstrip('/')
    status = m.group(2)
    all_scope.add(path)
    if status == 'COVERED':
        covered.add(path)
    elif status == 'EXCLUDED':
        excluded.add(path)

has_pending = 'PENDING' in scope_text

# ─── Check 1: Scope map completeness ─────────────────────────────────────────
print("=== CHECK 1: Scope map ===")
result = subprocess.run(
    ["find", ".", "-type", "d", "-not", "-path", "*/.*"],
    cwd=REPO, capture_output=True, text=True
)
dirs = sorted(r.lstrip("./") for r in result.stdout.strip().splitlines() if r and r != ".")

check1_issues = []
if has_pending:
    check1_issues.append("PENDING entries found in adr-scope.md")

for d in dirs:
    parts = d.split("/")
    depth = len(parts)
    if depth == 1:
        if d not in all_scope:
            check1_issues.append(f"Depth-1 dir missing: ./{d}")
        continue
    if any("/".join(parts[:i]) in excluded for i in range(1, depth)):
        continue
    if any("/".join(parts[:i]) in covered for i in range(1, depth)):
        continue
    if d not in all_scope:
        check1_issues.append(f"Not in scope (no ancestor rule applies): ./{d}")

for issue in check1_issues[:20]:
    print(f"  ISSUE: {issue}")
print(f"  Total issues: {len(check1_issues)}")

# ─── Check 2: ADR files match COVERED ────────────────────────────────────────
print("\n=== CHECK 2: ADR files ===")
# Wrong nesting
wrong_nest = list((REPO / "src").rglob("ADR.md")) if (REPO / "src").exists() else []
print(f"  Double-nested ADRs (./src/src/): {len(wrong_nest)}")

actual_adrs = sorted(str(p.relative_to(REPO)) for p in REPO.rglob("ADR.md"))
print(f"  Actual ADR.md files: {len(actual_adrs)}")
print(f"  COVERED entries: {len(covered)}")

check2_issues = []
for c in sorted(covered):
    expected = REPO / c / "ADR.md"
    if not expected.exists():
        check2_issues.append(f"Missing ADR: ./{c}/ADR.md")

# Check no extra ADRs
covered_paths = {str(REPO / c / "ADR.md") for c in covered}
for adr in actual_adrs:
    full = str(REPO / adr)
    if full not in covered_paths:
        check2_issues.append(f"Orphan ADR not in COVERED: {adr}")

if len(actual_adrs) != len(covered):
    check2_issues.append(f"Count mismatch: {len(actual_adrs)} ADRs vs {len(covered)} COVERED entries")

for issue in check2_issues:
    print(f"  ISSUE: {issue}")
if not check2_issues:
    print("  PASS: all COVERED dirs have ADR.md, counts match")

# ─── Check 3: Exclusion justifications ───────────────────────────────────────
print("\n=== CHECK 3: Exclusion justifications ===")
VALID_REASONS = [
    "Auto-generated code", "Build/config only", "Vendored/third-party",
    "Test data only", "Test suite", "Empty or stub", "Leaf with no architectural boundary",
]

check3_issues = []
# Parse with reasons
for line in scope_text.splitlines():
    m = re.match(r'\|\s*\.(\/[^\|]+?)\s*\|\s*[^\|]+\|\s*EXCLUDED\s*\|\s*(.*?)\s*\|?$', line)
    if not m:
        continue
    path = m.group(1).strip().lstrip('/')
    reason = m.group(2).strip()
    if not any(reason.startswith(r) for r in VALID_REASONS):
        check3_issues.append(f"Invalid reason for ./{path}: '{reason[:60]}'")
        continue
    
    # Line count checks
    dir_path = REPO / path
    if not dir_path.exists():
        continue
    
    limit = None
    if reason.startswith("Build/config only"):
        limit = 2000
    elif reason.startswith("Empty or stub"):
        limit = 50
    elif reason.startswith("Leaf with no architectural boundary"):
        limit = 200

    if limit is not None:
        r = subprocess.run(
            ["find", str(dir_path), "-maxdepth", "1",
             "(", "-name", "*.py", "-o", "-name", "*.cpp", "-o", "-name", "*.h",
             "-o", "-name", "*.cu", "-o", "-name", "*.cc", "-o", "-name", "*.cxx",
             "-o", "-name", "*.hpp", ")"],
            capture_output=True, text=True
        )
        files = [f for f in r.stdout.strip().splitlines() if f]
        if files:
            wc = subprocess.run(["wc", "-l"] + files, capture_output=True, text=True)
            try:
                total = int(wc.stdout.strip().splitlines()[-1].split()[0])
            except (IndexError, ValueError):
                total = 0
        else:
            total = 0
        
        if total > limit:
            check3_issues.append(f"Line count violation: ./{path} ({reason.split('—')[0].strip()}) "
                                  f"has {total} lines at maxdepth 1, limit is {limit}")

# Check book mentions for "Leaf" exclusion
book_texts = {}
for bf in BOOK.glob("*.md"):
    book_texts[bf.name] = bf.read_text()

for line in scope_text.splitlines():
    m = re.match(r'\|\s*\.(\/[^\|]+?)\s*\|\s*[^\|]+\|\s*EXCLUDED\s*\|\s*(.*?)\s*\|?$', line)
    if not m:
        continue
    path = m.group(1).strip().lstrip('/')
    reason = m.group(2).strip()
    if not reason.startswith("Leaf with no architectural boundary"):
        continue
    # Check if named in book as distinct architectural unit (in directory structure or component tables)
    dir_name = path.split("/")[-1]
    for fname, content in book_texts.items():
        # Look for directory entries in tables/structures (not just API calls)
        if re.search(rf'`{re.escape(path)}/`\s*\|', content) or \
           re.search(rf'`{re.escape(path)}/`\s*—', content):
            # Verify it's a meaningful reference (with a description)
            check3_issues.append(f"Named in book ({fname}): ./{path} uses 'Leaf' but appears as named unit in {fname}")
            break

print(f"  Check 3 issues: {len(check3_issues)}")
for issue in check3_issues[:40]:
    print(f"  ISSUE: {issue}")

# ─── Check 4: ADR content non-stub ───────────────────────────────────────────
print("\n=== CHECK 4: ADR content ===")
check4_issues = {}
required_sections = ['Role', 'Key Files', 'Public Interface', 'Dependencies',
                     'Runtime Behaviour', 'Performance Profile', 'Design Rationale']

for c in sorted(covered):
    adr_path = REPO / c / "ADR.md"
    if not adr_path.exists():
        continue
    content = adr_path.read_text()
    lines = content.splitlines()
    problems = []
    
    # Title
    first = next((l for l in lines if l.strip()), '')
    if not re.match(r'^# `', first):
        problems.append(f"Title not # `<dir>`: {first[:60]}")
    
    # Section index
    ti = next((i for i, l in enumerate(lines) if l.strip()), None)
    if ti is not None:
        rest = lines[ti+1:]
        nni = next((i for i, l in enumerate(rest) if l.strip()), None)
        if nni is not None:
            bullets = []
            for l in rest[nni:]:
                if l.strip().startswith('-'):
                    bullets.append(l)
                elif not l.strip():
                    break
                else:
                    break
            if not bullets:
                problems.append("No section index bullet list after title")
            else:
                for s in required_sections:
                    if not any(s in b for b in bullets):
                        problems.append(f"Section index missing: {s}")
    
    # Key Files
    kf_count = content.count('## Key Files')
    if kf_count != 1:
        problems.append(f"## Key Files appears {kf_count} times")
    
    # Dependency links
    if re.search(r'\.\.\/', content):
        problems.append("Contains ../ relative path")
    
    # Runtime Behaviour
    in_rb = False; rb_text = []
    for line in lines:
        if '## Runtime Behaviour' in line: in_rb = True; continue
        if in_rb:
            if line.startswith('## '): break
            rb_text.append(line)
    sents = [s.strip() for s in re.split(r'[.!?]', ' '.join(rb_text)) if len(s.strip()) > 10]
    if len(sents) < 2:
        problems.append(f"Runtime Behaviour < 2 sentences ({len(sents)})")
    
    # Performance Profile
    in_pp = False; pp_text = []
    for line in lines:
        if '## Performance Profile' in line: in_pp = True; continue
        if in_pp:
            if line.startswith('## '): break
            pp_text.append(line)
    sents = [s.strip() for s in re.split(r'[.!?]', ' '.join(pp_text)) if len(s.strip()) > 10]
    if len(sents) < 2:
        problems.append(f"Performance Profile < 2 sentences ({len(sents)})")
    
    if problems:
        check4_issues[c] = problems

print(f"  ADRs with issues: {len(check4_issues)}")
for adr, probs in sorted(check4_issues.items()):
    print(f"  {adr}: {probs}")
if not check4_issues:
    print("  PASS: all ADRs pass content checks")

# ─── Check 5: Book cross-reference ───────────────────────────────────────────
print("\n=== CHECK 5: Book cross-reference ===")
check5_issues = []
# Parse component-map for distinct units
comp_map = (BOOK / "_generated" / "component-map.md").read_text()
arch_map = (BOOK / "_generated" / "architecture-map.md").read_text()
ch_map = (BOOK / "_generated" / "chapter-map.md").read_text()

# Key subsystems from maps — check if their directories are COVERED
subsystem_dir_map = {
    "c10/core": "TensorImpl/Storage/Dispatcher",
    "aten/src/ATen": "ATen operator library",
    "torch/autograd": "Autograd engine",
    "torch/nn": "nn.Module",
    "torchgen": "Code generation",
    "torch/jit": "JIT/TorchScript",
    "torch/_dynamo": "TorchDynamo",
    "torch/_inductor": "TorchInductor",
    "torch/fx": "FX Graph",
    "torch/distributed": "Distributed",
    "torch/profiler": "Profiler",
    "torch/csrc": "C++ bindings",
    "c10/util": "C++ utilities",
    "c10/cuda": "CUDA abstractions",
    "c10/mobile": "Mobile portability",
    "torch/_functorch": "Functional transforms",
    "torch/_export": "Export pipeline",
    "functorch": "Functional transforms bridge",
    "torch/optim": "Optimizers",
    "torch": "Python API surface",
    "c10": "Core C++ library",
}

for d, desc in subsystem_dir_map.items():
    if d in covered:
        print(f"  COVERED: ./{d} — {desc}")
        continue
    # Check for covered ancestor
    parts = d.split("/")
    has_ancestor = any("/".join(parts[:i]) in covered for i in range(1, len(parts)))
    if has_ancestor:
        print(f"  COVERED (via ancestor): ./{d} — {desc}")
    else:
        print(f"  UNCOVERED: ./{d} — {desc}")
        check5_issues.append(f"./{d} ({desc}) is not COVERED and has no COVERED ancestor")

print(f"\n  Check 5 issues: {len(check5_issues)}")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Check 1: {'FAIL' if check1_issues else 'PASS'} ({len(check1_issues)} issues)")
print(f"Check 2: {'FAIL' if check2_issues else 'PASS'} ({len(check2_issues)} issues)")
print(f"Check 3: {'FAIL' if check3_issues else 'PASS'} ({len(check3_issues)} issues)")
print(f"Check 4: {'FAIL' if check4_issues else 'PASS'} ({len(check4_issues)} issues)")
print(f"Check 5: {'FAIL' if check5_issues else 'PASS'} ({len(check5_issues)} issues)")
overall = any([check1_issues, check2_issues, check3_issues, check4_issues, check5_issues])
print(f"\nOverall: {'FAIL' if overall else 'PASS'}")
