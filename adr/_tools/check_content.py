#!/usr/bin/env python3
"""Check 4: Verify ADR content quality - all 7 sections, dependencies, title backticks."""
import re
from pathlib import Path
import os

os.chdir('/home/github-runner/actions-runner/_work/CodeWeave-PyTorch/CodeWeave-PyTorch')

adrs = sorted(Path('src').rglob('ADR.md'))
issues = []

REQUIRED_SECTIONS = ['Role', 'Key Files', 'Public Interface', 'Dependencies',
                     'Runtime Behaviour', 'Performance Profile', 'Design Rationale']

for adr in adrs:
    content = adr.read_text()
    rel = str(adr)

    # Check all 7 required sections present
    for sec in REQUIRED_SECTIONS:
        if f'## {sec}' not in content:
            issues.append(f'{rel}: Missing section ## {sec}')

    # Check title backtick path
    lines = content.split('\n')
    h1 = next((l for l in lines if l.startswith('# ')), '')
    if h1 and '`' not in h1:
        issues.append(f'{rel}: H1 title does not contain backtick path: {repr(h1[:80])}')

    # Check Dependencies section has table or explicit no-dep statement
    dep_match = re.search(r'## Dependencies\n(.*?)(?=\n## |\Z)', content, re.DOTALL)
    if dep_match:
        dep_text = dep_match.group(1).strip()
        has_table = '|' in dep_text
        has_no_dep_stmt = bool(re.search(r'no (notable |external )?depend', dep_text, re.IGNORECASE))
        if not has_table and not has_no_dep_stmt:
            issues.append(f'{rel}: Dependencies: no table and no explicit no-dependency statement')

    # Check section bullet index after H1
    h1_idx = next((i for i, l in enumerate(lines) if l.strip() and l.startswith('# ')), -1)
    if h1_idx >= 0:
        after_h1 = [l for l in lines[h1_idx+1:] if l.strip()]
        found_bullet = False
        for l in after_h1:
            if l.startswith('## '):
                break
            if l.startswith('- [') or l.startswith('- **'):
                found_bullet = True
        if not found_bullet:
            issues.append(f'{rel}: No bare bullet list of section links after title heading')

print(f'Total ADRs checked: {len(adrs)}')
if issues:
    print(f'ISSUES ({len(issues)}):')
    for i in issues:
        print(f'  {i}')
else:
    print('All section checks PASS.')
