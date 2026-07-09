#!/usr/bin/env python3
"""Check 5: Book subsystem cross-reference.

For each subsystem named as a distinct architectural unit in the book,
verify it is covered (directly COVERED or has a COVERED ancestor in adr-scope.md).
"""
import re
import os
from pathlib import Path

os.chdir('/home/github-runner/actions-runner/_work/CodeWeave-PyTorch/CodeWeave-PyTorch')

SCOPE_FILE = Path('src/adr-scope.md')

def parse_scope(scope_file):
    excluded = set()
    covered = set()
    with open(scope_file) as f:
        for line in f:
            m = re.match(r'\|\s*(\./[^|]+?)\s*\|[^|]*\|\s*(EXCLUDED|COVERED)\s*\|', line)
            if m:
                d = m.group(1).strip().lstrip('./').rstrip()
                status = m.group(2).strip()
                if status == 'EXCLUDED':
                    excluded.add(d)
                elif status == 'COVERED':
                    covered.add(d)
    return excluded, covered

def is_covered(path_str, covered):
    """Is this dir directly COVERED or has a COVERED ancestor?"""
    if path_str in covered:
        return True
    parts = path_str.split('/')
    for i in range(1, len(parts)):
        ancestor = '/'.join(parts[:i])
        if ancestor in covered:
            return True
    return False

def is_excluded_no_covered_ancestor(path_str, excluded, covered):
    """Is this dir excluded AND has no COVERED ancestor?"""
    if path_str in excluded:
        # Check for COVERED ancestor
        parts = path_str.split('/')
        for i in range(1, len(parts)):
            ancestor = '/'.join(parts[:i])
            if ancestor in covered:
                return False  # has a COVERED ancestor
        return True  # excluded with no covered ancestor
    return False

excluded, covered = parse_scope(SCOPE_FILE)

# Subsystems named as distinct architectural units in the book (from architecture-map.md and chapter-map.md)
# These are the directories called out as architectural candidates in the architecture-map
BOOK_SUBSYSTEMS = [
    # from architecture-map.md
    ('torch', 'Core Python API'),
    ('torch/autograd', 'Autograd system'),
    ('torch/nn', 'Neural network module system'),
    ('torch/nn/modules', 'Module implementations'),
    ('torch/optim', 'Optimisers'),
    ('torch/distributed', 'Distributed training'),
    ('torch/distributed/fsdp', 'FSDP parameter sharding'),
    ('torch/distributed/rpc', 'RPC framework'),
    ('torch/fx', 'FX graph system'),
    ('torch/jit', 'JIT compilation'),
    ('torch/_dynamo', 'TorchDynamo'),
    ('torch/_inductor', 'TorchInductor'),
    ('torch/profiler', 'Profiler'),
    ('torch/cuda', 'CUDA support'),
    ('torch/amp', 'Mixed precision'),
    ('torch/utils', 'Utilities'),
    ('torch/quantization', 'Quantisation'),
    ('torch/_export', 'Export pipeline'),
    ('torch/_functorch', 'Functional transforms'),
    ('c10/core', 'Core abstractions'),
    ('c10/util', 'C++ utilities'),
    ('c10/cuda', 'CUDA abstractions'),
    ('c10/mobile', 'Mobile support'),
    ('aten/src/ATen', 'ATen tensor library'),
    ('aten/src/ATen/core', 'ATen core dispatch'),
    ('aten/src/ATen/core/dispatch', 'Dispatcher'),
    ('aten/src/ATen/native', 'Native kernels'),
    ('aten/src/ATen/cpu', 'CPU backend'),
    ('aten/src/ATen/cuda', 'CUDA backend'),
    ('torchgen', 'Code generation'),
    ('torchgen/api', 'API translation'),
    ('torchgen/dest', 'Code generation targets'),
    ('torch/csrc', 'C++ binding bridge'),
    ('torch/csrc/autograd', 'C++ autograd'),
    ('torch/csrc/jit', 'JIT backend'),
    ('torch/csrc/jit/serialization', 'JIT serialisation'),
    ('functorch', 'Functional transforms bridge'),
    ('functorch/_src/aot_autograd', 'AOT Autograd'),
    ('tools', 'Build tools'),
]

results = []
failures = []

for d, desc in BOOK_SUBSYSTEMS:
    cov = is_covered(d, covered)
    excl_no_parent = is_excluded_no_covered_ancestor(d, excluded, covered)
    
    if not cov:
        status = 'NOT COVERED'
        failures.append(f'{d}: named in book as "{desc}" but not covered (no COVERED entry or ancestor)')
    elif excl_no_parent:
        status = 'EXCLUDED (no COVERED ancestor)'
        failures.append(f'{d}: named in book as "{desc}" but EXCLUDED with no COVERED ancestor')
    else:
        status = 'COVERED'
    
    results.append((d, desc, status))

print('=== Check 5: Book Subsystem Cross-Reference ===\n')
print(f'{"Directory":<40} {"Status":<30} Description')
print('-' * 100)
for d, desc, status in results:
    print(f'{d:<40} {status:<30} {desc}')

print(f'\nTotal subsystems: {len(results)}')
if failures:
    print(f'\nFAILURES ({len(failures)}):')
    for f in failures:
        print(f'  - {f}')
else:
    print('\nAll book subsystems are covered.')
print(f'\nCheck 5: {"FAIL" if failures else "PASS"}')
