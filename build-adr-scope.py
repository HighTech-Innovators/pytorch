#!/usr/bin/env python3
"""
Generate adr-scope.md by cataloging all directories in src and determining source file presence.
"""
import os
import subprocess
from pathlib import Path
from collections import defaultdict

# Extensions that indicate source code
SOURCE_EXTENSIONS = {
    '.py', '.cpp', '.h', '.cu', '.cc', '.cxx', '.hpp', '.ts', '.js', '.kt', '.java'
}

def find_all_directories(root):
    """Find all directories under root, excluding dot directories."""
    dirs = set()
    for dirpath, dirnames, _ in os.walk(root):
        # Remove dot directories from dirnames to skip them
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
        rel_path = os.path.relpath(dirpath, root)
        if rel_path != '.':
            dirs.add(rel_path)
    return sorted(dirs)

def has_source_files(directory):
    """Check if directory contains source files."""
    try:
        for root, dirs, files in os.walk(directory):
            # Skip dot directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if any(f.endswith(ext) for ext in SOURCE_EXTENSIONS):
                    return True
    except (OSError, PermissionError):
        pass
    return False

def classify_directory(rel_path):
    """Classify directory as PENDING, EXCLUDED, or COVERED based on heuristics."""
    path_lower = rel_path.lower()
    
    # Auto-generated code
    if 'generated' in path_lower or '_generated' in path_lower:
        return 'EXCLUDED', 'Auto-generated code'
    
    # Build/config only
    if path_lower in ['cmake', 'build', '.cmake', 'build_tools']:
        return 'EXCLUDED', 'Build/config only'
    if any(x in path_lower for x in ['cmake', 'build_info', 'windows_ci']):
        return 'EXCLUDED', 'Build/config only'
    
    # Vendored/third-party
    if 'third_party' in path_lower or 'vendor' in path_lower:
        return 'EXCLUDED', 'Vendored/third-party'
    
    # Test data
    if 'test' in path_lower and 'data' in path_lower:
        return 'EXCLUDED', 'Test data only'
    if 'fixtures' in path_lower or 'testdata' in path_lower:
        return 'EXCLUDED', 'Test data only'
    
    # Empty or stub
    full_path = os.path.join('/home/rob/repos/github/OpenWeave-PyTorch/src', rel_path)
    if not has_source_files(full_path):
        return 'EXCLUDED', 'Empty or stub'
    
    return 'PENDING', ''

src_root = '/home/rob/repos/github/OpenWeave-PyTorch/src'
all_dirs = find_all_directories(src_root)

# Generate scope table
with open(os.path.join(src_root, 'adr-scope.md'), 'w') as f:
    f.write('# ADR Scope\n\n')
    f.write('| Directory | Source files present | Status | Reason (if EXCLUDED) |\n')
    f.write('|---|---|---|---|\n')
    
    for rel_path in all_dirs:
        full_path = os.path.join(src_root, rel_path)
        has_src = has_source_files(full_path)
        src_str = 'yes' if has_src else 'no'
        
        status, reason = classify_directory(rel_path)
        reason_str = reason if reason else ''
        
        f.write(f'| ./{rel_path} | {src_str} | {status} | {reason_str} |\n')

print(f"Generated adr-scope.md with {len(all_dirs)} directories")
