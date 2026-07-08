#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


SECTION_NAMES = [
    "Role",
    "Key Files",
    "Public Interface",
    "Dependencies",
    "Runtime Behaviour",
    "Performance Profile",
    "Design Rationale",
]
ALLOWED_EXCLUSION_REASONS = {
    "Auto-generated code",
    "Build/config only",
    "Vendored/third-party",
    "Test data only",
    "Test suite",
    "Empty or stub",
    "Leaf with no architectural boundary",
}
LINE_COUNT_LIMITS = {
    "Build/config only": 2000,
    "Empty or stub": 50,
    "Leaf with no architectural boundary": 200,
}
SOURCE_EXTENSIONS = {".py", ".cpp", ".h", ".cu", ".cc", ".cxx", ".hpp"}
ROW_RE = re.compile(
    r"^\|\s*`(?P<directory>[^`]+)`\s*\|\s*(?P<status>COVERED|EXCLUDED|PENDING)\s*\|\s*(?P<detail>.*?)\s*\|$"
)
HEADER_RE = re.compile(r"^##\s+(.*)$")
ADR_LINK_RE = re.compile(r"\]\(([^)]+ADR\.md)\)")
CODE_PATH_RE = re.compile(r"`([^`]+)`")
FILE_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+")
SENTENCE_RE = re.compile(r"[.!?](?=\s|$)")
BOOK_DIR_MENTION_RE = re.compile(r"`([A-Za-z0-9_./-]+/?)`")


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def load_scope(scope_path: Path):
    rows = []
    for line_no, line in enumerate(scope_path.read_text().splitlines(), start=1):
        match = ROW_RE.match(line.strip())
        if match:
            row = match.groupdict()
            row["line"] = line_no
            rows.append(row)
    return rows


def numbered_chapter_files(book_root: Path):
    return sorted(book_root.glob("[0-9][0-9]-*.md"))


def collect_book_dir_mentions(book_root: Path, src_root: Path):
    mentions = defaultdict(list)
    for chapter in numbered_chapter_files(book_root):
        for line_no, line in enumerate(chapter.read_text().splitlines(), start=1):
            for match in BOOK_DIR_MENTION_RE.finditer(line):
                raw = match.group(1).strip()
                candidate = raw.rstrip("/")
                if not candidate or "." in Path(candidate).name:
                    continue
                if (src_root / candidate).is_dir():
                    mentions[candidate].append(f"{chapter.relative_to(book_root.parent).as_posix()}:{line_no}")
    return mentions


def collect_generated_subsystems(book_root: Path, src_root: Path):
    subsystem_mentions = defaultdict(list)
    for generated_name in ["chapter-map.md", "architecture-map.md", "component-map.md"]:
        generated_path = book_root / "_generated" / generated_name
        if not generated_path.exists():
            continue
        for line_no, line in enumerate(generated_path.read_text().splitlines(), start=1):
            for match in re.finditer(r"`(\./src/[^`]+)`", line):
                raw = match.group(1)
                relative = raw[len("./src/") :]
                path = src_root / relative
                if path.is_dir():
                    subsystem_mentions[relative].append(
                        f"book/_generated/{generated_name}:{line_no}"
                    )
    return subsystem_mentions


def split_sections(lines):
    sections = {}
    current = None
    for index, line in enumerate(lines):
        header = HEADER_RE.match(line.strip())
        if header:
            current = header.group(1).strip()
            sections[current] = {"start": index, "lines": []}
            continue
        if current is not None:
            sections[current]["lines"].append(line)
    return sections


def count_sentences(text):
    return len(SENTENCE_RE.findall(text))


def find_existing_file_reference(section_lines, src_root: Path, adr_dir: Path):
    for line in section_lines:
        for token in FILE_TOKEN_RE.findall(line):
            token = token.rstrip(".,:;")
            candidate = token.split(":")[0]
            if candidate.startswith("./"):
                candidate = candidate[2:]
            relative_path = adr_dir / candidate
            absolute_path = src_root / candidate
            if relative_path.exists():
                return relative_path.relative_to(src_root).as_posix()
            if absolute_path.exists():
                return candidate
    return None


def extract_adr_links(text):
    return ADR_LINK_RE.findall(text)


def normalize_dependency_target(target):
    return target.split("#", 1)[0]


def check1(repo_root: Path):
    script_path = repo_root / "src" / "adr" / "_tools" / "validate_scope.py"
    proc = run(["python3", str(script_path), "--repo-root", str(repo_root)], repo_root)
    data = json.loads(proc.stdout)
    notes = []
    failures = []
    if not data["scope_exists"]:
        failures.append("src/adr-scope.md is missing")
    if data["missing_depth1"]:
        failures.append(
            "missing depth-1 dirs: " + ", ".join(sorted(data["missing_depth1"]))
        )
    if data["missing_other"]:
        failures.append(
            "missing unclassified dirs: " + ", ".join(sorted(data["missing_other"]))
        )
    if data["pending_rows"]:
        failures.append(
            "PENDING rows: "
            + ", ".join(
                f"{row['directory']} (src/adr-scope.md:{row['line']})"
                for row in data["pending_rows"]
            )
        )
    if not failures:
        notes.append("all present")
    return {
        "status": "PASS" if not failures else "FAIL",
        "notes": "; ".join(notes or failures),
        "details": data,
        "failures": failures,
    }


def check2(repo_root: Path, scope_rows):
    src_root = repo_root / "src"
    nested = sorted(path.relative_to(repo_root).as_posix() for path in (src_root / "src").glob("**/ADR.md")) if (src_root / "src").exists() else []
    actual = sorted(path.relative_to(src_root).as_posix() for path in src_root.glob("**/ADR.md"))
    covered_dirs = sorted(row["directory"] for row in scope_rows if row["status"] == "COVERED")
    actual_dirs = sorted(path[: -len("/ADR.md")] for path in actual)

    missing = [directory for directory in covered_dirs if f"{directory}/ADR.md" not in actual]
    unexpected = [path for path in actual if path[: -len("/ADR.md")] not in covered_dirs]

    failures = []
    if nested:
        failures.append("wrong-depth ADRs under ./src/src: " + ", ".join(nested))
    if missing:
        failures.append("missing ADR files: " + ", ".join(f"./src/{d}/ADR.md" for d in missing))
    if unexpected:
        failures.append("ADR files without COVERED entry: " + ", ".join(f"./src/{p}" for p in unexpected))
    if len(actual) != len(covered_dirs):
        failures.append(
            f"ADR count mismatch: actual {len(actual)} vs COVERED {len(covered_dirs)}"
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "notes": f"count matches ({len(actual)})" if not failures else "; ".join(failures),
        "details": {
            "nested": nested,
            "actual": actual,
            "actual_dirs": actual_dirs,
            "covered_dirs": covered_dirs,
            "missing": missing,
            "unexpected": unexpected,
        },
        "failures": failures,
    }


def line_count_for_directory(directory: Path):
    total = 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix in SOURCE_EXTENSIONS:
            total += len(path.read_text(errors="ignore").splitlines())
    return total


def check3(repo_root: Path, scope_rows):
    src_root = repo_root / "src"
    book_root = repo_root / "book"
    chapter_mentions = collect_book_dir_mentions(book_root, src_root)
    invalid = []

    for row in scope_rows:
        if row["status"] != "EXCLUDED":
            continue
        directory = row["directory"]
        reason = row["detail"]
        depth = len(directory.split("/"))
        entry = {
            "directory": directory,
            "reason": reason,
            "line": row["line"],
            "issues": [],
        }
        if reason not in ALLOWED_EXCLUSION_REASONS:
            entry["issues"].append("reason not in approved set")
        if reason == "Leaf with no architectural boundary" and depth > 2:
            entry["issues"].append("leaf reason used deeper than depth 2")
        mentions = chapter_mentions.get(directory, [])
        if mentions:
            entry["issues"].append("named in book chapters at " + ", ".join(mentions))
        if reason in LINE_COUNT_LIMITS and (src_root / directory).exists():
            lines = line_count_for_directory(src_root / directory)
            entry["line_count"] = lines
            if lines > LINE_COUNT_LIMITS[reason]:
                entry["issues"].append(
                    f"line count {lines} exceeds limit {LINE_COUNT_LIMITS[reason]}"
                )
        if entry["issues"]:
            invalid.append(entry)

    failures = []
    for entry in invalid:
        failures.append(
            f"{entry['directory']} (src/adr-scope.md:{entry['line']}): " + "; ".join(entry["issues"])
        )
    return {
        "status": "PASS" if not failures else "FAIL",
        "notes": "all exclusions valid" if not failures else "; ".join(failures),
        "details": {"invalid": invalid, "chapter_mentions": chapter_mentions},
        "failures": failures,
    }


def section_text(section):
    return "\n".join(section["lines"]).strip()


def check_adr_content(adr_path: Path, directory: str, src_root: Path, repo_root: Path):
    lines = adr_path.read_text().splitlines()
    failures = []
    first_nonempty = next((line.strip() for line in lines if line.strip()), "")
    expected_title = f"# `{directory}`"
    if first_nonempty != expected_title:
        failures.append(f"{adr_path.relative_to(src_root).as_posix()}: title should be '{expected_title}'")

    title_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    bullet_lines = []
    if title_index is None:
        failures.append(f"{adr_path.relative_to(src_root).as_posix()}: file is empty")
    else:
        i = title_index + 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        while i < len(lines) and lines[i].strip().startswith("- "):
            bullet_lines.append(lines[i].strip())
            i += 1
        if not bullet_lines:
            failures.append(f"{adr_path.relative_to(src_root).as_posix()}: missing bullet index after title")
        expected_links = {f"- [{name}](#{name.lower().replace(' ', '-')})" for name in SECTION_NAMES}
        actual_links = set(bullet_lines)
        if not expected_links.issubset(actual_links):
            failures.append(
                f"{adr_path.relative_to(src_root).as_posix()}: section index missing entries for "
                + ", ".join(name for name in SECTION_NAMES if f"- [{name}](#{name.lower().replace(' ', '-')})" not in actual_links)
            )

    sections = split_sections(lines)
    for section_name in SECTION_NAMES:
        if section_name not in sections:
            failures.append(f"{adr_path.relative_to(src_root).as_posix()}: missing section '## {section_name}'")

    key_files_count = sum(1 for line in lines if line.strip() == "## Key Files")
    if key_files_count != 1:
        failures.append(f"{adr_path.relative_to(src_root).as_posix()}: '## Key Files' appears {key_files_count} times")

    actual_ref = None
    if "Key Files" in sections:
        key_lines = [line for line in sections["Key Files"]["lines"] if line.strip()]
        if any(line.strip().startswith("- ") for line in key_lines):
            failures.append(f"{adr_path.relative_to(src_root).as_posix()}: Key Files section uses bullets instead of a table")
        table_rows = [line for line in key_lines if line.strip().startswith("|")]
        if len(table_rows) < 3:
            failures.append(f"{adr_path.relative_to(src_root).as_posix()}: Key Files table has no data rows")
        actual_ref = find_existing_file_reference(key_lines, src_root, adr_path.parent)
        if actual_ref is None:
            failures.append(f"{adr_path.relative_to(src_root).as_posix()}: Key Files section has no real file paths")

    if "Dependencies" in sections:
        dep_text = section_text(sections["Dependencies"])
        dep_lines = [line for line in sections["Dependencies"]["lines"] if line.strip()]
        has_table = len([line for line in dep_lines if line.strip().startswith("|")]) >= 3
        has_none_statement = "no notable dependencies" in dep_text.lower()
        if not has_table and not has_none_statement:
            failures.append(
                f"{adr_path.relative_to(src_root).as_posix()}: Dependencies section needs a table or 'no notable dependencies'"
            )

    for section_name in ["Runtime Behaviour", "Performance Profile"]:
        if section_name in sections:
            if count_sentences(section_text(sections[section_name])) < 2:
                failures.append(
                    f"{adr_path.relative_to(src_root).as_posix()}: {section_name} has fewer than 2 sentences"
                )

    if actual_ref is None:
        failures.append(f"{adr_path.relative_to(src_root).as_posix()}: no actual file, function, or type reference detected")

    file_text = adr_path.read_text()
    dotdot_lines = []
    broken_links = []
    for line_no, line in enumerate(file_text.splitlines(), start=1):
        if ".." in line:
            dotdot_lines.append(f"{adr_path.relative_to(repo_root).as_posix()}:{line_no}:{line}")
        for target in extract_adr_links(line):
            normalized = normalize_dependency_target(target)
            if target.startswith("../"):
                failures.append(
                    f"{adr_path.relative_to(src_root).as_posix()}:{line_no}: dependency link uses ../ -> {target}"
                )
            if not (src_root / normalized).is_file():
                broken_links.append(
                    f"{adr_path.relative_to(src_root).as_posix()}:{line_no}: broken ADR link '{target}'"
                )
    failures.extend(broken_links)
    return failures, dotdot_lines


def check4(repo_root: Path, scope_rows):
    src_root = repo_root / "src"
    dotdot_matches = []
    failures = []
    for row in scope_rows:
        if row["status"] != "COVERED":
            continue
        adr_path = src_root / row["directory"] / "ADR.md"
        if not adr_path.exists():
            continue
        adr_failures, adr_dotdot = check_adr_content(adr_path, row["directory"], src_root, repo_root)
        failures.extend(adr_failures)
        dotdot_matches.extend(adr_dotdot)
    return {
        "status": "PASS" if not failures else "FAIL",
        "notes": "all ADRs satisfy content requirements" if not failures else "; ".join(failures),
        "details": {"failures": failures, "dotdot_matches": dotdot_matches},
        "failures": failures,
    }


def covered_by_ancestor(directory: str, covered_set):
    parts = directory.split("/")
    for index in range(len(parts), 0, -1):
        candidate = "/".join(parts[:index])
        if candidate in covered_set:
            return candidate
    return None


def check5(repo_root: Path, scope_rows):
    src_root = repo_root / "src"
    book_root = repo_root / "book"
    covered_set = {row["directory"] for row in scope_rows if row["status"] == "COVERED"}
    excluded_set = {row["directory"] for row in scope_rows if row["status"] == "EXCLUDED"}
    generated = collect_generated_subsystems(book_root, src_root)
    chapters = collect_book_dir_mentions(book_root, src_root)

    subsystems = defaultdict(list)
    for directory, refs in generated.items():
        subsystems[directory].extend(refs)
    for directory, refs in chapters.items():
        if len(directory.split("/")) >= 2:
            subsystems[directory].extend(refs)

    failures = []
    subsystem_rows = []
    for directory in sorted(subsystems):
        covered_by = covered_by_ancestor(directory, covered_set)
        status = "COVERED" if covered_by else "NOT COVERED"
        if covered_by:
            status = f"COVERED via {covered_by}"
        elif directory in excluded_set:
            status = "EXCLUDED"
        subsystem_rows.append(
            {"directory": directory, "status": status, "refs": sorted(set(subsystems[directory]))}
        )
        if covered_by:
            continue
        if directory in excluded_set:
            failures.append(
                f"{directory}: EXCLUDED but named in book at " + ", ".join(sorted(set(subsystems[directory])))
            )
        else:
            failures.append(
                f"{directory}: not covered in adr-scope.md; named in book at "
                + ", ".join(sorted(set(subsystems[directory])))
            )
    return {
        "status": "PASS" if not failures else "FAIL",
        "notes": "all book-named subsystems are covered" if not failures else "; ".join(failures),
        "details": {"subsystems": subsystem_rows},
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    scope_rows = load_scope(repo_root / "src" / "adr-scope.md")

    results = {
        "check1": check1(repo_root),
        "check2": check2(repo_root, scope_rows),
        "check3": check3(repo_root, scope_rows),
        "check4": check4(repo_root, scope_rows),
        "check5": check5(repo_root, scope_rows),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
