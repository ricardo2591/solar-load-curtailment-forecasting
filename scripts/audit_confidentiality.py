"""Automated confidentiality & PII leak scanner for the public portfolio repo."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]

# Case-insensitive proprietary company, client, operator, and real site terms
FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("company_name", re.compile(r"\bhybrico\b", re.IGNORECASE)),
    ("internal_platform", re.compile(r"hyforce", re.IGNORECASE)),
    ("site_prefix_hyb", re.compile(r"\bhyb_[a-z0-9_-]+", re.IGNORECASE)),
    ("operator_tigo", re.compile(r"\btigo\b", re.IGNORECASE)),
    ("operator_claro", re.compile(r"\bclaro\b", re.IGNORECASE)),
    ("operator_pti", re.compile(r"\bpti\b", re.IGNORECASE)),
    ("operator_movistar", re.compile(r"\bmovistar\b|\btelefonica\b", re.IGNORECASE)),
    ("operator_digicel", re.compile(r"\bdigicel\b|\baltice\b", re.IGNORECASE)),
    ("towerco", re.compile(r"\bamerican tower\b|\bsba communications\b", re.IGNORECASE)),
    (
        "real_site_names",
        re.compile(
            r"fesitranh|aguacatan|chuipoclaj|tocoa|locomapa|finca_sonora|acirdu|cascabel|"
            r"colonia_indiana|el_porvenir|los_encuentros|talgua|lempira",
            re.IGNORECASE,
        ),
    ),
    (
        "real_site_codes",
        re.compile(
            r"\b(?:hn2076|hht780|put7031|san7001|mag7022|qtz275|ttn323|cun7024|hn223|hn670|hn1345|sol740)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "real_country_references",
        re.compile(r"\b(?:honduras|guatemala|el salvador|nicaragua|colombia|chile)\b", re.IGNORECASE),
    ),
    ("windows_user_path", re.compile(r"C:\\\\?Users\\\\?PC", re.IGNORECASE)),
]

TEXT_EXTENSIONS = {".py", ".ipynb", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".cfg"}
IGNORED_DIRS = {".git", ".pytest_cache", ".ruff_cache", "__pycache__", ".venv", "build", "dist"}


def _extract_scannable_text(path: Path) -> str:
    """Return text content to scan; strip opaque base64 image payloads from notebooks."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() != ".ipynb":
        return raw

    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return raw

    parts: list[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        parts.append("".join(src) if isinstance(src, list) else str(src))
        for out in cell.get("outputs", []):
            for key in ("text", "ename", "evalue"):
                val = out.get(key)
                if val:
                    parts.append("".join(val) if isinstance(val, list) else str(val))
            data = out.get("data", {})
            if isinstance(data, dict):
                for mime, payload in data.items():
                    if str(mime).startswith("image/"):
                        continue
                    parts.append("".join(payload) if isinstance(payload, list) else str(payload))
    return "\n".join(parts)


def scan_text_files(repo_root: Path = REPO_ROOT) -> list[str]:
    """Scan all tracked/source text files (including notebooks) for forbidden tokens."""
    findings: list[str] = []
    self_path = Path(__file__).resolve()

    for path in repo_root.rglob("*"):
        if path.is_dir() or any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.resolve() == self_path or path.name == "test_confidentiality.py":
            continue
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        content = _extract_scannable_text(path)
        rel = path.relative_to(repo_root)
        for label, pattern in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(content):
                findings.append(f"{rel} [{label}]: '{match.group(0)}'")
    return findings


def scan_parquet_files(repo_root: Path = REPO_ROOT) -> list[str]:
    """Scan Parquet schemas, categorical metadata, and string columns for leaks or land GPS."""
    findings: list[str] = []
    for parquet_file in repo_root.rglob("*.parquet"):
        if any(part in IGNORED_DIRS for part in parquet_file.parts):
            continue
        rel = parquet_file.relative_to(repo_root)
        df = pl.read_parquet(parquet_file)

        for col in df.columns:
            for label, pattern in FORBIDDEN_PATTERNS:
                if pattern.search(col):
                    findings.append(f"{rel} [col:{label}]: '{col}'")

            if df[col].dtype in (pl.Utf8, pl.Categorical):
                unique_vals = df[col].cast(pl.Utf8).drop_nulls().unique().to_list()
                for val in unique_vals:
                    for label, pattern in FORBIDDEN_PATTERNS:
                        if pattern.search(val):
                            findings.append(f"{rel} [{col}:{label}]: '{val}'")

        # Ensure all coordinates reside strictly within synthetic mid-ocean boxes
        if "latitude" in df.columns and "longitude" in df.columns:
            coords = df.select(["latitude", "longitude"]).unique()
            for lat, lon in coords.iter_rows():
                in_atlantic_box = -55.0 <= lon <= -35.0 and -25.0 <= lat <= 25.0
                in_pacific_box = -145.0 <= lon <= -125.0 and -25.0 <= lat <= 25.0
                if not (in_atlantic_box or in_pacific_box):
                    findings.append(f"{rel} [gps_out_of_synthetic_ocean_box]: ({lat}, {lon})")
    return findings


def scan_git_history(repo_root: Path = REPO_ROOT) -> list[str]:
    """Scan local git commit headers/messages (author, email, subject) if `.git` exists."""
    if not (repo_root / ".git").exists():
        return []
    result = subprocess.run(
        ["git", "-C", str(repo_root), "log", "--all", "--format=%an|%ae|%cn|%ce|%s"],
        capture_output=True,
        text=True,
        check=False,
    )
    findings: list[str] = []
    for line in result.stdout.splitlines():
        for label, pattern in FORBIDDEN_PATTERNS:
            for match in pattern.finditer(line):
                findings.append(f"git_history [{label}]: '{match.group(0)}' in '{line}'")
    return findings


def main() -> int:
    findings = scan_text_files() + scan_parquet_files() + scan_git_history()
    if findings:
        print("CONFIDENTIALITY AUDIT FAILED:")
        for item in findings:
            print(f"  - {item}")
        return 1
    print("Confidentiality audit passed: 0 leaks detected across code, notebooks, docs, and Parquet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
