"""Confidentiality guardrail tests run in local dev and CI."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_confidentiality import (  # noqa: E402
    scan_git_history,
    scan_parquet_files,
    scan_text_files,
)


def test_zero_confidential_tokens_in_repo() -> None:
    leaks = scan_text_files(REPO_ROOT)
    assert leaks == [], f"Confidential tokens detected: {leaks}"


def test_zero_confidential_tokens_or_real_gps_in_parquet() -> None:
    leaks = scan_parquet_files(REPO_ROOT)
    assert leaks == [], f"Confidential metadata or non-synthetic GPS detected: {leaks}"


def test_zero_confidential_tokens_in_git_history() -> None:
    leaks = scan_git_history(REPO_ROOT)
    assert leaks == [], f"Confidential tokens in git history: {leaks}"
