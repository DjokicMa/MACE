"""Tests new in 1.1.5 name no real SLURM job of the HPCC QA runs.

A test needs only some job id; a real one says which job ran where and
when. Placeholders are 999xxx (tests/test_job_left_queue_without_output.py).
"""
import re

import pytest

from conftest import REPO_ROOT

NEW_IN_1_1_5 = [
    "tests/test_screened_hybrid_crash_recovery.py",
    "tests/test_job_left_queue_without_output.py",
]


@pytest.mark.parametrize("path", NEW_IN_1_1_5)
def test_job_ids_are_placeholders(path):
    text = (REPO_ROOT / path).read_text()
    ids = re.findall(r"(?i)\bjob(?:\s+id)?\s*[:=(]?\s*['\"]?(\d{5,})", text)
    ids += re.findall(r"slurm_job_id\s*=\s*['\"]?(\d{5,})", text)
    assert all(i.startswith("999") for i in ids), ids
