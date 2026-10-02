"""Job scripts written by a MACE older than 1.1.3.

`mace submit` reuses an existing <job>.sh unless --overwrite-sh is given, and
a workflow builds every step's script from the workflow_scripts/ copy taken
when it was planned. Scripts from before 1.1.3 have no scratch guard - on a
node where $SCRATCH is empty CRYSTAL then reads no INPUT and stops with
"END OF DATA IN INPUT DECK" - and none of the later staging (GUESSP .f20).
They are not rewritten here, since a user may have edited them; reusing one
is only reported.
"""
import re

#: Every MACE-generated job script copies the deck to INPUT in scratch.
_INPUT_COPY = re.compile(
    r'^\s*cp\s+"?\$\{?DIR\}?/\$\{?JOB\}?\.(?:d12|d3)"?\s+"?\$\{?scratch\}?/\$\{?JOB\}?/INPUT',
    re.M)
GUARD_MARKER = "SCRATCH GUARD"


def predates_scratch_guard(text: str) -> bool:
    """True for a MACE-generated job script (or generator) without the guard."""
    return bool(text) and GUARD_MARKER not in text and bool(_INPUT_COPY.search(text))


def reused_script_warning(name: str, properties: bool = False) -> str:
    also = "" if properties else " and no GUESSP restart staging"
    return (f"  WARNING: {name} was written by an older MACE (before 1.1.3) and is reused as it is.\n"
            f"    It has no scratch guard (on a node where $SCRATCH is empty CRYSTAL stops with\n"
            f"    END OF DATA IN INPUT DECK){also}. Regenerate it with\n"
            f"    `mace submit --overwrite-sh ...` (this replaces {name}, including any edits).")


def workflow_template_warning(name: str) -> str:
    return (f"  WARNING: workflow_scripts/{name} was copied from an older MACE (before 1.1.3):\n"
            f"    it has no scratch guard (on a node where $SCRATCH is empty CRYSTAL stops with\n"
            f"    END OF DATA IN INPUT DECK) and no GUESSP restart staging. workflow_scripts/ is a\n"
            f"    snapshot taken when the workflow was planned: re-plan the workflow to pick up the\n"
            f"    job-script fixes.")
