"""Run a job-script generator without letting it overwrite itself.

The shipped templates (submitcrystal23.sh, submit_prop.sh and their
workflow_scripts/ copies) are generators: `bash <generator> <job>` writes the
batch file <job>.sh in the current directory. The workflow engine and
executor keep the customized generator for a step as <job>.sh. Running that
file with <job> truncated and rewrote the file bash was reading, and bash then
read the rest of the new file from the old offset. The submitted script
stopped after the module loads (no CRYSTAL run and no completion callback), so
every automatic follow-up was a job that ended in seconds with no output.
"""
import os
import subprocess
import tempfile
from pathlib import Path


def is_script_generator(content: str) -> bool:
    """True for a template that writes the batch file rather than being one."""
    return "echo '#!/bin/bash --login' >" in content or 'echo "#SBATCH' in content


def run_script_generator(script_path, job_name: str, cwd) -> subprocess.CompletedProcess:
    """Run ``bash <script_path> <job_name>`` in ``cwd`` from a separate copy.

    The generator runs from a temporary copy in ``cwd`` (so a generator that
    looks at its own directory sees the same one), and the copy is removed
    afterwards. ``<cwd>/<job_name>.sh`` then holds the generated batch file.
    """
    cwd = Path(cwd)
    content = Path(script_path).read_text()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{job_name}.", suffix=".generator.sh",
                                    dir=str(cwd))
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(content)
        return subprocess.run(["bash", tmp_name, job_name], cwd=str(cwd),
                              capture_output=True, text=True)
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
