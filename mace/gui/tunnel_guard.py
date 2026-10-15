"""The ``mace gui --slurm`` tunnel's guard: ends the tunnel when the session dies.

    python tunnel_guard.py PARENT_PID ssh ARGS...

Run by path (standard library only) as the leader of the tunnel's own
process group. It starts the ssh in that group and waits for it. When its
parent is gone (``mace gui --slurm`` killed with SIGKILL, which runs no
cleanup) or it gets SIGTERM/SIGHUP, it ends the whole group: SIGTERM, then
SIGKILL after a grace period, so an ssh wrapper's children and a
ProxyCommand go too, not only the process the session started. The parent
is checked every 0.2 s (``os.getppid()`` changes when the parent dies, on
every POSIX system); on Linux the session also sets PR_SET_PDEATHSIG on this
process, which delivers SIGTERM at once.

ssh's stdin, stdout and stderr are this process's (the session reads the
stderr pipe); the exit code is ssh's (128 + N for a signal).
"""
import os
import signal
import subprocess
import sys
import time

POLL_S = 0.2
GRACE_S = 3.0


def main(argv):
    parent = int(argv[1])
    stop = []

    def on_signal(signum, frame):
        stop.append(signum)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, on_signal)
    if os.getppid() != parent:                         # the session died before we got here
        return 1
    proc = subprocess.Popen(argv[2:])
    while True:
        try:
            code = proc.wait(POLL_S)
            return 128 - code if code < 0 else code
        except subprocess.TimeoutExpired:
            pass
        if stop or os.getppid() != parent:
            break
    # the session ended (or asked us to end): the whole group, this process aside
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        os.killpg(os.getpgrp(), signal.SIGTERM)
    except OSError:
        pass
    end = time.monotonic() + GRACE_S
    while proc.poll() is None and time.monotonic() < end:
        time.sleep(0.05)
    try:
        os.killpg(os.getpgrp(), signal.SIGKILL)       # anything left, this process included
    except OSError:
        pass
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
