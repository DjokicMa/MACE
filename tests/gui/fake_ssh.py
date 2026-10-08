#!/usr/bin/env python3
"""A fake `ssh` for the connect tests (no network): it parses -J, -o and -L,
binds the -L forward on 127.0.0.1 (ExitOnForwardFailure: a busy port is exit
255), and runs the remote command like a node would: under `env -i` with a
temporary HOME and a minimal PATH (so `bash -lc` sees no conda init and no
personal profile), in its own session (killing this process leaves the
"remote" process running, as a dropped connection does). stdin, stdout and
stderr are relayed through pipes, so EOF and a kill -9 reach the remote
process the way a real ssh connection's end does. A command that itself
starts with `ssh` (the nested hop) finds this fake again on PATH.

Environment: FAKE_SSH_HOME (the node's HOME), FAKE_SSH_ENV (JSON: more
variables for the node), FAKE_SSH_LOG (a folder: one <pid>.json per run with
the argv, the target and the remote child's pid), FAKE_SSH_REFUSE (a target
that answers "Permission denied (publickey)." with exit 255). With -N it only
forwards, until it is killed.
"""
import json
import os
import socket
import subprocess
import sys
import threading

FLAGS_WITH_VALUE = {"-J", "-o", "-L", "-p", "-i", "-l", "-F", "-E", "-R", "-D", "-W", "-b", "-c", "-m", "-O", "-S"}


def parse(argv):
    i, fwd, jumps = 0, [], []
    while i < len(argv):
        a = argv[i]
        if a == "-G":
            return "G", argv[i + 1], [], [], ""
        if a in FLAGS_WITH_VALUE:
            if a == "-L":
                fwd.append(argv[i + 1])
            if a == "-J":
                jumps.append(argv[i + 1])
            i += 2
            continue
        if a.startswith("-L") and len(a) > 2:
            fwd.append(a[2:])
            i += 1
            continue
        if a.startswith("-o") or a.startswith("-") and len(a) > 1:
            i += 1
            continue
        break
    target = argv[i] if i < len(argv) else ""
    cmd = " ".join(argv[i + 1:])
    return "run", target, fwd, jumps, cmd


def pump(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        for s in (a, b):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def forward(spec):
    parts = spec.split(":")
    lhost, lport, rhost, rport = (["127.0.0.1"] + parts)[-4:] if len(parts) == 3 else parts
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        ls.bind((lhost, int(lport)))
    except OSError as exc:
        sys.stderr.write(f"bind [{lhost}]:{lport}: {exc.strerror}\nchannel_setup_fwd_listener_tcpip: cannot listen "
                         f"to port: {lport}\nCould not request local forwarding.\n")
        sys.stderr.flush()
        os._exit(255)
    ls.listen(16)

    def loop():
        while True:
            try:
                c, _ = ls.accept()
            except OSError:
                return
            try:
                u = socket.create_connection((rhost, int(rport)), timeout=5)
                u.settimeout(None)
            except OSError:
                sys.stderr.write(f"channel 3: open failed: connect failed: Connection refused\n")
                c.close()
                continue
            threading.Thread(target=pump, args=(c, u), daemon=True).start()
            threading.Thread(target=pump, args=(u, c), daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()


def main():
    mode, target, fwd, jumps, cmd = parse(sys.argv[1:])
    if mode == "G":
        print(f"hostname {target}\nuser {os.environ.get('USER', 'user')}\nport 22")
        return 0
    if os.environ.get("FAKE_SSH_REFUSE") and target.split("@")[-1] == os.environ["FAKE_SSH_REFUSE"]:
        sys.stderr.write(f"{target}: Permission denied (publickey).\n")
        return 255
    for spec in fwd:
        forward(spec)
    if "-N" in sys.argv[1:sys.argv.index(target) if target in sys.argv else len(sys.argv)]:
        # -N: no remote command, only the forward (mace gui --slurm's hop to the job's node)
        log = os.environ.get("FAKE_SSH_LOG")
        if log:
            with open(os.path.join(log, f"{os.getpid()}.json"), "w") as fh:
                json.dump({"argv": sys.argv[1:], "target": target, "jumps": jumps, "fwd": fwd, "child": None,
                           "pid": os.getpid()}, fh)
        threading.Event().wait()
        return 0
    here = os.path.dirname(os.path.abspath(__file__))
    node_env = {"HOME": os.environ.get("FAKE_SSH_HOME", "/nonexistent"), "USER": os.environ.get("USER", "user"),
                "PATH": os.pathsep.join([os.environ.get("FAKE_SSH_BIN", here), "/usr/bin", "/bin"])}
    for k in ("FAKE_SSH_HOME", "FAKE_SSH_ENV", "FAKE_SSH_LOG", "FAKE_SSH_BIN", "FAKE_SSH_REFUSE"):
        if os.environ.get(k):
            node_env[k] = os.environ[k]
    node_env.update(json.loads(os.environ.get("FAKE_SSH_ENV") or "{}"))
    child = subprocess.Popen(["/usr/bin/env", "-i"] + [f"{k}={v}" for k, v in node_env.items()]
                             + ["bash", "-c", cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
    log = os.environ.get("FAKE_SSH_LOG")
    if log:
        with open(os.path.join(log, f"{os.getpid()}.json"), "w") as fh:
            json.dump({"argv": sys.argv[1:], "target": target, "jumps": jumps, "fwd": fwd, "child": child.pid,
                       "pid": os.getpid()}, fh)

    def relay_in():
        try:
            while True:
                d = os.read(0, 65536)
                if not d:
                    break
                child.stdin.write(d)
                child.stdin.flush()
        except OSError:
            pass
        try:
            child.stdin.close()
        except OSError:
            pass

    def relay(src, fd):
        try:
            while True:
                d = src.read1(65536) if hasattr(src, "read1") else src.read(65536)
                if not d:
                    break
                os.write(fd, d)
        except OSError:
            pass
    threading.Thread(target=relay_in, daemon=True).start()
    t1 = threading.Thread(target=relay, args=(child.stdout, 1), daemon=True)
    t2 = threading.Thread(target=relay, args=(child.stderr, 2), daemon=True)
    t1.start()
    t2.start()
    code = child.wait()
    t1.join(2)
    t2.join(2)
    return code


if __name__ == "__main__":
    sys.exit(main())
