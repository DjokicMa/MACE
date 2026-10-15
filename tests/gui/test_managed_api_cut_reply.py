"""managed._api (the control calls of mace gui --managed and --slurm: stop,
detach, ping, attach) when the server ends while it answers: a stop shuts
the server down from a second thread, and under load the process can end
after the reply's headers but before its body. http.client then raises
IncompleteRead, an HTTPException rather than an OSError, which ended
`mace gui --slurm`'s relay with exit 1 instead of its MACE_GUI_STOPPED line
(CI, test_slurm_session_e2e: a port taken while the job waits)."""
import socket
import threading

import pytest

from mace.gui import managed


def _cut_server(reply: bytes):
    """A one-shot HTTP server that sends `reply` and closes."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve():
        c, _ = srv.accept()
        c.recv(65536)
        c.sendall(reply)
        c.close()
        srv.close()
    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


@pytest.mark.parametrize("reply", [
    b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: 18\r\n\r\n",       # body cut
    b"HTTP/1.0 200 OK\r\nContent-Length: 18\r\n\r\n{\"ok\": tr",                               # half a body
    b"HTTP/1.0 2",                                                                              # status line cut
    b"",                                                                                        # nothing
])
def test_a_reply_cut_by_the_stopping_server_is_no_answer(reply):
    port = _cut_server(reply)
    assert managed._api(port, "/s/x/", "t", "managed/stop", {"watcher": "w1"}, timeout=5) is None


def test_a_whole_reply_still_reads():
    port = _cut_server(b"HTTP/1.0 200 OK\r\nContent-Length: 10\r\n\r\n{\"ok\": 1} ")
    assert managed._api(port, "/s/x/", "t", "managed/stop", {"watcher": "w1"}, timeout=5) == {"ok": 1}


@pytest.mark.parametrize("reply", [b"HTTP/1.0 200 OK\r\nContent-Length: 18\r\n\r\n", b"HTTP/1.0 2"])
def test_connect_and_login_url_read_a_cut_reply_as_no_answer(reply):
    """The same for mace gui connect's requests through the tunnel (wait_ready,
    running_tasks) and mace gui login-url."""
    from mace.gui import connect, loginurl
    assert connect._http(f"http://127.0.0.1:{_cut_server(reply)}/s/x/api/info", "t") is None
    assert loginurl.new_link({"base": f"http://127.0.0.1:{_cut_server(reply)}/s/x/", "token": "t"}) is None
