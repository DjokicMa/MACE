"""Settings › Advanced: cluster defaults (1.1.6): the site settings MACE
writes into job scripts, read and changed through `mace site`.

GET  site/settings   the resolved settings with where each comes from, as
                     `mace site` prints them (site_cli.render of the same
                     site.describe), the site.toml path and the keys
                     `mace site set` takes
POST site/edit       {root_gen, action: set|unset|init, key, value}: runs
                     `mace site set KEY VALUE`, `mace site unset KEY` or
                     `mace site --init` (the CLI itself, with the
                     environment a submit gets); its words and exit code
                     are the answer

An edit needs the page's root_gen (409 "reload" without it or with another
one), and none is made while MACE GUI is open to the network (409).
"""
from __future__ import annotations

import shlex
from typing import Any, Dict, List

from . import submit as sub

ROUTES_GET = {"site/settings"}
ROUTES_POST = {"site/edit"}
NETWORK_OFF = "Changing the cluster defaults is turned off while MACE GUI is open to the network"
MAX_WORD = 200


def keys() -> List[str]:
    """Every key `mace site set` takes, the per-type ones written out."""
    from mace.submission import site, site_edit
    out = [k for k in site_edit.valid_keys() if "<" not in k]
    for level in ("properties",) + tuple(site.RESOURCE_TYPES):
        out += [f"resources.{level}.{k}" for k in site.TABLE_KEYS]
    return out


def settings(st) -> Dict[str, Any]:
    from mace.submission import site, site_cli
    from .submit_routes import _network
    env = sub.submit_env()
    try:
        d = site.describe(env, check_account=True)
        text = "\n".join(site_cli.render(d))
    except Exception as exc:              # pragma: no cover - a broken site.toml is shown, not raised
        d, text = {"errors": [str(exc)]}, f"mace site: {exc}"
    cfg = d.get("config") or {}
    return {"text": text, "command": "mace site", "path": cfg.get("path"), "exists": bool(cfg.get("exists")),
            "errors": d.get("errors") or [], "warnings": d.get("warnings") or [], "keys": keys(),
            "network": _network(st), "root_gen": getattr(st, "root_gen", None)}


def get(handler, route: str, query):
    return handler._json(settings(handler.state))


def _word(body: Dict[str, Any], name: str) -> str:
    v = body.get(name)
    if not isinstance(v, str) or not v.strip() or len(v) > MAX_WORD or "\n" in v or "\x00" in v:
        return ""
    return v.strip()


def post(handler, route: str, body: Dict[str, Any]):
    from . import clirun
    from .submit_routes import _SITE_CACHE, _gen, _network
    st = handler.state
    if body.get("root_gen") is None and not handler.headers.get("X-Mace-Root-Gen"):
        raise sub.SubmitConflict("The folder changed; reload")
    _gen(handler, body)
    if _network(st):
        raise sub.SubmitConflict(NETWORK_OFF)
    action = body.get("action")
    if action == "init":
        argv = ["site", "--init"]
    elif action == "set":
        key, value = _word(body, "key"), _word(body, "value")
        if not key or not value:
            raise sub.SubmitError("mace site set takes a key and a value")
        argv = ["site", "set", key, value]
    elif action == "unset":
        key = _word(body, "key")
        if not key:
            raise sub.SubmitError("mace site unset takes a key")
        argv = ["site", "unset", key]
    else:
        raise sub.SubmitError("action is set, unset or init")
    # set/unset take the words as written after "--" (a value may start with --)
    run = argv if action == "init" else argv[:2] + ["--"] + argv[2:]
    r = clirun.run(sub.mace_argv(*run), st.root, None, timeout=120, env=sub.submit_env(), cancelable=False)
    _SITE_CACHE.clear()                   # the Submit dialog's Site row reads the new settings
    out = "\n".join(x for x in ((r.stdout or "").strip(), (r.stderr or "").strip()) if x)
    return handler._json({"ok": r.returncode == 0, "code": r.returncode, "output": out,
                          "argv": ["mace", *argv], "command": "mace " + " ".join(shlex.quote(a) for a in argv),
                          "settings": settings(st)})
