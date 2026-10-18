"""Run ``mace_cli`` and answer its questions by rule, as a user at the prompt would.

Calculation Setup never writes settings of its own: a field the CLI asks
about is set by answering that question, so the saved-answers JSON and the
deck are exactly what a user typing the same answers gets (plan B1). This
script is that user. It is run by path in its own process:

    python answer_driver.py RULES.json TRANSCRIPT.json MACE_CLI ARGS...

and replaces ``builtins.input`` before anything imports, so every question
the CLI asks (``input()``, the menu helpers, the back-navigation wrapper,
which captures ``builtins.input`` at import) reaches ``answer()``. What the
CLI printed since the previous question is the question's text; the first
rule whose ``match`` (a regular expression, case-insensitive) finds that
text, or the prompt itself, gives the answer:

* ``{"match": "...", "answer": "y"}``: type that;
* ``{"match": "...", "choose": "PBE0"}``: the number of the menu line
  ``<n>: PBE0 ...`` (or ``<n>. PBE0``) printed with the question;
* ``{"match": "...", "answer": "POTC", "or_choose": true}``: the number of
  the menu line naming the answer where the question shows one, else the
  answer as it is;
* ``"once": true``: the rule is used for one question only.

A question no rule matches gets Enter, the CLI's own default, only when
it is a known question: one the spec's ``known`` list (regular expressions,
matched the same way) names as a question where Enter is the intended
answer. Any other question is not answered: the run stops (exit 3) and the
transcript's error quotes the question, so a reworded prompt is reported,
never answered with a default nobody chose. Each question and its answer
go to the transcript (``rule``: the rule's index, ``known``: the list
entry's). A ``choose`` that finds no such line, or a run past
``max_prompts`` questions, stops the run the same way, so a changed CLI
menu is reported, never guessed.

stdin stays a non-terminal (as ``yes '' | mace ...``): the CLI behaves as it
does with piped answers.
"""
import builtins
import json
import os
import re
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


class _Tee:
    """sys.stdout / sys.stderr that also keeps what was printed since the last question."""

    def __init__(self, real, sink):
        self._real, self._sink = real, sink

    def write(self, s):
        self._sink.append(s)
        return self._real.write(s)

    def __getattr__(self, name):
        return getattr(self._real, name)


class Stop(SystemExit):
    pass


_OPTION = re.compile(r"^\s*(\d+)\s*[:.)]\s*(\S.*?)\s*$", re.M)


def pick(text, label):
    """The number of the menu line naming ``label``: ``7: HSE06 - ...``,
    ``2. Density Functional Theory (DFT)``, ``1: Standard - ...``. The text
    before " - " equal to the label wins, then a line that starts with it,
    then one that names it in parentheses; the last such line counts (a menu
    printed twice, as a description and then as the choices, is one menu)."""
    want = label.strip().lower()
    found = {1: None, 2: None, 3: None}
    for m in _OPTION.finditer(text):
        num, body = m.group(1), m.group(2)
        low = body.lower()
        head = re.split(r"\s+[-–]\s+", low, maxsplit=1)[0].strip()
        if head == want or low == want:
            found[1] = num
        elif re.match(re.escape(want) + r"(?=$|[\s,(:\[])", low):
            found[2] = num
        elif f"({want})" in low:
            found[3] = num
    for k in (1, 2, 3):
        if found[k] is not None:
            return found[k]
    return None


def main(argv):
    rules_path, transcript_path, cli = argv[1], argv[2], argv[3]
    with open(rules_path) as fh:
        spec = json.load(fh)
    rules = [dict(r) for r in spec.get("rules", [])]
    for r in rules:
        r["_re"] = re.compile(r["match"], re.I | re.S)
    known = [re.compile(k, re.I | re.S) for k in spec.get("known", [])]
    max_prompts = int(spec.get("max_prompts", 400))
    printed = []
    transcript = {"questions": [], "error": None}
    sys.stdout = _Tee(sys.stdout, printed)
    sys.stderr = _Tee(sys.stderr, printed)

    def save():
        try:
            with open(transcript_path, "w") as fh:
                json.dump(transcript, fh, indent=1)
        except OSError:
            pass

    def answer(prompt=""):
        text = "".join(printed)
        printed.clear()
        # the prompt goes out as input() writes it, but not into the next question's text
        out = sys.stdout._real if isinstance(sys.stdout, _Tee) else sys.stdout
        out.write(str(prompt))
        question = (text + str(prompt))[-6000:]
        if len(transcript["questions"]) >= max_prompts:
            transcript["error"] = f"more than {max_prompts} questions: stopped"
            save()
            raise Stop(3)
        reply, used = "", None
        for k, r in enumerate(rules):
            if r.get("done") or not r["_re"].search(question):
                continue
            used = k
            if "choose" in r:
                m = pick(text, str(r["choose"]))
                if m is None:
                    transcript["error"] = (f"no menu line '<n>: {r['choose']}' for the question "
                                           f"matching {r['match']!r}")
                    transcript["questions"].append({"question": question[-1500:], "answer": None, "rule": k})
                    save()
                    raise Stop(3)
                reply = m
            else:
                reply = str(r.get("answer", ""))
                # an option's name for a numbered menu ("POTC" for "2: POTC - ..."); not a menu, typed as is
                if r.get("or_choose"):
                    reply = pick(text, reply) or reply
            if r.get("once"):
                r["done"] = True
            break
        known_as = None
        if used is None:
            known_as = next((k for k, kr in enumerate(known) if kr.search(question)), None)
            if known_as is None:
                # the prompt and the line before it, menu lines left out ("Select ...:" / "Enter your choice:")
                lines = [x.strip() for x in question.splitlines() if x.strip() and not _OPTION.match(x)]
                asked = " / ".join(lines[-2:])[-400:]
                transcript["error"] = f"a question Setup does not know (not answered): {asked!r}"
                transcript["questions"].append({"question": question[-1500:], "prompt": str(prompt)[-300:],
                                                "answer": None, "rule": None, "known": None})
                save()
                raise Stop(3)
        transcript["questions"].append({"question": question[-1500:], "prompt": str(prompt)[-300:],
                                        "answer": reply, "rule": used, "known": known_as})
        return reply

    builtins.input = answer
    # as `python mace_cli ...` would run it: the script's folder first on sys.path
    sys.path[:] = [os.path.dirname(os.path.abspath(cli))] + [p for p in sys.path
                                                             if os.path.abspath(p or os.curdir) != _HERE]
    sys.argv = [cli] + argv[4:]
    code = 0
    try:
        runpy.run_path(cli, run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        save()
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
