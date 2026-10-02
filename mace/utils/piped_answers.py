"""Stop an interactive run whose piped answers are out of step.

The same rule as Crystal_d12/menu_nav.py (which must stay importable without
the mace package): with no terminal, a question that rejects several answers
in a row stops the run instead of asking forever.
"""
import sys

# A question with no terminal stops after this many invalid answers in a row:
# answers piped from a file that no longer matches the questions (one was
# added or merged) would otherwise be re-asked until a blank took the default
# and the run carried on out of step - or, with no blank, spin on the menu.
MAX_INVALID_PIPED_ANSWERS = 3


def stdin_is_terminal():
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def stop_if_out_of_step(invalid, prompt, expected):
    """Stop a run whose piped answers no longer match its questions.

    ``invalid`` is how many answers in a row this question rejected and
    ``expected`` says what it takes ("yes or no", "one of 1, 2, 3"). At a
    terminal the question just asks again.
    """
    if invalid >= MAX_INVALID_PIPED_ANSWERS and not stdin_is_terminal():
        raise SystemExit(
            f"Error: {invalid} answers in a row to '{str(prompt).strip()}' were not "
            f"{expected}, and stdin is not a terminal. The piped answers are "
            f"probably out of step with the questions, so the run stops here.")
