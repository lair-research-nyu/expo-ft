"""Episode outcome labels and the sparse terminal reward.

The keypress flow is dsrl_pi0's (examples/train_agent_server.py read_outcome_keypress):
the server blocks inside its ENDPOINT_DONE handling until the operator labels the episode,
so the robot cannot start the next episode before the last one is recorded.

One change of meaning: in DSRL "0" was "abort/unsafe, worse than a failure" (-1). Here
the reward is the paper's binary {0, 1}, which cannot rank an early ending above a long
success, so that distinction is no longer needed. "0" now means "not the policy's
fault" (reset problem, hardware fault, operator stop), and it is stored as a truncation
(see ExpoFtConfig.abort_mode). Unsafe or degenerate behaviour is a FAILURE.
"""

from __future__ import annotations

import enum
import select
import sys
from typing import Optional, Protocol


class Outcome(str, enum.Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ABORT = "abort"


def read_outcome_keypress() -> Outcome:
    """Block until the operator presses 2 (success), 1 (failure) or 0 (abort)."""
    import termios
    import tty

    prompt = ("Trial finished. Mark as (2) Success, (1) Failure [incl. unsafe], "
              "or (0) Abort [not the policy's fault; kept as a truncated episode]:")
    keys = {"2": Outcome.SUCCESS, "1": Outcome.FAILURE, "0": Outcome.ABORT}
    print(prompt, flush=True)
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if select.select([sys.stdin], [], [], 0.05) == ([sys.stdin], [], []):
                char_input = sys.stdin.read(1)
                if char_input in keys:
                    outcome = keys[char_input]
                    print(f"Trial marked as {outcome.name}.", flush=True)
                    return outcome
                print(f"Invalid input. {prompt}", flush=True)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


class SuccessDetector(Protocol):
    """Labels a finished episode. `done_reason` is the client's done_reason string."""

    def label(self, done_reason: str) -> Optional[Outcome]: ...


class HumanLabel:
    """Default: ask the operator on the server's terminal."""

    def label(self, done_reason: str) -> Optional[Outcome]:
        if done_reason:
            print(f"done_reason={done_reason!r}", flush=True)
        return read_outcome_keypress()


class FixedLabel:
    """For dry runs and tests: a fixed outcome, or one keyed on the client's done_reason."""

    def __init__(self, outcome: Outcome = Outcome.SUCCESS, by_reason: Optional[dict[str, Outcome]] = None):
        self.outcome = outcome
        self.by_reason = by_reason or {}

    def label(self, done_reason: str) -> Optional[Outcome]:
        return self.by_reason.get(done_reason, self.outcome)
