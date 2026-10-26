"""Turns one episode's stream of server events into decision-point transitions.

Events, in arrival order:

    policy(obs, chunk, d, candidates)    the server answered a request with `chunk`
    intervention(obs, human_rows)        a human-takeover frame (endpoint=intervention)
    done(obs | None)                     the episode ended (endpoint=done)

Every event after the first may carry `prev_executed` (the wire's
prev_skipped_action_start): how many rows of the *previous* event's actions ran before
this one. When absent, a policy chunk is assumed to have run exactly C rows (fixed-step
async: a request every C steps) and a human frame all of its rows. The recorder lays the
executed rows end to end into one 50 Hz timeline, so a takeover k rows into a chunk
leaves k policy rows followed by the human's rows -- the human rows *replace* the policy
rows they overrode.

Transitions start at anchors (every policy decision; the first human frame of a takeover
and then one every `hil_anchor_stride` rows of it; the terminal observation ends the
last one). A transition from an anchor at step t takes the timeline rows [t, t+C) as its
action and bootstraps at the latest anchor at or before t+C, with discount gamma**n for
its n rows. Between two policy decisions n = C, which is Eq. 5 exactly.

Assumption about the intervention frame (to be confirmed against the GEN2 client): its
observation is taken when the frame's first human row starts executing, and its
`action` rows are what the robot then executed until the next event.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from expo_ft.ultra.buffer import Episode, Slot, Transition
from expo_ft.ultra.config import ExpoFtConfig
from expo_ft.ultra.rewards import Outcome


@dataclasses.dataclass
class Observation:
    images: np.ndarray  # (V, S, S, 3) uint8, critic resolution
    state: np.ndarray  # (A,) normalized
    state_robot: np.ndarray  # (A,) robot space


@dataclasses.dataclass
class _Event:
    kind: str  # "policy" | "human" | "done"
    obs: Optional[Observation]
    rows: Optional[np.ndarray]  # (k, A) normalized actions the event commanded
    prev_executed: Optional[int]
    prefix_len: int = 0
    prefix_robot: Optional[np.ndarray] = None
    candidates: Optional[np.ndarray] = None
    step: int = 0
    n_exec: int = 0


class EpisodeRecorder:
    def __init__(self, cfg: ExpoFtConfig):
        self.cfg = cfg
        self.events: list[_Event] = []
        self.prompt = ""

    def __len__(self) -> int:
        return len(self.events)

    @property
    def done(self) -> bool:
        return bool(self.events) and self.events[-1].kind == "done"

    def on_policy(self, obs: Observation, chunk: np.ndarray, *, prefix_len: int,
                  prefix_robot: Optional[np.ndarray], candidates: Optional[np.ndarray],
                  prev_executed: Optional[int] = None, prompt: str = "") -> None:
        """chunk: (H, A) normalized rows of what was *sent* (after any limiter);
        candidates: (N, >=C, A) normalized base chunks sampled at this observation, or None
        to have them filled later (demos)."""
        assert not self.done
        self.prompt = prompt or self.prompt
        c = self.cfg.exec_steps
        self.events.append(_Event("policy", obs, np.asarray(chunk, np.float32), prev_executed,
                                  prefix_len=int(prefix_len), prefix_robot=prefix_robot,
                                  candidates=None if candidates is None else np.asarray(candidates, np.float32)[:, :c]))

    def on_intervention(self, obs: Observation, human_rows: np.ndarray,
                        prev_executed: Optional[int] = None) -> None:
        assert not self.done
        self.events.append(_Event("human", obs, np.asarray(human_rows, np.float32), prev_executed))

    def on_done(self, obs: Optional[Observation], prev_executed: Optional[int] = None) -> None:
        assert not self.done
        self.events.append(_Event("done", obs, None, prev_executed))

    # ------------------------------------------------------------------ build
    def _lay_out_timeline(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assign steps; return (executed rows (T, A), human flag (T,), planned tail rows)."""
        cfg = self.cfg
        step = 0
        rows, human = [], []
        evs = self.events
        for i, e in enumerate(evs):
            e.step = step
            if e.kind == "done":
                break
            nxt = evs[i + 1] if i + 1 < len(evs) else None
            default = cfg.exec_steps if e.kind == "policy" else len(e.rows)
            n = default if nxt is None or nxt.prev_executed is None else int(nxt.prev_executed)
            n = int(np.clip(n, 0, len(e.rows)))
            e.n_exec = n
            rows.append(e.rows[:n])
            human.append(np.full(n, e.kind == "human"))
            step += n
        last = next((e for e in reversed(evs) if e.kind != "done"), None)
        tail = last.rows[last.n_exec:] if last is not None else np.zeros((0, cfg.action_dim), np.float32)
        a = cfg.action_dim
        return (np.concatenate(rows) if rows else np.zeros((0, a), np.float32),
                np.concatenate(human) if human else np.zeros(0, bool), tail)

    def _anchors(self) -> list[_Event]:
        stride = self.cfg.resolved_hil_anchor_stride
        out: list[_Event] = []
        for e in self.events:
            if e.obs is None:
                continue
            if e.kind == "human" and out and out[-1].kind == "human" and e.step - out[-1].step < stride:
                continue
            if out and e.step == out[-1].step:
                if e.kind == "done":
                    continue  # nothing ran since the last decision: no terminal transition to make
                out[-1] = e  # zero rows ran since the last anchor: keep the newer observation
                continue
            out.append(e)
        return out

    def build(self, outcome: Outcome) -> Optional[Episode]:
        """The episode as buffer transitions, or None if there is nothing to store."""
        cfg = self.cfg
        if outcome == Outcome.ABORT and cfg.abort_mode == "discard":
            return None
        if not self.events or self.events[0].kind == "done":
            return None
        timeline, human, tail = self._lay_out_timeline()
        anchors = self._anchors()
        terminal = anchors[-1] if anchors and anchors[-1].kind == "done" else None
        sources = [e for e in anchors if e.kind != "done"]
        if not sources:
            return None

        slots = [Slot(images=e.obs.images, state=e.obs.state, state_robot=e.obs.state_robot,
                      prefix_len=e.prefix_len if e.kind == "policy" else 0,
                      prefix_robot=e.prefix_robot if e.kind == "policy" else None,
                      candidates=e.candidates if e.kind == "policy" else None) for e in anchors]
        padded = np.concatenate([timeline, tail]) if len(tail) else timeline
        c = cfg.exec_steps
        reward_terminal = cfg.success_reward if outcome == Outcome.SUCCESS else cfg.failure_reward
        transitions: list[Transition] = []
        for i, e in enumerate(sources):
            later = anchors[i + 1:]
            if later:
                within = [j for j, x in enumerate(later) if x.step <= e.step + c]
                j = (within[-1] if within else 0) + i + 1
                nxt = anchors[j]
                n = nxt.step - e.step
            else:
                j, nxt, n = -1, None, len(timeline) - e.step
            window = padded[e.step: e.step + c]
            if len(window) < c:  # nothing planned past the end: hold the last row
                fill = window[-1:] if len(window) else e.rows[:1]
                window = np.concatenate([window, np.repeat(fill, c - len(window), 0)])
            ends_episode = nxt is None or nxt is terminal
            reward, discount = 0.0, cfg.discount ** max(n, 1)
            if ends_episode:
                reward = reward_terminal if outcome != Outcome.ABORT else 0.0
                if outcome != Outcome.ABORT:
                    discount = 0.0
                elif nxt is None:
                    continue  # truncated with no observation to bootstrap from: unusable
            transitions.append(Transition(
                src=i, next=j, actions=window.astype(np.float32), reward=float(reward),
                discount=float(discount), human_rows=int(human[e.step: e.step + c].sum())))
        if not transitions:
            return None
        return Episode(slots=slots, transitions=transitions, prompt=self.prompt, is_demo=False,
                       outcome=outcome.value,
                       info={"steps": len(timeline), "human_rows": int(human.sum()),
                             "decisions": sum(e.kind == "policy" for e in self.events)})
