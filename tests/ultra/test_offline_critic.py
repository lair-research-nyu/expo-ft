import importlib.util
from pathlib import Path

import numpy as np

from expo_ft.ultra.buffer import Episode, Slot, Transition

_spec = importlib.util.spec_from_file_location(
    "expo_offline_critic", Path(__file__).resolve().parents[2] / "scripts" / "expo_offline_critic.py")
oc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc)


def _episode(n, reward, gamma=0.9, outcome="success"):
    slots = [Slot(images=np.zeros((1, 2, 2, 3), np.uint8), state=np.zeros(2), state_robot=np.zeros(2))
             for _ in range(n)]
    trans = [Transition(src=i, next=i + 1 if i < n - 1 else -1, actions=np.zeros((1, 2)),
                        reward=reward if i == n - 1 else 0.0, discount=gamma if i < n - 1 else 0.0)
             for i in range(n)]
    return Episode(slots=slots, transitions=trans, outcome=outcome)


def test_mc_returns_discount_the_terminal_reward_back():
    np.testing.assert_allclose(oc.mc_returns(_episode(4, 1.0)), [0.9**3, 0.9**2, 0.9, 1.0], rtol=1e-6)
    np.testing.assert_allclose(oc.mc_returns(_episode(3, 0.0)), 0.0)


def test_auc():
    assert oc.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert oc.auc(np.array([0.0]), np.array([1.0])) == 0.0
    assert oc.auc(np.array([1.0, 1.0]), np.array([1.0])) == 0.5


def test_holdout_keeps_each_outcome_in_training():
    eps = [_episode(2, 1.0, outcome="success") for _ in range(5)] + \
          [_episode(2, 0.0, outcome="failure") for _ in range(8)]
    train, hold = oc.split_holdout(eps, 0.25, seed=0)
    assert sorted(train + hold) == list(range(13)) and not set(train) & set(hold)
    for outcome in ("success", "failure"):
        assert any(eps[i].outcome == outcome for i in hold)
        assert any(eps[i].outcome == outcome for i in train)
    assert oc.split_holdout(eps, 0.0, seed=0)[1] == []
