"""Tests for ReplayBuffer."""

import numpy as np

from numpy_dl.memory.replay_buffer import ReplayBuffer


class TestReplayBufferPush:
    def test_push_increases_length(self) -> None:
        buf = ReplayBuffer(capacity=10)
        s = np.array([1.0, 2.0])
        buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 1

    def test_push_up_to_capacity(self) -> None:
        buf = ReplayBuffer(capacity=5)
        s = np.zeros(2)
        for i in range(5):
            buf.push(s, i, float(i), s, False)
        assert len(buf) == 5

    def test_push_overwrites_when_full(self) -> None:
        buf = ReplayBuffer(capacity=3)
        s = np.zeros(2)
        for i in range(5):
            buf.push(s, i, float(i), s, False)
        assert len(buf) == 3


class TestReplayBufferSample:
    def test_sample_returns_batch(self) -> None:
        buf = ReplayBuffer(capacity=100)
        s = np.array([1.0, 2.0])
        for i in range(50):
            buf.push(s * i, i % 4, float(i), s * (i + 1), i == 49)
        states, actions, rewards, next_states, dones = buf.sample(16)
        assert states.shape == (16, 2)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)
        assert next_states.shape == (16, 2)
        assert dones.shape == (16,)

    def test_sample_actions_valid(self) -> None:
        buf = ReplayBuffer(capacity=100)
        s = np.zeros(2)
        for i in range(20):
            buf.push(s, i % 3, 1.0, s, False)
        _, actions, _, _, _ = buf.sample(10)
        assert all(0 <= a < 3 for a in actions)

    def test_sample_no_duplicates(self) -> None:
        buf = ReplayBuffer(capacity=100)
        for i in range(20):
            s = np.array([float(i), float(i)])
            buf.push(s, i % 3, float(i), s, False)
        states, _, _, _, _ = buf.sample(20)
        assert len(set(map(tuple, states))) == 20
