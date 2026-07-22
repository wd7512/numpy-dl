"""Tests for FrozenLake environment."""

from numpy_dl.environments.frozen_lake import FrozenLake


class TestFrozenLakeReset:
    def test_reset_returns_valid_state(self) -> None:
        env = FrozenLake(seed=0)
        s = env.reset()
        assert 0 <= s < env.n_states

    def test_reset_specific_state(self) -> None:
        env = FrozenLake(seed=0)
        s = env.reset(state=7)
        assert s == 7


class TestFrozenLakeStep:
    def test_action_space(self) -> None:
        env = FrozenLake(is_slippery=False, seed=0)
        for a in range(env.n_actions):
            env.reset(state=0)
            ns, r, done = env.step(a)
            assert 0 <= ns < env.n_states
            assert isinstance(r, float)
            assert isinstance(done, bool)

    def test_step_returns_tuple(self) -> None:
        env = FrozenLake(seed=0)
        env.reset()
        result = env.step(0)
        assert len(result) == 3

    def test_reaching_goal_gives_reward(self) -> None:
        env = FrozenLake(is_slippery=False, seed=0)
        env.reset(state=14)
        ns, r, done = env.step(2)
        assert ns == 15
        assert r == 1.0
        assert done is True
