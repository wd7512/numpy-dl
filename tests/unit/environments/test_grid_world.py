"""Tests for GridWorld environment."""

from numpy_dl.environments.grid_world import GridWorld


class TestGridWorldReset:
    def test_reset_returns_valid_state(self) -> None:
        env = GridWorld(n=4)
        s = env.reset()
        assert 0 <= s < env.n_states

    def test_reset_specific_state(self) -> None:
        env = GridWorld(n=4)
        s = env.reset(state=5)
        assert s == 5


class TestGridWorldStep:
    def test_step_returns_valid_transition(self) -> None:
        env = GridWorld(n=4, seed=0)
        env.reset(state=0)
        ns, r, done = env.step(2)
        assert 0 <= ns < env.n_states
        assert isinstance(r, float)
        assert isinstance(done, bool)

    def test_wall_blocks_movement(self) -> None:
        env = GridWorld(n=4)
        env.reset(state=0)
        ns, _, _ = env.step(0)
        assert ns == 0

    def test_episode_terminates_at_goal(self) -> None:
        env = GridWorld(n=4, goal=5)
        env.reset(state=1)
        ns, r, done = env.step(2)
        assert ns == 5
        assert r == 10.0
        assert done is True

    def test_negative_reward_per_step(self) -> None:
        env = GridWorld(n=4, goal=15)
        env.reset(state=0)
        ns, r, done = env.step(1)
        assert r == -1.0
        assert done is False
