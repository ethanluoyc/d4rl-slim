import os

import gymnasium
import numpy as np
import pytest
from gymnasium.utils import env_checker


class TestAdroit:
    @pytest.mark.parametrize(
        "env_name",
        [
            "door-v0",
            "hammer-v0",
            "pen-v0",
            "relocate-v0",
        ],
    )
    def test_env(self, env_name):
        env = gymnasium.make(f"d4rl_slim/{env_name}").unwrapped
        env_checker.check_env(env, skip_render_check=True)
        state = env.get_env_state()
        env.set_env_state(state)
        assert (env.action_space.low == -1.0).all()
        assert (env.action_space.high == 1.0).all()


class TestAntmaze:
    @pytest.mark.parametrize(
        "env_name",
        [
            "antmaze-large-play-v2",
        ],
    )
    def test_env(self, env_name):
        env = gymnasium.make(f"d4rl_slim/{env_name}")
        env_checker.check_env(env.unwrapped, skip_render_check=True)
        assert (env.action_space.low == -1.0).all()
        assert (env.action_space.high == 1.0).all()

    def test_antmaze_seeding(self):
        env_name = "d4rl_slim/antmaze-large-play-v2"
        env = gymnasium.make(env_name)
        seed = 0
        obs0, _ = env.reset(seed=seed)
        env = gymnasium.make(env_name)
        obs1, _ = env.reset(seed=seed)

        np.testing.assert_allclose(obs0, obs1)

    def test_antmaze_closes_remove_file(self):
        env_name = "d4rl_slim/antmaze-large-play-v2"
        env = gymnasium.make(env_name)
        assert os.path.exists(env.unwrapped._maze_file_path)
        env.close()
        assert not os.path.exists(env.unwrapped._maze_file_path)
