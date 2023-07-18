import unittest

import gymnasium
from gymnasium.utils import env_checker

import d4rl_slim  # noqa: F401


class AdroitTest(unittest.TestCase):
    def test_door_v0(self):
        env = gymnasium.make("d4rl_slim/door-v0").unwrapped
        env.reset()
        state = env.get_env_state()
        env.set_env_state(state)
        env_checker.check_env(env, skip_render_check=True)

    def test_hammer_v0(self):
        env = gymnasium.make("d4rl_slim/hammer-v0").unwrapped
        env.reset()
        env_checker.check_env(env, skip_render_check=True)

    def test_pen_v0(self):
        env = gymnasium.make("d4rl_slim/pen-v0").unwrapped
        env.reset()
        state = env.get_env_state()
        env.set_env_state(state)
        env_checker.check_env(env, skip_render_check=True)

    def test_relocate_v0(self):
        env = gymnasium.make("d4rl_slim/relocate-v0").unwrapped
        print(env.reset()[0].shape)
        state = env.get_env_state()
        env.set_env_state(state)
        env_checker.check_env(env, skip_render_check=True)


if __name__ == '__main__':
    unittest.main()
