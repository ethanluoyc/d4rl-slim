import d4rl_slim._vendor.mj_envs
import unittest
import numpy as np
import gymnasium
from gymnasium.utils import env_checker

class AdroitTest(unittest.TestCase):
    def test_door_v1(self):
        env = gymnasium.make('d4rl_slim/door-v0').unwrapped
        env_checker.check_env(env, skip_render_check=True)
        state = env.get_env_state()
        print(env.action_space)
        env.set_env_state(state)

    def test_hammer_v1(self):
        env = gymnasium.make('d4rl_slim/hammer-v0').unwrapped
        env_checker.check_env(env, skip_render_check=True)
        state = env.get_env_state()
        env.set_env_state(state)

    def test_pen_v1(self):
        env = gymnasium.make('d4rl_slim/pen-v0').unwrapped
        env_checker.check_env(env, skip_render_check=True)
        state = env.get_env_state()
        env.set_env_state(state)

    def test_relocate_v1(self):
        env = gymnasium.make('d4rl_slim/relocate-v0').unwrapped
        env_checker.check_env(env, skip_render_check=True)
        state = env.get_env_state()
        env.set_env_state(state)

if __name__ == '__main__':
    unittest.main()
