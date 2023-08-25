import gymnasium
from gymnasium.utils import env_checker
import unittest
import numpy as np

class AntmazeTest(unittest.TestCase):
    def test_antmaze_large_play(self):
        np.random.seed(0)
        env = gymnasium.make('d4rl_slim/antmaze-large-play-v2')
        env_checker.check_env(env.unwrapped, skip_render_check=True)

if __name__ == '__main__':
    unittest.main()
