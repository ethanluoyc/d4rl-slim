import unittest

from gymnasium.utils import env_checker

import d4rl_slim  # noqa: F401


class AntmazeTest(unittest.TestCase):
    # @unittest.skip("Skip due to weird seeding issue")
    def test_antmaze(self):
        env = d4rl_slim.get_environment("antmaze-umaze-v2").unwrapped
        env.reset()
        env.step(env.action_space.sample())
        env_checker.check_env(env, skip_render_check=True)


if __name__ == '__main__':
    unittest.main()
