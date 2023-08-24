import unittest

import d4rl_slim


class InfoTest(unittest.TestCase):
    def test_get_normalized_score(self):
        d4rl_slim.get_normalized_score("antmaze-medium-play-v2", 0.0)


if __name__ == "__main__":
    unittest.main()
