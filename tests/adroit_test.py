import gym
import gym
import d4rl_slim._vendor.mj_envs
import unittest
import numpy as np

class AdroitTest(unittest.TestCase):
    def test_door(self):
        np.random.seed(0)
        env = gym.make('door-v0')
        env.reset()
        env.step(env.action_space.sample())

    def test_hammer(self):
        np.random.seed(0)
        env = gym.make('door-v0')
        env.reset()
        env.step(env.action_space.sample())

    def test_relocate(self):
        np.random.seed(0)
        env = gym.make('relocate-v0')
        env.reset()
        env.step(env.action_space.sample())

    def test_pen(self):
        np.random.seed(0)
        env = gym.make('pen-v0')
        env.reset()
        env.step(env.action_space.sample())

if __name__ == '__main__':
    unittest.main()
