import unittest

import d4rl  # noqa: F401
import d4rl.gym_mujoco  # noqa: F401
import gym
import pytest

import d4rl_slim  # noqa: F401


def assert_same_space(a, b):
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert (a.low == b.low).all()
    assert (a.high == b.high).all()


@pytest.mark.parametrize(
    "d4rl_name",
    [
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        "walker2d-medium-v2",
        "ant-medium-v2",
    ],
)
def test_same_space(d4rl_name):
    d4rl_env = gym.make(d4rl_name)
    gymnasium_env = d4rl_slim.get_environment(d4rl_name).unwrapped
    assert_same_space(d4rl_env.action_space, gymnasium_env.action_space)
    assert_same_space(d4rl_env.observation_space, gymnasium_env.observation_space)


if __name__ == '__main__':
    unittest.main()
