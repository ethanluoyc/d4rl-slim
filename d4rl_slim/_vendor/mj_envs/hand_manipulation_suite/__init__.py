from gym.envs.registration import register
from mj_envs.mujoco_env import MujocoEnv

# Swing the door open
register(
    id='door-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
)
register(
    id='door-sparse-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='door-binary-v0',
    entry_point='mj_envs.hand_manipulation_suite:DoorEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="binary",
    ),
)
from mj_envs.hand_manipulation_suite.door_v0 import DoorEnvV0

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
)
register(
    id='hammer-sparse-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='hammer-binary-v0',
    entry_point='mj_envs.hand_manipulation_suite:HammerEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="binary",
    ),
)
from mj_envs.hand_manipulation_suite.hammer_v0 import HammerEnvV0

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
    kwargs=dict(
        early_termination=True,
    ),
)
register(
    id='pen-notermination-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
)
register(
    id='pen-sparse-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='pen-binary-v0',
    entry_point='mj_envs.hand_manipulation_suite:PenEnvV0',
    max_episode_steps=100,
    kwargs=dict(
        reward_type="binary",
    ),
)
from mj_envs.hand_manipulation_suite.pen_v0 import PenEnvV0

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
)
register(
    id='relocate-sparse-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='relocate-binary-v0',
    entry_point='mj_envs.hand_manipulation_suite:RelocateEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="binary",
    ),
)
from mj_envs.hand_manipulation_suite.relocate_v0 import RelocateEnvV0
