from gymnasium.envs import registration


def register(id, **kwargs):
    namedspaced_id = "d4rl_slim/" + id
    registration.register(
        id=namedspaced_id,
        order_enforce=False,
        apply_api_compatibility=False,
        **kwargs,
    )


register(
    id='door-v0',
    entry_point='d4rl_slim.envs.adroit_binary.door_v0:DoorEnvV0',
    max_episode_steps=200,
)
register(
    id='door-sparse-v0',
    entry_point='d4rl_slim.envs.adroit_binary.door_v0:DoorEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='door-binary-v0',
    entry_point='d4rl_slim.envs.adroit_binary.door_v0:DoorEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="binary",
    ),
)

# Hammer a nail into the board
register(
    id='hammer-v0',
    entry_point='d4rl_slim.envs.adroit_binary.hammer_v0:HammerEnvV0',
    max_episode_steps=200,
)
register(
    id='hammer-sparse-v0',
    entry_point='d4rl_slim.envs.adroit_binary.hammer_v0:HammerEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='hammer-binary-v0',
    entry_point='d4rl_slim.envs.adroit_binary.hammer_v0:HammerEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="binary",
    ),
)

# Reposition a pen in hand
register(
    id='pen-v0',
    entry_point='d4rl_slim.envs.adroit_binary.pen_v0:PenEnvV0',
    max_episode_steps=100,
    kwargs=dict(
        early_termination=True,
    ),
)
register(
    id='pen-notermination-v0',
    entry_point='d4rl_slim.envs.adroit_binary.pen_v0:PenEnvV0',
    max_episode_steps=100,
)
register(
    id='pen-sparse-v0',
    entry_point='d4rl_slim.envs.adroit_binary.pen_v0:PenEnvV0',
    max_episode_steps=100,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='pen-binary-v0',
    entry_point='d4rl_slim.envs.adroit_binary.pen_v0:PenEnvV0',
    max_episode_steps=100,
    kwargs=dict(
        reward_type="binary",
    ),
)

# Relcoate an object to the target
register(
    id='relocate-v0',
    entry_point='d4rl_slim.envs.adroit_binary.relocate_v0:RelocateEnvV0',
    max_episode_steps=200,
)
register(
    id='relocate-sparse-v0',
    entry_point='d4rl_slim.envs.adroit_binary.relocate_v0:RelocateEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="sparse",
    ),
)
register(
    id='relocate-binary-v0',
    entry_point='d4rl_slim.envs.adroit_binary.relocate_v0:RelocateEnvV0',
    max_episode_steps=200,
    kwargs=dict(
        reward_type="binary",
    ),
)
