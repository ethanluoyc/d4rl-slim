from gymnasium.envs import registration

from d4rl_slim.dataset import get_dataset
from d4rl_slim.envs import get_environment
from d4rl_slim.envs.antmaze import maps
from d4rl_slim.infos import get_normalized_score
from d4rl_slim.infos import list_datasets
from d4rl_slim.tfds import get_tfds_name


def register(id, **kwargs):
    # Register our environments under d4rl_slim namespace.
    namedspaced_id = "d4rl_slim/" + id
    registration.register(
        id=namedspaced_id,
        order_enforce=False,
        **kwargs,
    )


# Antmaze
register(
    id="antmaze-umaze-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "maze_map": maps.U_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-umaze-diverse-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "maze_map": maps.U_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-medium-play-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.BIG_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-medium-diverse-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.BIG_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-large-diverse-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.HARDEST_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-large-play-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "deprecated": True,
        "maze_map": maps.HARDEST_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-umaze-v1",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "maze_map": maps.U_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-umaze-diverse-v1",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "deprecated": True,
        "maze_map": maps.U_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-medium-play-v1",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.BIG_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-medium-diverse-v1",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.BIG_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-large-diverse-v1",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.HARDEST_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-large-play-v1",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "deprecated": True,
        "maze_map": maps.HARDEST_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-eval-umaze-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "maze_map": maps.U_MAZE_EVAL_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-eval-umaze-diverse-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "maze_map": maps.U_MAZE_EVAL_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-eval-medium-play-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.BIG_MAZE_EVAL_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-eval-medium-diverse-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.BIG_MAZE_EVAL_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-eval-large-diverse-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.HARDEST_MAZE_EVAL_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)

register(
    id="antmaze-eval-large-play-v0",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.HARDEST_MAZE_EVAL_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
    },
)


register(
    id="antmaze-umaze-v2",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "maze_map": maps.U_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
        "v2_resets": True,
    },
)

register(
    id="antmaze-umaze-diverse-v2",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=700,
    kwargs={
        "maze_map": maps.U_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
        "v2_resets": True,
    },
)

register(
    id='antmaze-medium-play-v2',
    entry_point='d4rl_slim.envs.antmaze.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maps.BIG_MAZE_TEST,
        'reward_type': 'sparse',
        'non_zero_reset': False,
        'eval': True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    },
)

register(
    id="antmaze-medium-diverse-v2",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.BIG_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
        "v2_resets": True,
    },
)

register(
    id="antmaze-large-diverse-v2",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.HARDEST_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
        "v2_resets": True,
    },
)

register(
    id="antmaze-large-play-v2",
    entry_point="d4rl_slim.envs.antmaze.ant:make_ant_maze_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maps.HARDEST_MAZE_TEST,
        "reward_type": "sparse",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
        "v2_resets": True,
    },
)

# Adroit binary
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

__all__ = [
    "get_dataset",
    "list_datasets",
    "get_normalized_score",
    "get_environment",
    "get_tfds_name"
]
