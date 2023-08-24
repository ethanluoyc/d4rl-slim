from gymnasium.envs.registration import register as _register
from d4rl_slim._vendor.d4rl.locomotion import maze_env

def register(*, id, entry_point, max_episode_steps, kwargs):
    _register(f"d4rl_slim/{id}", entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs)

register(
    id='antmaze-umaze-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-umaze-diverse-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-medium-play-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-medium-diverse-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-large-diverse-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-large-play-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-umaze-v1',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-umaze-diverse-v1',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-medium-play-v1',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-medium-diverse-v1',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-large-diverse-v1',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-large-play-v1',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-eval-umaze-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-eval-umaze-diverse-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-eval-medium-play-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-eval-medium-diverse-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-eval-large-diverse-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)

register(
    id='antmaze-eval-large-play-v0',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_EVAL_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
    }
)


register(
    id='antmaze-umaze-v2',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-umaze-diverse-v2',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=700,
    kwargs={
        'maze_map': maze_env.U_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-medium-play-v2',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-medium-diverse-v2',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.BIG_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-large-diverse-v2',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    }
)

register(
    id='antmaze-large-play-v2',
    entry_point='d4rl_slim._vendor.d4rl.locomotion.ant:make_ant_maze_env',
    max_episode_steps=1000,
    kwargs={
        'maze_map': maze_env.HARDEST_MAZE_TEST,
        'reward_type':'sparse',
        'non_zero_reset':False, 
        'eval':True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    }
)