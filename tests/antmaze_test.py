from d4rl_slim._vendor.d4rl.locomotion import ant
from d4rl_slim._vendor.d4rl.locomotion import maze_env
import unittest

class AntmazeTest(unittest.TestCase):
    def test_antmaze(self):
        env = ant.make_ant_maze_env(**{
            'maze_map': maze_env.HARDEST_MAZE_TEST,
            'reward_type':'sparse',
            'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5',
            'non_zero_reset':False, 
            'eval':True,
            'maze_size_scaling': 4.0,
            'ref_min_score': 0.0,
            'ref_max_score': 1.0,
            'v2_resets': True,
        })
        env.reset()
        env.step(env.action_space.sample())

if __name__ == '__main__':
    unittest.main()
