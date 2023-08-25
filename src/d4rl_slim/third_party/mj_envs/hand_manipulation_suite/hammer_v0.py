import numpy as np
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env
from ..utils.quatmath import *
from .mujoco_utils import actuator_name2id, joint_name2id, site_name2id, body_name2id, sensor_name2id
import mujoco
import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 45.0,
}

class HammerEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type="dense"):
        self.target_obj_sid = -1
        self.S_grasp_sid = -1
        self.obj_bid = -1
        self.tool_sid = -1
        self.goal_sid = -1

        self.reward_type = reward_type

        mujoco_env.MujocoEnv.__init__(
            self,
            ASSETS_DIR + '/assets/DAPG_hammer.xml',
            5,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(46,), dtype=np.float64),
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )
        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.model.actuator_gainprm[actuator_name2id(self.model, 'A_WRJ1'):actuator_name2id(self.model, 'A_WRJ0')+1,:3] = np.array([10, 0, 0])
        self.model.actuator_gainprm[actuator_name2id(self.model, 'A_FFJ3'):actuator_name2id(self.model, 'A_THJ0')+1,:3] = np.array([1, 0, 0])
        self.model.actuator_biasprm[actuator_name2id(self.model, 'A_WRJ1'):actuator_name2id(self.model, 'A_WRJ0')+1,:3] = np.array([0, -10, 0])
        self.model.actuator_biasprm[actuator_name2id(self.model, 'A_FFJ3'):actuator_name2id(self.model, 'A_THJ0')+1,:3] = np.array([0, -1, 0])

        self.target_obj_sid = site_name2id(self.model, 'S_target')
        self.S_grasp_sid = site_name2id(self.model, 'S_grasp')
        self.obj_bid = body_name2id(self.model, 'Object')
        self.tool_sid = site_name2id(self.model, 'tool')
        self.goal_sid = site_name2id(self.model, 'nail_goal')
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.action_space = spaces.Box(
            low=-1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0]),
            high=np.ones_like(self.model.actuator_ctrlrange[:,1]), dtype=np.float32
        )


    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        obj_pos = self.data.xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        tool_pos = self.data.site_xpos[self.tool_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        goal_pos = self.data.site_xpos[self.goal_sid].ravel()

        goal_achieved = True if np.linalg.norm(target_pos - goal_pos) < 0.010 else False

        sparse_reward = 75 * goal_achieved + \
            25 * (np.linalg.norm(target_pos - goal_pos) < 0.020) - \
            10 * np.linalg.norm(target_pos - goal_pos) # make nail go inside
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            # get to hammer
            reward -= - 0.1 * np.linalg.norm(palm_pos - obj_pos)
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2
        elif self.reward_type == "binary":
            reward = goal_achieved - 1
        else:
            raise ValueError

        # take hammer head to nail # does not seem to be in the reward fn
        # reward -= np.linalg.norm((tool_pos - target_pos))

        return ob, reward, False, False, dict(goal_achieved=goal_achieved)

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        qv = np.clip(self.data.qvel.ravel(), -1.0, 1.0)
        obj_pos = self.data.xpos[self.obj_bid].ravel()
        obj_rot = quat2euler(self.data.xquat[self.obj_bid].ravel()).ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        nail_impact = np.clip(self.data.sensordata[sensor_name2id(self.model, 'S_nail')], -1.0, 1.0)
        return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])

    def reset_model(self):
        # self.sim.reset()
        mujoco.mj_resetData(self.model, self.data)
        target_bid = body_name2id(self.model, 'nail_board')
        self.model.body_pos[target_bid,2] = self.np_random.uniform(low=0.1, high=0.25)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[body_name2id(self.model, 'nail_board')].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.set_state(qp, qv)
        self.model.body_pos[body_name2id(self.model, 'nail_board')] = board_pos
        mujoco.mj_forward(self.model, self.data)

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if nail insude board for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success*100.0/num_paths
        return success_percentage
