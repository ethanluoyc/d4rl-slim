import numpy as np
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env
from d4rl_slim.envs.adroit_binary.quatmath import quat2euler
from d4rl_slim.envs.adroit_binary.mujoco_utils import MujocoModelNames
import os

ADD_BONUS_REWARDS = True
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

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(46,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            self,
            ASSETS_DIR + '/assets/DAPG_hammer.xml',
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=self.action_space.dtype)

        self._model_names = MujocoModelNames(self.model)
        utils.EzPickle.__init__(self)

        # change actuator sensitivity
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id['A_WRJ1'] : self._model_names.actuator_name2id['A_WRJ0'] + 1, :3
        ] = np.array([10, 0, 0])
        self.model.actuator_gainprm[
            self._model_names.actuator_name2id['A_FFJ3'] : self._model_names.actuator_name2id['A_THJ0'] + 1, :3
        ] = np.array([1, 0, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id['A_WRJ1'] : self._model_names.actuator_name2id['A_WRJ0'] + 1, :3
        ] = np.array([0, -10, 0])
        self.model.actuator_biasprm[
            self._model_names.actuator_name2id['A_FFJ3'] : self._model_names.actuator_name2id['A_THJ0'] + 1, :3
        ] = np.array([0, -1, 0])

        self.target_obj_sid = self._model_names.site_name2id['S_target']
        self.S_grasp_sid = self._model_names.site_name2id['S_grasp']
        self.obj_bid = self._model_names.body_name2id['Object']
        self.tool_sid = self._model_names.site_name2id['tool']
        self.goal_sid = self._model_names.site_name2id['nail_goal']
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

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

        sparse_reward = (
            75 * goal_achieved
            + 25 * (np.linalg.norm(target_pos - goal_pos) < 0.020)
            - 10 * np.linalg.norm(target_pos - goal_pos)
        )  # make nail go inside
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            # get to hammer
            reward -= -0.1 * np.linalg.norm(palm_pos - obj_pos)
            # velocity penalty
            reward -= 1e-2 * np.linalg.norm(self.data.qvel.ravel())
            # bonus for lifting up the hammer
            if obj_pos[2] > 0.04 and tool_pos[2] > 0.04:
                reward += 2
        elif self.reward_type == "binary":
            reward = goal_achieved - 1
        else:
            raise ValueError("Invalid reward type.")

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
        nail_impact = np.clip(self.data.sensordata[self._model_names.sensor_name2id['S_nail']], -1.0, 1.0)
        return np.concatenate([qp[:-6], qv[-6:], palm_pos, obj_pos, obj_rot, target_pos, np.array([nail_impact])])

    def reset_model(self):
        # self.sim.reset()
        target_bid = self._model_names.body_name2id['nail_board']
        self.model.body_pos[target_bid, 2] = self.np_random.uniform(low=0.1, high=0.25)
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        board_pos = self.model.body_pos[self._model_names.body_name2id['nail_board']].copy()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel().copy()
        return dict(qpos=qpos, qvel=qvel, board_pos=board_pos, target_pos=target_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        board_pos = state_dict['board_pos']
        self.model.body_pos[self._model_names.body_name2id['nail_board']] = board_pos
        self.set_state(qp, qv)

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if nail insude board for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
