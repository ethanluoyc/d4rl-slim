import os
import numpy as np
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env
from d4rl_slim.envs.adroit_binary.mujoco_utils import MujocoModelNames

ADD_BONUS_REWARDS = True
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}


class RelocateEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type="dense"):
        self.target_obj_sid = 0
        self.S_grasp_sid = 0
        self.obj_bid = 0
        self.reward_type = reward_type

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            self,
            ASSETS_DIR + '/assets/DAPG_relocate.xml',
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=self.action_space.shape, dtype=self.action_space.dtype)
        self._model_names = MujocoModelNames(self.model)

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

        self.target_obj_sid = self._model_names.site_name2id["target"]
        self.S_grasp_sid = self._model_names.site_name2id['S_grasp']
        self.obj_bid = self._model_names.body_name2id['Object']
        utils.EzPickle.__init__(self)
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
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()

        goal_achieved = True if np.linalg.norm(obj_pos - target_pos) < 0.1 else False

        sparse_reward = 20 * goal_achieved + 10 * (np.linalg.norm(obj_pos - target_pos) < 0.1)
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            reward += -0.1 * np.linalg.norm(palm_pos - obj_pos)  # take hand to object
            if obj_pos[2] > 0.04:  # if object off the table
                reward += 1.0  # bonus for lifting the object
                reward += -0.5 * np.linalg.norm(palm_pos - target_pos)  # make hand go to target
                reward += -0.5 * np.linalg.norm(obj_pos - target_pos)  # make object go to target
        elif self.reward_type == "binary":
            reward = goal_achieved - 1
        else:
            error

        return ob, reward, False, False, dict(goal_achieved=goal_achieved)

    def _get_obs(self):
        # qpos for hand
        # xpos for obj
        # xpos for target
        qp = self.data.qpos.ravel()
        obj_pos = self.data.xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return np.concatenate([qp[:-6], palm_pos - obj_pos, palm_pos - target_pos, obj_pos - target_pos])

    def reset_model(self):
        self.model.body_pos[self.obj_bid, 0] = self.np_random.uniform(low=-0.15, high=0.15)
        self.model.body_pos[self.obj_bid, 1] = self.np_random.uniform(low=-0.15, high=0.3)
        self.model.site_pos[self.target_obj_sid, 0] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid, 1] = self.np_random.uniform(low=-0.2, high=0.2)
        self.model.site_pos[self.target_obj_sid, 2] = self.np_random.uniform(low=0.15, high=0.35)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)
        return self._get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        hand_qpos = qp[:30]
        obj_pos = self.data.xpos[self.obj_bid].ravel()
        palm_pos = self.data.site_xpos[self.S_grasp_sid].ravel()
        target_pos = self.data.site_xpos[self.target_obj_sid].ravel()
        return dict(hand_qpos=hand_qpos, obj_pos=obj_pos, target_pos=target_pos, palm_pos=palm_pos, qpos=qp, qvel=qv)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        obj_pos = state_dict['obj_pos']
        target_pos = state_dict['target_pos']
        self.model.body_pos[self.obj_bid] = obj_pos
        self.model.site_pos[self.target_obj_sid] = target_pos
        self.set_state(qp, qv)

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if object close to target for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
