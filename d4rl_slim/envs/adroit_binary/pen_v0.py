import numpy as np
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env
from d4rl_slim.envs.adroit_binary.quatmath import euler2quat
from d4rl_slim.envs.adroit_binary.mujoco_utils import MujocoModelNames
import os

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.0,
    "azimuth": -45.0,
}


class PenEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type="dense", early_termination=False):
        self.target_obj_bid = 0
        self.S_grasp_sid = 0
        self.eps_ball_sid = 0
        self.obj_bid = 0
        self.obj_t_sid = 0
        self.obj_b_sid = 0
        self.tar_t_sid = 0
        self.tar_b_sid = 0
        self.pen_length = 1.0
        self.tar_length = 1.0

        self.reward_type = reward_type
        self.early_termination = early_termination

        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        # mujoco_env.MujocoEnv.__init__(self, curr_dir+'/assets/DAPG_pen.xml', 5)
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(45,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            self,
            ASSETS_DIR + '/assets/DAPG_pen.xml',
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

        utils.EzPickle.__init__(self)
        self.target_obj_bid = self._model_names.body_name2id["target"]
        self.S_grasp_sid = self._model_names.site_name2id['S_grasp']
        self.obj_bid = self._model_names.body_name2id['Object']
        self.eps_ball_sid = self._model_names.site_name2id['eps_ball']
        self.obj_t_sid = self._model_names.site_name2id['object_top']
        self.obj_b_sid = self._model_names.site_name2id['object_bottom']
        self.tar_t_sid = self._model_names.site_name2id['target_top']
        self.tar_b_sid = self._model_names.site_name2id['target_bottom']

        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            starting_up = False
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            starting_up = True
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)

        obj_pos = self.data.xpos[self.obj_bid].ravel()
        desired_loc = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid]) / self.tar_length

        # pos cost
        dist = np.linalg.norm(obj_pos - desired_loc)
        # orien cost
        orien_similarity = np.dot(obj_orien, desired_orien)

        goal_achieved = True if (dist < 0.075 and orien_similarity > 0.95) else False

        sparse_reward = 50 * goal_achieved
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            reward += -dist
            reward += orien_similarity
            # bonus for being close to desired orientation
            if orien_similarity > 0.9:
                reward += 10
            if obj_pos[2] < 0.075:
                reward -= 5
        elif self.reward_type == "binary":
            reward = goal_achieved - 1
        else:
            error

        # penalty for dropping the pen
        done = False
        if self.early_termination:
            done = True if not starting_up else False

        return self._get_obs(), reward, done, False, dict(goal_achieved=goal_achieved)

    def _get_obs(self):
        qp = self.data.qpos.ravel()
        obj_vel = self.data.qvel[-6:].ravel()
        obj_pos = self.data.xpos[self.obj_bid].ravel()
        desired_pos = self.data.site_xpos[self.eps_ball_sid].ravel()
        obj_orien = (self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid]) / self.pen_length
        desired_orien = (self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid]) / self.tar_length
        return np.concatenate(
            [qp[:-6], obj_pos, obj_vel, obj_orien, desired_orien, obj_pos - desired_pos, obj_orien - desired_orien]
        )

    def reset_model(self):
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(low=-1, high=1)
        desired_orien[1] = self.np_random.uniform(low=-1, high=1)
        self.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        self.set_state(qp, qv)

        self.pen_length = np.linalg.norm(self.data.site_xpos[self.obj_t_sid] - self.data.site_xpos[self.obj_b_sid])
        self.tar_length = np.linalg.norm(self.data.site_xpos[self.tar_t_sid] - self.data.site_xpos[self.tar_b_sid])

        return self._get_obs()

    def get_env_state(self):
        """
        Get state of hand as well as objects and targets in the scene
        """
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()
        desired_orien = self.model.body_quat[self.target_obj_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, desired_orien=desired_orien)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        desired_orien = state_dict['desired_orien']
        self.model.body_quat[self.target_obj_bid] = desired_orien
        self.set_state(qp, qv)

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if pen within 15 degrees of target for 20 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 20:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
