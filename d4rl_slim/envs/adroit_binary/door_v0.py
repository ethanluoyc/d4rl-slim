import numpy as np
from gymnasium import utils
from gymnasium import spaces
from gymnasium.envs.mujoco import mujoco_env
from d4rl_slim.envs.adroit_binary.mujoco_utils import MujocoModelNames
import os

ADD_BONUS_REWARDS = True
ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.5,
    "azimuth": 90.0,
}


class DoorEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, reward_type="dense"):
        self.door_hinge_did = 0
        self.door_bid = 0
        self.grasp_sid = 0
        self.handle_sid = 0
        self.reward_type = reward_type

        # curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self,
            ASSETS_DIR + '/assets/DAPG_door.xml',
            5,
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(39,), dtype=np.float64),
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
        ob = self.reset_model()
        self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        self.act_rng = 0.5 * (self.model.actuator_ctrlrange[:, 1] - self.model.actuator_ctrlrange[:, 0])
        self.door_hinge_did = self.model.jnt_dofadr[self._model_names.joint_name2id['door_hinge']]
        self.grasp_sid = self._model_names.site_name2id['S_grasp']
        self.handle_sid = self._model_names.site_name2id['S_handle']
        self.door_bid = self._model_names.body_name2id['frame']

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        try:
            a = self.act_mid + a * self.act_rng  # mean center and scale
        except:
            a = a  # only for the initialization phase
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = self.data.qpos[self.door_hinge_did]

        goal_achieved = True if door_pos >= 1.35 else False

        sparse_reward = (
            10 * goal_achieved
            + 8 * (door_pos > 1.0)
            + 2 * (door_pos > 1.2)
            - 0.1 * (door_pos - 1.57) * (door_pos - 1.57)
        )
        reward = sparse_reward
        if self.reward_type == "sparse":
            reward = sparse_reward
        elif self.reward_type == "dense":
            # get to handle
            reward += -0.1 * np.linalg.norm(palm_pos - handle_pos)
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
        handle_pos = self.data.site_xpos[self.handle_sid].ravel()
        palm_pos = self.data.site_xpos[self.grasp_sid].ravel()
        door_pos = np.array([self.data.qpos[self.door_hinge_did]])
        if door_pos > 1.0:
            door_open = 1.0
        else:
            door_open = -1.0
        latch_pos = qp[-1]
        # len(qp[1:-2]) == 27
        # import ipdb; ipdb.set_trace()
        return np.concatenate(
            [qp[1:-2], [latch_pos], door_pos, palm_pos, handle_pos, palm_pos - handle_pos, [door_open]]
        )

    def reset_model(self):
        self.model.body_pos[self.door_bid, 0] = self.np_random.uniform(low=-0.3, high=-0.2)
        self.model.body_pos[self.door_bid, 1] = self.np_random.uniform(low=0.25, high=0.35)
        self.model.body_pos[self.door_bid, 2] = self.np_random.uniform(low=0.252, high=0.35)
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
        door_body_pos = self.model.body_pos[self.door_bid].ravel().copy()
        return dict(qpos=qp, qvel=qv, door_body_pos=door_body_pos)

    def set_env_state(self, state_dict):
        """
        Set the state which includes hand as well as objects and targets in the scene
        """
        self.model.body_pos[self.door_bid] = state_dict['door_body_pos']
        qp = state_dict['qpos']
        qv = state_dict['qvel']
        self.set_state(qp, qv)

    def evaluate_success(self, paths):
        num_success = 0
        num_paths = len(paths)
        # success if door open for 25 steps
        for path in paths:
            if np.sum(path['env_infos']['goal_achieved']) > 25:
                num_success += 1
        success_percentage = num_success * 100.0 / num_paths
        return success_percentage
