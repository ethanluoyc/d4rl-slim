"""Wrapper for creating the ant environment."""

import os
import atexit
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy

import gymnasium as gym
import numpy as np
from gymnasium import utils
from gymnasium import wrappers
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box

from d4rl_slim.envs.antmaze.maps import GOAL
from d4rl_slim.envs.antmaze.maps import RESET

GYM_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")


class AntEnv(MujocoEnv, utils.EzPickle):
    """Basic ant locomotion environment."""

    FILE = os.path.join(GYM_ASSETS_DIR, "ant.xml")
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
        self,
        file_path=None,
        expose_all_qpos=False,
        expose_body_coms=None,
        expose_body_comvels=None,
        non_zero_reset=False,
    ):
        if non_zero_reset:
            raise NotImplementedError("Non-zero reset not implemented for AntEnv")
        if file_path is None:
            file_path = self.FILE
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}

        self._non_zero_reset = non_zero_reset

        obs_shape = 29
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64)

        MujocoEnv.__init__(self, file_path, 5, observation_space=observation_space)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            False,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        # No cfrc observation.
        if self._expose_all_qpos:
            obs = np.concatenate(
                [
                    self.data.qpos.flat[:15],  # Ensures only ant obs.
                    self.data.qvel.flat[:14],
                ]
            )
        else:
            obs = np.concatenate(
                [
                    self.data.qpos.flat[2:15],
                    self.data.qvel.flat[:14],
                ]
            )

        if self._expose_body_coms is not None:
            for name in self._expose_body_coms:
                com = self.get_body_com(name)
                if name not in self._body_com_indices:
                    indices = range(len(obs), len(obs) + len(com))
                    self._body_com_indices[name] = indices
                obs = np.concatenate([obs, com])

        if self._expose_body_comvels is not None:
            for name in self._expose_body_comvels:
                comvel = self.get_body_comvel(name)
                if name not in self._body_comvel_indices:
                    indices = range(len(obs), len(obs) + len(comvel))
                    self._body_comvel_indices[name] = indices
                obs = np.concatenate([obs, comvel])
        return obs

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.1, high=0.1)
        qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1

        # if self._non_zero_reset:
        #     """Now the reset is supposed to be to a non-zero location"""
        #     reset_location = self._get_reset_location()
        #     qpos[:2] = reset_location

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_xy(self):
        return self.data.qpos[:2]

    def set_xy(self, xy):
        qpos = np.copy(self.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]
        qvel = self.data.qvel
        self.set_state(qpos, qvel)


def disk_goal_sampler(np_random, goal_region_radius=10.0):
    th = 2 * np.pi * np_random.uniform()
    radius = goal_region_radius * np_random.uniform()
    return radius * np.array([np.cos(th), np.sin(th)])


def constant_goal_sampler(np_random, location=10.0 * np.ones([2])):
    return location


class AntMaze:
    def __init__(self, maze_map, maze_size_scaling, maze_height=0.5) -> None:
        xml_path = AntEnv.FILE
        tree = ET.parse(xml_path)
        worldbody = tree.find(".//worldbody")

        self._maze_map = maze_map

        self._maze_height = maze_height
        self._maze_size_scaling = maze_size_scaling

        self._maze_map = maze_map

        # Obtain a numpy array form for a maze map in case we want to reset
        # to multiple starting states
        temp_maze_map = deepcopy(self._maze_map)
        for i in range(len(maze_map)):
            for j in range(len(maze_map[0])):
                if temp_maze_map[i][j] in [
                    RESET,
                ]:
                    temp_maze_map[i][j] = 0
                elif temp_maze_map[i][j] in [
                    GOAL,
                ]:
                    temp_maze_map[i][j] = 1

        self._np_maze_map = np.array(temp_maze_map)

        torso_x, torso_y = self._find_robot()
        self._init_torso_x = torso_x
        self._init_torso_y = torso_y

        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                struct = self._maze_map[i][j]
                if struct == 1:  # Unmovable block.
                    # Offset all coordinates so that robot starts at the origin.
                    ET.SubElement(
                        worldbody,
                        "geom",
                        name="block_%d_%d" % (i, j),
                        pos="%f %f %f"
                        % (
                            j * self._maze_size_scaling - torso_x,
                            i * self._maze_size_scaling - torso_y,
                            self._maze_height / 2 * self._maze_size_scaling,
                        ),
                        size="%f %f %f"
                        % (
                            0.5 * self._maze_size_scaling,
                            0.5 * self._maze_size_scaling,
                            self._maze_height / 2 * self._maze_size_scaling,
                        ),
                        type="box",
                        material="",
                        contype="1",
                        conaffinity="1",
                        rgba="0.7 0.5 0.3 1.0",
                    )

        # torso = tree.find(".//body[@name='torso']")
        # geoms = torso.findall(".//geom")

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)
        atexit.register(os.remove, file_path)
        self.tmp_file_path = file_path

    def _xy_to_rowcol(self, xy):
        size_scaling = self._maze_size_scaling
        xy = (max(xy[0], 1e-4), max(xy[1], 1e-4))
        return (int(1 + (xy[1]) / size_scaling), int(1 + (xy[0]) / size_scaling))

    def _get_reset_location(
        self,
    ):
        prob = (1.0 - self._np_maze_map) / np.sum(1.0 - self._np_maze_map)
        prob_row = np.sum(prob, 1)
        row_sample = np.random.choice(np.arange(self._np_maze_map.shape[0]), p=prob_row)
        col_sample = np.random.choice(
            np.arange(self._np_maze_map.shape[1]),
            p=prob[row_sample] * 1.0 / prob_row[row_sample],
        )
        reset_location = self._rowcol_to_xy((row_sample, col_sample))

        # Add some random noise
        random_x = np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling
        random_y = np.random.uniform(low=0, high=0.5) * 0.5 * self._maze_size_scaling

        return (
            max(reset_location[0] + random_x, 0),
            max(reset_location[1] + random_y, 0),
        )

    def _rowcol_to_xy(self, rowcol, add_random_noise=False):
        row, col = rowcol
        x = col * self._maze_size_scaling - self._init_torso_x
        y = row * self._maze_size_scaling - self._init_torso_y
        if add_random_noise:
            x = x + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
            y = y + np.random.uniform(low=0, high=self._maze_size_scaling * 0.25)
        return (x, y)

    def goal_sampler(self, np_random, only_free_cells=True, interpolate=True):
        valid_cells = []
        goal_cells = []

        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                if self._maze_map[i][j] in [0, RESET, GOAL] or not only_free_cells:
                    valid_cells.append((i, j))
                if self._maze_map[i][j] == GOAL:
                    goal_cells.append((i, j))

        # If there is a 'goal' designated, use that. Otherwise, any valid cell can
        # be a goal.
        sample_choices = goal_cells if goal_cells else valid_cells
        cell = sample_choices[np_random.choice(len(sample_choices))]
        xy = self._rowcol_to_xy(cell, add_random_noise=True)

        random_x = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
        random_y = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling

        xy = (max(xy[0] + random_x, 0), max(xy[1] + random_y, 0))

        return xy

    def _find_robot(self):
        structure = self._maze_map
        size_scaling = self._maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == RESET:
                    return j * size_scaling, i * size_scaling
        raise ValueError("No robot in maze specification.")

    def _is_in_collision(self, pos):
        x, y = pos
        structure = self._maze_map
        size_scaling = self._maze_size_scaling
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if structure[i][j] == 1:
                    minx = j * size_scaling - size_scaling * 0.5 - self._init_torso_x
                    maxx = j * size_scaling + size_scaling * 0.5 - self._init_torso_x
                    miny = i * size_scaling - size_scaling * 0.5 - self._init_torso_y
                    maxy = i * size_scaling + size_scaling * 0.5 - self._init_torso_y
                    if minx <= x <= maxx and miny <= y <= maxy:
                        return True
        return False

    def _get_best_next_rowcol(self, current_rowcol, target_rowcol):
        """Runs BFS to find shortest path to target and returns best next rowcol.
        Add obstacle avoidance"""
        current_rowcol = tuple(current_rowcol)
        target_rowcol = tuple(target_rowcol)
        if target_rowcol == current_rowcol:
            return target_rowcol

        visited = {}
        to_visit = [target_rowcol]
        while to_visit:
            next_visit = []
            for rowcol in to_visit:
                visited[rowcol] = True
                row, col = rowcol
                left = (row, col - 1)
                right = (row, col + 1)
                down = (row + 1, col)
                up = (row - 1, col)
                for next_rowcol in [left, right, down, up]:
                    if next_rowcol == current_rowcol:  # Found a shortest path.
                        return rowcol
                    next_row, next_col = next_rowcol
                    if next_row < 0 or next_row >= len(self._maze_map):
                        continue
                    if next_col < 0 or next_col >= len(self._maze_map[0]):
                        continue
                    if self._maze_map[next_row][next_col] not in [0, RESET, GOAL]:
                        continue
                    if next_rowcol in visited:
                        continue
                    next_visit.append(next_rowcol)
            to_visit = next_visit

        raise ValueError("No path found to target.")


class AntMazeEnv(gym.Env):
    """Ant navigating a maze."""

    def __init__(
        self,
        maze_map,
        maze_size_scaling,
        non_zero_reset,
        maze_height=0.5,
        reward_type="dense",
        v2_resets=False,
        eval=False,
    ):
        self._maze = AntMaze(maze_map, maze_size_scaling, maze_height=maze_height)
        self.ant_env = AntEnv(
            file_path=self._maze.tmp_file_path,
            expose_all_qpos=True,
            expose_body_coms=None,
            expose_body_comvels=None,
            non_zero_reset=non_zero_reset,
        )

        self.action_space = self.ant_env.action_space

        self._goal_sampler = lambda np_rand: self._maze.goal_sampler(np_rand)
        self._goal = np.ones([2])
        self.target_goal = self._goal

        # This flag is used to make sure that when using this environment
        # for evaluation, that is no goals are appended to the state
        self.eval = eval
        if not self.eval:
            observation_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.ant_env.observation_space.shape[0] + 2,),
                dtype=self.ant_env.observation_space.dtype,
            )
        else:
            observation_space = self.ant_env.observation_space

        self.observation_space = observation_space

        # This is the reward type fed as input to the goal confitioned policy
        self.reward_type = reward_type

        ## We set the target foal here for evaluation
        self.set_target_goal()
        self.v2_resets = v2_resets

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed, **kwargs)
        if self.v2_resets:
            """
            The target goal for evaluation in antmazes is randomized.
            antmazes-v0 and -v1 resulted in really high-variance evaluations
            because the target goal was set once at the seed level. This led to
            each run running evaluations with one particular goal. To accurately
            cover each goal, this requires about 50-100 seeds, which might be
            computationally infeasible. As an alternate fix, to reduce variance
            in result reporting, we are creating the v2 environments
            which use the same offline dataset as v0 environments, with the distinction
            that the randomization of goals during evaluation is performed at the level of
            each rollout. Thus running a few seeds, but performing the final evaluation
            over 100-200 episodes will give a valid estimate of an algorithm's performance.
            """
            self.set_target_goal()

        if self.target_goal is not None or self.eval:
            self._goal = self.target_goal
        else:
            self._goal = self._goal_sampler(self.np_random)

        ant_obs, _ = self.ant_env.reset(seed=seed)

        return self._get_obs(ant_obs), {}

    def set_target_goal(self, goal_input=None):
        if goal_input is None:
            self.target_goal = self._goal_sampler(np.random)
        else:
            self.target_goal = goal_input

        print("Target Goal: ", self.target_goal)
        ## Make sure that the goal used in self._goal is also reset:
        self._goal = self.target_goal

    def _get_obs(self, base_obs):
        xy = base_obs[:2]
        goal_direction = self._goal - xy
        if not self.eval:
            obs = np.concatenate([base_obs, goal_direction])
            return obs
        else:
            return base_obs

    def step(self, a):
        ant_obs, _, _, _, _ = self.ant_env.step(a)
        xy = ant_obs[:2]
        if self.reward_type == "dense":
            reward = -np.linalg.norm(self.target_goal - xy)
        elif self.reward_type == "sparse":
            reward = 1.0 if np.linalg.norm(xy - self.target_goal) <= 0.5 else 0.0

        done = False
        # Terminate episode when we reach a goal
        if self.eval and np.linalg.norm(xy - self.target_goal) <= 0.5:
            done = True

        obs = self._get_obs(ant_obs)
        return obs, reward, done, False, {}


def make_ant_maze_env(**kwargs):
    env = AntMazeEnv(**kwargs)
    return wrappers.RescaleAction(env, -1, 1)
