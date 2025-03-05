from typing import Any, Dict, Optional, Tuple
import numpy as np

from gymnasium import spaces
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from custom_env.custom_task import CustomPickAndPlace

class CustomPickAndPlaceEnv(RobotTaskEnv):

    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = CustomPickAndPlace(sim, reward_type=reward_type)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
        self.obstacle_collision_margin = 0.022

        observation, _ = self.reset()  # required for init; seed can be changed later
        observation_shape = observation["observation"].shape
        achieved_goal_shape = observation["achieved_goal"].shape
        desired_goal_shape = observation["desired_goal"].shape
        obstacle_dist_shape = observation["obstacle_dist"].shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32),
                desired_goal=spaces.Box(-10.0, 10.0, shape=desired_goal_shape, dtype=np.float32),
                achieved_goal=spaces.Box(-10.0, 10.0, shape=achieved_goal_shape, dtype=np.float32),
                obstacle_dist=spaces.Box(-10.0, 10.0, shape=obstacle_dist_shape, dtype=np.float32),
            )
        )


    def _get_obs(self) -> Dict[str, np.ndarray]:
        ee_position = self.robot.get_link_position(8)
        obstacle_dist = self.sim.get_closest_dist(ee_position)
        obstacle_dist_vector = obstacle_dist[0].astype(np.float32)
        robot_obs = self.robot.get_obs().astype(np.float32)  # robot state
        task_obs = self.task.get_obs().astype(np.float32)  # object position, velocity, etc...
        observation = np.concatenate([robot_obs, task_obs])
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
            "desired_goal": self.task.get_goal().astype(np.float32),
            "obstacle_dist": obstacle_dist_vector,
        }

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        if self.sim.is_collision(self.obstacle_collision_margin):
            with open("collision.txt", "w") as file:
                file.write("Collision detected\n")
            terminated = True
            info = {"is_success": False, "is_collision": True}
        else:
            terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
            info = {"is_success": terminated, "is_collision": False}
        truncated = False
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        return observation, reward, terminated, truncated, info
