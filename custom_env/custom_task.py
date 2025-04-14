from typing import Any, Dict

import numpy as np
import pybullet
from pybullet import getContactPoints, getNumBodies, getCollisionShapeData
from pybullet_utils.examples.mjcf2urdf import robotName

from panda_gym.envs.core import Task
from panda_gym.pybullet import PyBullet
from panda_gym.utils import distance


class CustomPickAndPlace(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.02,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.2,
        obj_xy_range: float = 0.2,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, 0.0, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

        #swap depth and width around
        shelf_width = 0.2
        shelf_depth = 0.6
        shelf_height = 0.8
        thickness = 0.02

        self.sim.create_box(
            body_name="shelf_base",
            half_extents=np.array([shelf_width / 2, shelf_depth / 2, thickness / 2]),
            mass=0.0,
            position=np.array([0.15, 0.0, 0.0]),
            rgba_color=np.array([0.6, 0.3, 0.1, 1.0]),
        )

        self.sim.create_box(
            body_name="shelf_surface",
            half_extents=np.array([shelf_width / 2, shelf_depth / 2, thickness / 2]),
            mass=0.0,
            position=np.array([0.15, 0.0, shelf_height / 2]),
            rgba_color=np.array([0.6, 0.3, 0.1, 1.0]),
        )

        self.sim.create_box(
            body_name="shelf_left",
            half_extents=np.array([shelf_width / 2, thickness / 2, shelf_height / 2]),
            mass=0.0,
            position=np.array([0.15, shelf_depth / 2, shelf_height / 2]),
            rgba_color=np.array([0.6, 0.3, 0.1, 1.0]),
        )

        self.sim.create_box(
            body_name="shelf_right",
            half_extents=np.array([thickness / 2, shelf_depth / 2, shelf_height / 2]),
            mass=0.0,
            position=np.array([(shelf_width / 2) + 0.15, 0.0, 0.0 + shelf_height / 2]),
            rgba_color=np.array([0.6, 0.3, 0.1, 1.0]),
        )

        self.sim.create_box(
            body_name="shelf_back",
            half_extents=np.array([shelf_width / 2, thickness / 2, shelf_height / 2]),
            mass=0.0,
            position=np.array([0.15, -shelf_depth / 2, 0.0 + shelf_height / 2]),
            rgba_color=np.array([0.6, 0.3, 0.1, 1.0]),
        )

        self.sim.create_box(
            body_name="shelf_top",
            half_extents=np.array([shelf_width / 2, shelf_depth / 2, thickness / 2]),
            mass=0.0,
            position=np.array([0.15, 0.0, shelf_height]),
            rgba_color=np.array([0.6, 0.3, 0.1, 1.0]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        object_position = np.array(self.sim.get_base_position("object"))
        return object_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.80, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        # Shelf center position
        goal_x = 0.15
        goal_y = self.np_random.uniform(-0.1, 0.1)  # Add randomness along y-axis
        goal_z = (self.object_size / 2) + 0.41  # Height of the shelf

        return np.array([goal_x, goal_y, goal_z])

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([-0.05, 0.0, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)

        robot_id = self.sim.get_bodies_id("panda")
        target_id = self.sim.get_bodies_id("target")
        object_id = self.sim.get_bodies_id("object")
        shelf_surface_id = self.sim.get_bodies_id("shelf_surface")
        punishment = 0
        object_position = self.sim.get_base_position("object")
        target_position = self.sim.get_base_position("target")
        for i in range(self.sim.physics_client.getNumBodies()):
            if i not in {robot_id, target_id, object_id}:
                contact_points = self.sim.physics_client.getContactPoints(robot_id, i)
                if contact_points and not (np.linalg.norm(target_position - object_position) < self.distance_threshold):
                    print(f"collision with {self.sim.get_bodies_name(i)}")
                    punishment = 0.4
                if contact_points and i == shelf_surface_id and object_position[2] < target_position[2]:
                    punishment = 1.0

        gripper_distance = np.linalg.norm(self.sim.get_link_position("panda",8)- object_position)
        approach_bonus = 0.7 * (1 - np.tanh(gripper_distance * 5))

        lift_bonus = 0.7 if object_position[2] > 0.02 else 0.0

        if object_position[2] >= target_position[2]:
            target_proximity_bonus = 1.0 * (1 - np.tanh(np.linalg.norm(object_position - target_position) * 5))
        else:
            target_proximity_bonus = 0.0

        placement_bonus = 2.0 if np.linalg.norm(target_position - object_position) <= self.distance_threshold else 0.0
        print(placement_bonus)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32) + approach_bonus - punishment + lift_bonus + placement_bonus + target_proximity_bonus
        else:
            return -d.astype(np.float32) + approach_bonus - punishment + lift_bonus + placement_bonus + target_proximity_bonus