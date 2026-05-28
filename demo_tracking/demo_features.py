"""Demo-local feature mixins used by demo tracking launcher.

These mirror selected classes from features modules while avoiding import-time
side effects from importing the full features package in demo/binary runs.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pygame
from OpenGL.GL import GL_REPEAT
from built_in_tasks.target_graphics import TextTarget, VirtualRectangularTarget
from riglib.experiment import traits
from riglib.stereo_opengl.primitives import Sphere, TexSphere
from riglib.stereo_opengl.textures import Texture


def _resource_path(*parts: str) -> str:
    """Return an absolute path to a bundled resource for source or frozen runs."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_dir, *parts)


class Mouse:
    """Simple mouse-backed data source compatible with TrackingTask."""

    def __init__(self, window_size, screen_cm, start_pos):
        self.window_size = window_size
        self.screen_cm = screen_cm
        self.pos = [float(start_pos[0]), float(start_pos[1])]

    def get(self):
        pos = pygame.mouse.get_pos()
        self.pos[0] = (pos[0] / self.window_size[0] - 0.5) * self.screen_cm[0]
        self.pos[1] = -(pos[1] / self.window_size[1] - 0.5) * self.screen_cm[1]
        return [self.pos]


class MouseControl(traits.HasTraits):
    def init(self, *args, **kwargs):
        super().init(*args, **kwargs)
        self.joystick = Mouse(self.window_size, self.screen_cm, np.array(self.starting_pos[::2]))


class SpheresToImages:
    """Convert spheres to textured disks facing the camera."""

    def add_model(self, model):
        if isinstance(model, Sphere) and model.radius > 1.0:
            texture = Texture(_resource_path("features", "images", "moon.png"), wrap_x=GL_REPEAT, wrap_y=GL_REPEAT)
        elif isinstance(model, Sphere):
            texture = Texture(_resource_path("features", "images", "ship.png"), wrap_x=GL_REPEAT, wrap_y=GL_REPEAT)
        else:
            super().add_model(model)
            return

        tmp_model = TexSphere(
            model.radius,
            color=[0, 0, 0, 1],
            specular_color=[0, 0, 0, 0],
            tex=texture,
            texture_mapping="planar",
        )
        model.__class__ = TexSphere
        model.verts = tmp_model.verts
        model.polys = tmp_model.polys
        model.tcoords = tmp_model.tcoords
        model.normals = tmp_model.normals
        model.tex = tmp_model.tex
        model.shader = tmp_model.shader
        model.rotate_x(90)
        super().add_model(model)


class ProgressBar(traits.HasTraits):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bar_width = 12
        self.bar = VirtualRectangularTarget(
            target_width=1,
            target_height=0,
            target_color=(0.0, 1.0, 0.0, 0.75),
            starting_pos=[0, -15, 9],
        )

    def setup_start_wait(self):
        super().setup_start_wait()
        for model in self.bar.graphics_models:
            self.add_model(model)
            self.bar.hide()
        self.tracking_frame_index = 0

    def setup_screen_reset(self):
        super().setup_screen_reset()
        self.bar.hide()
        self.bar.reset()

    def _rebuild_bar(self):
        self.tracking_rate = self.tracking_frame_index / self.trajectory_length * self.bar_width
        if hasattr(self, "bar"):
            for model in self.bar.graphics_models:
                self.remove_model(model)
            del self.bar
        self.bar = VirtualRectangularTarget(
            target_width=1.3,
            target_height=self.tracking_rate,
            target_color=(0.0, 1.0, 0.0, 0.75),
            starting_pos=[self.tracking_rate - self.bar_width, -15, 9],
        )
        for model in self.bar.graphics_models:
            self.add_model(model)
        self.bar.show()

    def _while_tracking_in(self):
        super()._while_tracking_in()
        self.tracking_frame_index += 1
        self._rebuild_bar()

    def _while_tracking_in_ramp(self):
        super()._while_tracking_in_ramp()
        self.tracking_frame_index += 1
        self._rebuild_bar()

    def _start_reward(self):
        super()._start_reward()
        self.reward_frame_index = 0

    def _while_reward(self):
        super()._while_reward()
        if hasattr(self, "bar"):
            for model in self.bar.graphics_models:
                self.remove_model(model)
            del self.bar

        self.reward_frame_index += 1
        reward_numframe = self.reward_time * self.fps * 0.85
        reward_amount = self.tracking_rate - self.reward_frame_index * self.tracking_rate / reward_numframe
        if reward_amount < 0:
            reward_amount = 0

        self.bar = VirtualRectangularTarget(
            target_width=1.3,
            target_height=reward_amount,
            target_color=(0.0, 1.0, 0.0, 0.75),
            starting_pos=[reward_amount - self.bar_width, -15, 9],
        )
        for model in self.bar.graphics_models:
            self.add_model(model)
        self.bar.show()


class ScoreRewards(traits.HasTraits):
    score_display_location = traits.Tuple((10, 0, 10), desc="Location to display the score (in cm)")
    score_display_size = traits.Int(36, desc="Font size of the score display")
    score_display_color = traits.Tuple((1, 1, 1, 1), desc="Color of the score display")
    score_multiplier = traits.Int(100, desc="Value to multiple the score by")
    score_function = traits.OptionsList(
        "timed",
        ["fixed", "timed"],
        desc="Function to calculate the score for each reward",
        bmi3d_input_options=["fixed", "timed"],
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reportstats["Score"] = 0

    def init(self):
        self.add_dtype("reward_score", "int", (1,))
        super().init()
        self.task_data["reward_score"] = 0

    def setup_start_wait(self):
        super().setup_start_wait()
        self.tracking_in_counter = 0

    def _while_tracking_in(self):
        super()._while_tracking_in()
        self.tracking_in_counter += 1

    def _while_tracking_in_ramp(self):
        super()._while_tracking_in_ramp()
        self.tracking_in_counter += 1

    def _start_reward(self):
        if hasattr(super(), "_start_reward"):
            super()._start_reward()

        if self.score_function == "fixed":
            score = self.score_multiplier
        elif hasattr(self, "tracking_in_counter") and hasattr(self, "trajectory_length"):
            score = int(self.score_multiplier * self.tracking_in_counter / self.trajectory_length)
        else:
            timed_state = None
            idx = -1
            while timed_state is None and -idx - 1 < len(self.state_log):
                if self.state_log[idx][0] == "target":
                    timed_state = self.state_log[-1][1] - self.state_log[idx][1]
                idx -= 1
            if timed_state is None or timed_state == 0.0:
                score = 0
            else:
                score = int(self.score_multiplier / timed_state)

        self.reportstats["Score"] += score
        self.task_data["reward_score"] += score
        self.score_display = TextTarget(str(score), font_size=self.score_display_size, color=self.score_display_color)
        self.score_display.move_to_position(self.score_display_location)
        self.add_model(self.score_display.model)

    def _end_reward(self):
        if hasattr(super(), "_end_reward"):
            super()._end_reward()
        self.remove_model(self.score_display.model)
        self.score_display.model.release()
