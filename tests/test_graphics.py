import sys

import time
import numpy as np

import os
os.environ['DISPLAY'] = ':0'

from riglib.stereo_opengl.window import Window, Window2D, FPScontrol, WindowVR, WindowSSAO
from riglib.stereo_opengl.environment import Box, Grid
from riglib.stereo_opengl.primitives import Cylinder, Cube, Plane, Sphere, Cone, Text, TexSphere, TexCube, TexPlane
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.render import ssao, stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from riglib.stereo_opengl.ik import RobotArmGen2D
from riglib.stereo_opengl.xfm import Quaternion
import time

from riglib.stereo_opengl.ik import RobotArm

from built_in_tasks.target_graphics import TextTarget

import pygame
from OpenGL.GL import *

# arm4j = RobotArmGen2D(link_radii=.2, joint_radii=.2, link_lengths=[4,4,2,2])
moon = Sphere(radius=1, color=[0.25,0.25,0.75,0.5])
orbit_radius = 4
orbit_speed = 1
#TexSphere = type("TexSphere", (Sphere, TexModel), {})
#TexPlane = type("TexPlane", (Plane, TexModel), {})
#reward_text = Text(7.5, "123", justify='right', color=[1,0,1,1])

pos_list = np.array([[0,0,0],[0,0,5]])

class Test2( Window):

    def __init__(self, *args, **kwargs):
        self.count=0
        super().__init__(*args, **kwargs)

    def _start_draw(self):
        #arm4j.set_joint_pos([0,0,np.pi/2,np.pi/2])
        #arm4j.get_endpoint_pos()
        self.add_model(moon)
        self.add_model(Sphere(3, color=[0.75,0.25,0.25,0.75]).translate(-5,5,0))
        # self.add_model(arm4j)
        #self.add_model(reward_text.translate(5,0,-5))
        # self.add_model(TexSphere(radius=3, specular_color=[1,1,1,1], tex=cloudy_tex()).translate(5,0,0))
        # self.add_model(TexPlane(5,5, tex=cloudy_tex(), specular_color=(0.,0,0,1)).rotate_x(90))
        # self.add_model(TexPlane(5,5, specular_color=(0.,0,0,1), tex=cloudy_tex()).rotate_x(90))
        reward_text = Text(7.5, "123", justify='right', color=[1,0,1,0.75])
        self.add_model(reward_text)
        # self.add_model(TexPlane(4,4,color=[0,0,0,0.9], tex=cloudy_tex()).rotate_x(90).translate(0,0,-5))
        #self.screen_init()
        #self.draw_world()

    def _while_draw(self):
        ts = time.time() - self.start_time
        
        x = orbit_radius * np.cos(ts * orbit_speed)
        z = orbit_radius * np.sin(ts * orbit_speed)

        moon.translate(x-5,z+5,0,reset=True)


        if ts > 2 and self.count<len(pos_list):
            # reward_text.translate(*pos_list[self.count])
            self.count+=1
        if ts > 4 and self.count<len(pos_list)+1:
            # win.remove_model(reward_text)
            # target = TextTarget('hi', [1,1,0,1], 1)
            # win.add_model(target.model)
            self.count += 1
        self.draw_world()

        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error after drawing: {error}")

if __name__ == "__main__":
    win = Test2(window_size=(1000, 800), fullscreen=False, stereo_mode='projection')
    win.run()
