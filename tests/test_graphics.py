
import sys

import time
import numpy as np

import os
os.environ['DISPLAY'] = ':0'
os.environ['XR_RUNTIME_JSON'] = '/usr/share/openxr/1/openxr_monado.json'

from riglib.stereo_opengl.window import Window, Window2D, FPScontrol, WindowVR
from riglib.stereo_opengl.environment import Grid
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

# arm4j = RobotArmGen2D(link_radii=.2, joint_radii=.2, link_lengths=[4,4,2,2])
cone = Sphere(radius=1)
#TexSphere = type("TexSphere", (Sphere, TexModel), {})
#TexPlane = type("TexPlane", (Plane, TexModel), {})
#reward_text = Text(7.5, "123", justify='right', color=[1,0,1,1])

pos_list = np.array([[0,0,0],[0,0,5]])

class Test2(WindowVR, Window):

    def __init__(self, *args, **kwargs):
        self.count=0
        super().__init__(*args, **kwargs)

    def _start_draw(self):
        #arm4j.set_joint_pos([0,0,np.pi/2,np.pi/2])
        #arm4j.get_endpoint_pos()
        #reward_text.attach()
        pass

    def _while_draw(self):
        ts = time.time() - self.start_time
        if ts > 2 and self.count<len(pos_list):
            #reward_text.translate(*pos_list[self.count])
            self.count+=1
        if ts > 4 and self.count<len(pos_list)+1:
            #win.remove_model(reward_text)
            #target = TextTarget('hi', [1,1,1,1], [0,0,0,1], 1)
            #win.add_model(target.model)
            self.count += 1
        self.draw_world()

if __name__ == "__main__":
    win = Test2(window_size=(1000, 800), fullscreen=False, stereo_mode='projection')
    win.add_model(cone)
    # win.add_model(arm4j)
    #win.add_model(reward_text.translate(5,0,-5))
    #win.add_model(TexSphere(radius=1, shininess=30, tex=cloudy_tex()).translate(0,0,0))
    #win.add_model(TexPlane(5,5, tex=cloudy_tex(), specular_color=(0.,0,0,0)).rotate_x(90))
    win.add_model(Grid())
    win.add_model(Sphere(2, color=[0.75,0.25,0.25,0.75]).translate(-5,0,0))
    reward_text = Text(7.5, "123", justify='right', color=[1,0,1,0.75])
    win.add_model(reward_text.translate(5,0,-5))
    win.add_model(TexPlane(4,4,color=[0,0,0,0.9], tex=cloudy_tex()).rotate_x(90).translate(0,0,-5))
    #win.screen_init()
    #win.draw_world()
    win.run()
