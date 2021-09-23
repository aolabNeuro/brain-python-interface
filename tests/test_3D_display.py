import sys

import time
import numpy as np

import os
print(os.environ['DISPLAY'])
from riglib.stereo_opengl.window import Window, FPScontrol
from riglib.stereo_opengl.primitives import Cylinder, Plane, Sphere, Cone, Cube
from riglib.stereo_opengl.models import FlatMesh, Group
from riglib.stereo_opengl.textures import Texture, TexModel
from riglib.stereo_opengl.render import ssao, stereo, Renderer
from riglib.stereo_opengl.utils import cloudy_tex

from riglib.stereo_opengl.ik import RobotArmGen2D
from riglib.stereo_opengl.xfm import Quaternion
import time

from riglib.stereo_opengl.ik import RobotArm

import pygame


cone = Sphere(radius=0.5, color = (1,0,0,1))

pos_list = np.array([[0,0,0],[0,0,5]])

class Test(Window):

    def __init__(self, *args, **kwargs):
        self.task_data = None
        super(Test,self).__init__(*args, **kwargs)

if __name__ == "__main__":
    win = Test()
    win.add_model(cone)
    cone.translate(3.24,0,0)
    win.screen_init()
    win.draw_world()

    time.sleep(120)

