'''
Various graphical "environmental" or "world" add-ins to graphical tasks. 
'''
from riglib.stereo_opengl.utils import cloudy_tex, create_grid_texture
from .models import Group
from .xfm import Quaternion
from ..stereo_opengl.primitives import Cube, Plane, Sphere, Pipe, TexCube, TexModel, TexPlane


class Box(Group):
    '''
    Construct a 3D wireframe box in the world to add some depth cue references
    '''
    def __init__(self, sidelen=10, **kwargs):
        '''
        Constructor for Box 

        Parameters
        ----------
        kwargs: optional keyword arguments
            All passed to parent constructor

        Returns
        -------
        Box instance
        '''
        bcolor = (181/256., 116/256., 96/256., 1)
        linerad=.1
        self.vert_box = Group([
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, -sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, -sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, sidelen/2, -sidelen/2)])
        self.hor_box = Group([
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, -sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, -sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, sidelen/2, -sidelen/2)])
        self.depth_box = Group([
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, -sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, -sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(sidelen/2, sidelen/2, -sidelen/2),
            Pipe(radius=linerad, height=sidelen, color=bcolor).translate(-sidelen/2, sidelen/2, -sidelen/2)])
        self.hor_box.xfm.rotate = Quaternion.rotate_vecs((0,0,1), (1,0,0))
        self.depth_box.xfm.rotate = Quaternion.rotate_vecs((0,0,1), (0,1,0))
        self.box = Group([self.hor_box, self.depth_box, self.vert_box])
        super(Box, self).__init__([self.box], **kwargs)

class Grid(Group):

    def __init__(self, size=25, **kwargs):
        '''
        Constructor for Grid

        Parameters
        ----------
        kwargs: optional keyword arguments
            All passed to parent constructor

        Returns
        -------
        Grid instance
        '''
        grid_tex = create_grid_texture()
        self.grid = Group([
            TexPlane(size,size, specular_color=(0,0,0,0), tex=grid_tex).rotate_y(90).translate(-size/2,-size/2,size/2),
            TexPlane(size,size, specular_color=(0,0,0,0), tex=grid_tex).rotate_y(270).translate(size/2,-size/2,-size/2),
            TexPlane(size,size, specular_color=(0,0,0,0), tex=grid_tex).rotate_x(0).translate(-size/2,-size/2,-size/2),
            TexPlane(size,size, specular_color=(0,0,0,0), tex=grid_tex).rotate_x(180).translate(-size/2,size/2,size/2),
            TexPlane(size,size, specular_color=(0,0,0,0), tex=grid_tex).rotate_x(90).translate(-size/2,size/2,-size/2),
        ])
        super(Grid, self).__init__([self.grid], **kwargs)
