'''
Basic OpenGL shapes constructed out of triangular meshes
'''

import numpy as np
from numpy import pi
try:
    import os
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"    
    import pygame
except:
    import warnings
    warnings.warn('riglib/stereo_opengl_primitives.py: not importing name pygame')
import matplotlib.tri as mtri

from .models import TriMesh
from .textures import Texture, TexModel
from OpenGL.GL import GL_NEAREST
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

class Plane(TriMesh):
    def __init__(self, width=1, height=1, **kwargs):
        pts = np.array([[0,0,0],
                        [width,0,0],
                        [width,height,0],
                        [0,height,0]])
        polys = [(0,1,3),(1,2,3)]
        tcoords = np.array([[0,0],[1,0],[1,1],[0,1]])
        normals = [(0,0,1)]*4
        super(Plane, self).__init__(pts, np.array(polys), 
                tcoords=tcoords, normals=np.array(normals), **kwargs)

class Cube(TriMesh):
    def __init__(self, side_len=1, side_height=None, segments=36, **kwargs):
        self.side_len = side_len
        if side_height is None:
            side_len_half = side_len/2.
        else:
            side_len_half = side_height # 0.5
        side = np.linspace(-1, 1, int(segments/4), endpoint=True)
        
        unit1 = np.hstack(( side[:,np.newaxis], np.ones((len(side),1)), np.ones((len(side),1)) ))
        unit2 = np.hstack(( np.ones((len(side),1)), side[::-1,np.newaxis], np.ones((len(side),1)) ))
        unit3 = np.hstack(( side[::-1,np.newaxis], -1*np.ones((len(side),1)), np.ones((len(side),1)) ))
        unit4 = np.hstack(( -1*np.ones((len(side),1)), side[:,np.newaxis], np.ones((len(side),1)) ))

        unit = np.vstack((unit1, unit2, unit3, unit4))

        pts = np.vstack([unit*[side_len_half, side_len_half, 0], unit*[side_len_half,side_len_half,side_len]])
        normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])
        #pts = np.vstack([unit*[side_len, 0, 0], unit*[side_len,0,side_height]])
        #normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])
        polys = []
        for i in range(segments-1):
            polys.append((i, i+1, i+segments))
            polys.append((i+segments, i+1, i+1+segments))
        polys.append((segments-1, 0, segments*2-1))
        polys.append((segments*2-1, 0, segments))
        
        tcoord = np.array([np.arange(segments), np.ones(segments)]).T
        n = 1./segments
        tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])

        super(Cube, self).__init__(pts, np.array(polys), 
            tcoords=tcoord, normals=normals, **kwargs)

class Pipe(TriMesh):
    '''
    Open pipe standing on the xy-plane with the z-axis as the height dimension
    '''
    def __init__(self, height=1, radius=1, segments=36, **kwargs):
        self.height = height
        self.radius = radius
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        unit = np.array([np.cos(theta), np.sin(theta), np.ones(segments)]).T

        pts = np.vstack([unit*[radius, radius, 0], unit*[radius,radius,height]])
        normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])

        polys = []
        for i in range(segments-1):
            polys.append((i, i+1, i+segments))
            polys.append((i+segments, i+1, i+1+segments))
        polys.append((segments-1, 0, segments*2-1))
        polys.append((segments*2-1, 0, segments))
        
        tcoord = np.array([np.arange(segments), np.ones(segments)]).T
        n = 1./segments
        tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])

        super().__init__(pts, np.array(polys), 
            tcoords=tcoord, normals=normals, **kwargs)
        
class Disk(TriMesh):
    def __init__(self, radius=1, segments=36, **kwargs):
        pts = [[0, 0, 0]]  # Center point
        
        # Calculate points around the circumference
        angle_increment = 2 * pi / (segments - 2) # not sure why this works
        for i in range(segments):
            x = radius * np.cos(i * angle_increment)
            y = radius * np.sin(i * angle_increment)
            pts.append([x, y, 0])

        pts = np.array(pts)
        
        # Create polygons
        polys = [(0, i, (i + 1) % segments) for i in range(segments)]

        # Texture coordinates
        tcoords = np.zeros((segments + 1, 2))
        tcoords[1:, 0] = np.cos(np.linspace(0, 2 * np.pi, segments))
        tcoords[1:, 1] = np.sin(np.linspace(0, 2 * np.pi, segments))
        
        normals = [(0, 0, 1)] * segments
        
        super().__init__(pts, np.array(polys), 
                                    tcoords=tcoords, normals=np.array(normals), **kwargs)

class Cylinder(TriMesh):
    '''
    Closed cylinder centered on the xy-plane with the z-axis as the height dimension

    Args:
        TriMesh (_type_): _description_
    '''
    def __init__(self, height=1, radius=1, segments=36, **kwargs):
        self.height = height
        self.radius = radius

        body_mesh = Pipe(height, radius, segments)
        top_mesh = Disk(radius, segments)
        bottom_mesh = Disk(radius, segments)

        # Adjust polygon indices for top and bottom based on total number of vertices
        offset_body = len(body_mesh.verts)
        offset_top = offset_body + len(top_mesh.verts)

        total_pts = np.concatenate([body_mesh.verts + np.array([0, 0, -height/2, 0]),
                                    top_mesh.verts + np.array([0, 0, height/2, 0]), 
                                    bottom_mesh.verts + np.array([0, 0, -height/2, 0])])
        total_normals = np.concatenate([body_mesh.normals, top_mesh.normals, bottom_mesh.normals])
        total_tcoords = np.concatenate([body_mesh.tcoords, top_mesh.tcoords, bottom_mesh.tcoords])

        # Update polygons to account for mesh offsets 
        body_polys = body_mesh.polys + 0  # Keep body polygon indices the same 
        top_polys = top_mesh.polys + offset_body
        bottom_polys = bottom_mesh.polys + offset_top

        total_polys = np.concatenate([body_polys, top_polys, bottom_polys])

        # Create the final mesh with combined data
        super().__init__(total_pts, total_polys, tcoords=total_tcoords, normals=total_normals, **kwargs)

class Cable(TriMesh):
    def __init__(self,radius=.5, trajectory = np.array([np.sin(x) for x in range(60)]), segments=12,**kwargs):
        self.trial_trajectory = trajectory
        self.center_value = [0,0,0]
        self.radius = radius
        self.segments = segments
        self.update(**kwargs)
    
    def update(self, **kwargs):
        theta = np.linspace(0, 2*np.pi, self.segments, endpoint=False)
        unit = np.array([np.ones(self.segments),np.cos(theta) ,np.sin(theta)]).T
        intial = np.array([[0,0,self.trial_trajectory[x]] for x in range(len(self.trial_trajectory))])
        self.pts = (unit*[-30/1.36,self.radius,self.radius])+intial[0]
        for i in range(1,len(intial)):
            self.pts = np.vstack([self.pts, (unit*[(i-30)/3,self.radius,self.radius])+intial[i]])

        self.normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])
        self.polys = []
        for i in range(self.segments-1):
            for j in range(len(intial)-1): 
                self.polys.append((i+j*self.segments, i+1+j*self.segments, i+self.segments+j*self.segments))
                self.polys.append((i+self.segments+j*self.segments, i+1+j*self.segments, i+1+self.segments+j*self.segments))

        tcoord = np.array([np.arange(self.segments), np.ones(self.segments)]).T
        n = 1./self.segments
        self.tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])
        super(Cable, self).__init__(self.pts, np.array(self.polys), 
            tcoords=self.tcoord, normals=self.normals, **kwargs)

class Torus(TriMesh):
    '''
    Corrected triangle mesh of a torus. Should work well for 3D rendering with lighting.
    '''

    def __init__(self, major_radius=1, minor_radius=0.5, segments_major=36, segments_minor=18, **kwargs):
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        theta_major = np.linspace(0, 2*np.pi, segments_major, endpoint=False)
        phi_minor = np.linspace(0, 2*np.pi, segments_minor, endpoint=False)

        # Create the points for the torus
        pts = []
        for i in range(segments_major):
            for j in range(segments_minor):
                x = (major_radius + minor_radius * np.cos(phi_minor[j])) * np.cos(theta_major[i])
                y = (major_radius + minor_radius * np.cos(phi_minor[j])) * np.sin(theta_major[i])
                z = minor_radius * np.sin(phi_minor[j])
                pts.append([x, y, z])

        pts = np.array(pts)

        # Calculate the normals for the torus using cross product
        normals = []
        for i in range(segments_major):
            for j in range(segments_minor):
                current = pts[i * segments_minor + j]
                next_i = pts[((i + 1) % segments_major) * segments_minor + j]
                next_j = pts[i * segments_minor + (j + 1) % segments_minor]

                edge1 = next_i - current
                edge2 = next_j - current
                normal = np.cross(edge1, edge2)
                normals.append(normal)

        normals = np.array(normals)
        normals /= np.linalg.norm(normals, axis=1)[:, None]

        # Create the polygons for the torus
        polys = []
        for i in range(segments_major):
            for j in range(segments_minor):
                p1 = i * segments_minor + j
                p2 = ((i + 1) % segments_major) * segments_minor + j
                p3 = ((i + 1) % segments_major) * segments_minor + (j + 1) % segments_minor
                p4 = i * segments_minor + (j + 1) % segments_minor
                polys.append((p1, p2, p3))
                polys.append((p3, p4, p1))

        # Create the texture coordinates for the torus
        tcoord = np.array([[i, j] for i in np.linspace(0, 1, segments_major) for j in np.linspace(0, 1, segments_minor)])

        super(Torus, self).__init__(pts, np.array(polys), tcoords=tcoord, normals=normals, **kwargs)

class Sphere(TriMesh):
    def __init__(self, radius=1, segments=36, **kwargs):
        self.radius = radius
        zvals = radius * np.cos(np.linspace(0, np.pi, num=segments))
        circlevals = np.linspace(0, 2*pi, num=segments, endpoint=False)

        vertices = np.zeros(((len(zvals)-2) * len(circlevals), 3))

        for i, z in enumerate(zvals[1:-1]):
            circlepoints = np.zeros((segments, 3))
            circlepoints[:,2] = z
            r = np.sqrt(radius**2 - z**2)
            circlepoints[:,0] = r*np.sin(circlevals)
            circlepoints[:,1] = r*np.cos(circlevals)
            vertices[segments*i:segments*(i+1),:] = circlepoints
        
        vertices = np.vstack([vertices,(0,0,radius),(0,0,-radius)])
        allpointinds = np.arange(len(vertices))
        
        triangles = np.zeros((segments,3))
        firstcirc = allpointinds[0:segments]
        triangles[0,:] = (allpointinds[-2],firstcirc[0], firstcirc[-1])
        for i in range(segments-1):
            triangles[i+1,:] = (allpointinds[-2], firstcirc[i+1], firstcirc[i])
        
        triangles = list(triangles)
        for i in range(segments-3):
            points1 = allpointinds[i*segments:(i+1)*segments]
            points2 = allpointinds[(i+1)*segments:(i+2)*segments]
            for ind, p in enumerate(points1[:-1]):
                t1 = (p, points1[ind+1], points2[ind+1])
                t2 = (p, points2[ind+1], points2[ind])
                triangles += [t1, t2]
            triangles += [(points1[-1], points1[0], points2[0]), (points1[-1], points2[0], points2[-1])]
        
        bottom = np.zeros((segments,3))
        lastcirc = allpointinds[-segments-2:-2]
        bottom[0,:] = (allpointinds[-1], lastcirc[-1], lastcirc[0]) 
        for i in range(segments-1):
            bottom[i+1,:] = (allpointinds[-1], lastcirc[i], lastcirc[i+1])
        triangles = np.vstack([triangles, bottom])
        
        normals = vertices/radius
        hcoord = np.arctan2(normals[:,1], normals[:,0])
        vcoord = np.arctan2(normals[:,2], np.sqrt(vertices[:,0]**2 + vertices[:,1]**2))
        tcoord = np.array([(hcoord+pi) / (2*pi), (vcoord+pi/2) / pi]).T

        super(Sphere, self).__init__(vertices, np.array(triangles), 
            tcoords=tcoord, normals=normals, **kwargs)


class Cone(TriMesh):
    def __init__(self, height=1, radius1=1, radius2=1, segments=36, **kwargs):
        self.height = height
        self.radius1 = radius1
        self.radius2 = radius2
        self.radius = radius1 # for pretending it's a cylinder..
        theta = np.linspace(0, 2*np.pi, segments, endpoint=False)
        unit = np.array([np.cos(theta), np.sin(theta), np.ones(segments)]).T

        pts = np.vstack([unit*[radius1, radius1, 0], unit*[radius2,radius2,height]])
        normals = np.vstack([unit*[1,1,0], unit*[1,1,0]])

        polys = []
        for i in range(segments-1):
            polys.append((i, i+1, i+segments))
            polys.append((i+segments, i+1, i+1+segments))
        polys.append((segments-1, 0, segments*2-1))
        polys.append((segments*2-1, 0, segments))
        
        tcoord = np.array([np.arange(segments), np.ones(segments)]).T
        n = 1./segments
        tcoord = np.vstack([tcoord*[n,1], tcoord*[n,0]])

        super(Cone, self).__init__(pts, np.array(polys), 
            tcoords=tcoord, normals=normals, **kwargs)


class Chain(object):
    '''
    An open chain of cylinders and cones, e.g. to simulate a stick-figure arm/robot
    '''
    def __init__(self, link_radii, joint_radii, link_lengths, joint_colors, link_colors):
        from .models import Group
        from .xfm import Quaternion
        self.num_joints = num_joints = len(link_lengths)

        self.link_radii = self.make_list(link_radii, num_joints)
        self.joint_radii = self.make_list(joint_radii, num_joints)
        self.link_lengths = self.make_list(link_lengths, num_joints)
        self.joint_colors = self.make_list(joint_colors, num_joints)
        self.link_colors = self.make_list(link_colors, num_joints)        

        self.links = []

        # Create the link graphics
        for i in range(self.num_joints):
            joint = Sphere(radius=self.joint_radii[i], color=self.joint_colors[i])

            # The most distal link gets a tapered cylinder (for purely stylistic reasons)
            if i < self.num_joints - 1:
                link = Cylinder(radius=self.link_radii[i], height=self.link_lengths[i], color=self.link_colors[i])
            else:
                link = Cone(radius1=self.link_radii[-1], radius2=self.link_radii[-1]/2, height=self.link_lengths[-1], color=self.link_colors[-1])
            link_i = Group((link, joint))
            self.links.append(link_i)

        link_offsets = [0] + self.link_lengths[:-1]
        self.link_groups = [None]*self.num_joints
        for i in range(self.num_joints)[::-1]:
            if i == self.num_joints-1:
                self.link_groups[i] = self.links[i]
            else:
                self.link_groups[i] = Group([self.links[i], self.link_groups[i+1]])

            self.link_groups[i].translate(0, 0, link_offsets[i])

    def _update_link_graphics(self, curr_vecs):
        from .models import Group
        from .xfm import Quaternion

        for i in range(self.num_joints):
            # Rotate each joint to the vector specified by the corresponding row in self.curr_vecs
            # Annoyingly, the baseline orientation of the first group is always different from the 
            # more distal attachments, so the rotations have to be found relative to the orientation 
            # established at instantiation time.
            if i == 0:
                baseline_orientation = (0, 0, 1)
            else:
                baseline_orientation = (1, 0, 0)

            # Find the normalized quaternion that represents the desired joint rotation
            self.link_groups[i].xfm.rotate = Quaternion.rotate_vecs(baseline_orientation, curr_vecs[i]).norm()

            # Recompute any cached transformations after the change
            self.link_groups[i]._recache_xfm()

    def translate(self, *args, **kwargs):
        self.link_groups[0].translate(*args, **kwargs)

    @staticmethod
    def make_list(value, num_joints):
        '''
        Helper function to allow joint/link properties of the chain to be specified
        as one value for all joints/links or as separate values for each
        '''
        if isinstance(value, list) and len(value) == num_joints:
            return value
        else:
            return [value] * num_joints

TexCube = type("TexCube", (Cube, TexModel), {})
TexPlane = type("TexPlane", (Plane, TexModel), {})
TexSphere = type("TexSphere", (Sphere, TexModel), {})
TexCylinder = type("TexCylinder", (Cylinder, TexModel), {})
TexPipe = type("TexPipe", (Pipe, TexModel), {})
TexCone = type("TexCone", (Cone, TexModel), {})

class Text(TexPlane):
    '''
    A 2D plane with text rendered on. The plane coordinates are its bottom-left corner,
    and the text is rendered along the bottom edge of the plane, either left or right justified.
    Text is always rendered on a square texture, so the width and height of the plane
    are the same.
    '''

    @staticmethod
    def find_font_file(font_name):
        try:
            font_prop = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family=font_name)))
            font_path = font_prop.get_file()
            return font_path
        except Exception as e:
            return None

    def __init__(self, height, text, font_size=28, color=[1, 1, 1, 1], justify='left', 
                 font_name='ubuntu', texture_size=(256,256), shader='ui', **kwargs):
        color = tuple((255*np.array(color)).astype(int))

        # Create a PIL image with a transparent background
        image = Image.new('RGBA', texture_size, (0,0,0,0))
        draw = ImageDraw.Draw(image)

        # Load a font
        font_path = Text.find_font_file(font_name)
        font = ImageFont.truetype(font_path, font_size)

        # Draw text onto the image
        if justify == 'left':
            draw.text((10, texture_size[0]-10), text, font=font, 
                      anchor='lb', fill=color)
        else:
            draw.text((texture_size[0]-10, texture_size[1]-10), text, font=font, 
                    anchor='rb', fill=color)

        # Convert PIL image to OpenGL texture format
        texture_data = np.flipud(image)
        tex = Texture(texture_data, mipmap=True, anisotropic_filtering=2)
        width = height * texture_size[0] / texture_size[1]

        super().__init__(width, height, specular_color=[0,0,0,0], tex=tex, shader=shader, **kwargs)
        self.rotate_x(90) # Make the text face the camera

class AprilTag(TexPlane):

    def __init__(self, id, size, alpha=1, **kwargs):
        filepath = f"riglib/pupillabs/tag36h11/tag36_11_{id:05d}.png"
        apriltag = Texture(filepath, minfilter=GL_NEAREST, magfilter=GL_NEAREST)
        super().__init__(size, size, color=[0,0,0,alpha], specular_color=[0,0,0,0], tex=apriltag)
        self.rotate_x(90)
        
##### 2-D primitives #####

class Shape2D(object):
    '''Abstract base class for shapes that live in the 2-dimension xz-plane
    and are intended only for use with the WindowDispl2D class (not Window).
    '''

    def __init__(self, color, visible=True):
        self.color   = color
        self.visible = visible

    def draw(self, surface, pos2pix_fn):
        '''Draw itself on the given pygame.Surface object using the given
        position-to-pixel_position function.'''

        raise NotImplementedError  # implement in subclasses

    def _recache_xfm(self):
        pass


class Circle(Shape2D):
    def __init__(self, center_pos, radius, *args, **kwargs):
        super(Circle, self).__init__(*args, **kwargs)
        self.center_pos = center_pos
        self.radius     = radius

    def draw(self, surface, pos2pix_fn):
        if self.visible:
            color = tuple([int(255*x) for x in self.color[0:3]])

            pix_pos    = pos2pix_fn(self.center_pos)
            pix_radius = pos2pix_fn([self.radius, 0])[0] - pos2pix_fn([0, 0])[0]
            pygame.draw.circle(surface, color, pix_pos, pix_radius)

        return self.visible  # return True if object was drawn


class Sector(Shape2D):
    def __init__(self, center_pos, radius, ang_range, *args, **kwargs):
        super(Sector, self).__init__(*args, **kwargs)
        self.center_pos = center_pos
        self.radius     = radius
        self.ang_range  = ang_range

    def draw(self, surface, pos2pix_fn):
        if self.visible:
            color = tuple([int(255*x) for x in self.color[0:3]])
            
            arc_angles = np.linspace(self.ang_range[0], self.ang_range[1], 5)
            pts = list(self.center_pos + self.radius*np.c_[np.cos(arc_angles), np.sin(arc_angles)])
            pts.append(self.center_pos)
            
            point_list = list(map(pos2pix_fn, pts))
            pygame.draw.polygon(surface, color, point_list)
        
        return self.visible  # return True if object was drawn


class Line(Shape2D):
    def __init__(self, start_pos, length, width, angle, *args, **kwargs):
        super(Line, self).__init__(*args, **kwargs)
        self.start_pos = start_pos
        self.length    = length
        self.width     = width  # draw a line as thin rectangle
        self.angle     = angle

    def draw(self, surface, pos2pix_fn):
        if self.visible:
            color = tuple([int(255*x) for x in self.color[0:3]])

            # create points and then rotate to correct orientation
            pts = np.array([[          0,  self.width/2], 
                            [          0, -self.width/2], 
                            [self.length, -self.width/2], 
                            [self.length,  self.width/2]])
            rot_mat = np.array([[np.cos(self.angle), -np.sin(self.angle)], 
                                [np.sin(self.angle),  np.cos(self.angle)]])
            pts = np.dot(rot_mat, pts.T).T + self.start_pos
            
            point_list = list(map(pos2pix_fn, pts))
            pygame.draw.polygon(surface, color, point_list)

        return self.visible  # return True if object was drawn
