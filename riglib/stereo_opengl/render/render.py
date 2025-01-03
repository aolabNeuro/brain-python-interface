'''Needs docs'''


import os
import operator
import numpy as np
from OpenGL.GL import *

from ..utils import perspective, orthographic
from .shader import ShaderProgram

cwd = os.path.join(os.path.abspath(os.path.split(__file__)[0]), "..")

class _textrack(object):
    pass

class Renderer(object):
    def __init__(self, window_size, fov, near, far, modelview=None, shaders=None, programs=None, light_direction=None):
        self.render_queue = None
        self.size = window_size
        self.drawpos = 0,0
        w, h = window_size
        self.projection = perspective(fov, w / h, near, far)
        self.light_direction = light_direction or (-1., -2., -2., 0.)
        self.modelview = np.eye(4) if modelview is None else modelview

        #Add the default shaders
        if shaders is None:
            shaders = dict()
            shaders['passthru'] = GL_VERTEX_SHADER, "passthrough.v.glsl"
            shaders['default'] = GL_FRAGMENT_SHADER, "default.f.glsl", "phong.f.glsl"
            shaders['ui_passthru'] = GL_VERTEX_SHADER, "passthrough2d.v.glsl"
            shaders['ui_default'] = GL_FRAGMENT_SHADER, "none.f.glsl"
        if programs is None:
            programs = dict()
            programs['default'] = "passthru", "default"
            programs['ui'] = "ui_passthru", "ui_default"

        #compile the given shaders and the programs
        self.shaders = dict()
        for k, v in list(shaders.items()):
            # print("Compiling shader %s..."%k)
            self.add_shader(k, *v)
        
        self.programs = dict()
        for name, shaders in list(programs.items()):
            self.add_program(name, shaders)
        
        #Set up the texture units
        self.reset_texunits()

        #Generate the default fullscreen quad
        verts = np.array([(-1,-1,0,1), (1,-1,0,1), (1,1,0,1), (-1,1,0,1)]).astype(np.float32)
        polys = np.array([(0,1,2),(0,2,3)]).astype(np.uint16)
        vbuf = glGenBuffers(1)
        ebuf = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbuf)
        glBufferData(GL_ARRAY_BUFFER, verts, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebuf)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, polys, GL_STATIC_DRAW)
        self.fsquad_buf = vbuf, ebuf
    
    def _queue_render(self, root, shader=None):
        '''

        Parameters
        ----------
        root: stereo_opengl.models.Model instance
            The root model from which to start drawing
        shader: ???????, default=None
            ????????
        '''
        queue = dict((k, dict()) for k in list(self.programs.keys()))

        for pname, drawfunc, tex in root.render_queue(shader=shader):
            if tex not in queue[pname]:
                queue[pname][tex] = []
            queue[pname][tex].append(drawfunc)
        
        for pname in list(self.programs.keys()):
            #assert len(self.texavail) > len(queue[pname])
            for tex in list(queue[pname].keys()):
                if tex is not None:
                    self.get_texunit(tex)
        
        self.render_queue = queue
    
    def get_texunit(self, tex):
        '''Input a Texture object, output a tuple (index, TexUnit)'''
        if tex not in self.texunits:
            unit = self.texavail.pop()
            glActiveTexture(unit[1])
            if tex == "None":
                glBindTexture(GL_TEXTURE_2D, 0)
            else:
                glBindTexture(GL_TEXTURE_2D, tex.tex)
            #print "Binding %r to %d"%(tex, unit[0])
            self.texunits[tex] = unit[0]
        
        return self.texunits[tex]
    
    def reset_texunits(self):
        maxtex = glGetIntegerv(GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS)
        # print(f"Max texture units: {maxtex}")
        # glActiveTexture(GL_TEXTURE0)  # Reset to default texture unit
        self.texavail = set((i, globals()['GL_TEXTURE%d'%i]) for i in range(1, maxtex))
        self.texunits = dict() 

    def add_shader(self, name, stype, filename, *includes):
        src = []
        main_path = os.path.join(cwd, "shaders", filename)
        with open(main_path, 'r') as main:
            version = main.readline().strip()
            main_content = main.read()
        
        for inc in includes:
            inc_path = os.path.join(cwd, "shaders", inc)
            with open(inc_path, 'r') as incfile:
                ver = incfile.readline().strip()
                assert ver == version, f"Version mismatch: {ver} != {version}"
                src.append(incfile.read())
        
        src.append(main_content)
        
        full_src = "\n".join([version] + src)
        shader = glCreateShader(stype)
        glShaderSource(shader, full_src)
        glCompileShader(shader)

        if not glGetShaderiv(shader, GL_COMPILE_STATUS):
            err = glGetShaderInfoLog(shader)
            glDeleteShader(shader)
            print(f"Shader compilation error for {filename}:")  # Debug print
            print(err.decode('utf-8'))  # Debug print
            raise Exception(err)
        
        self.shaders[name] = shader
    
    def add_program(self, name, shaders):
        shaders = [self.shaders[i] for i in shaders]
        sp = ShaderProgram(shaders)
        self.programs[name] = sp
    
    def draw(self, root, shader=None, apply_default=False, requeue=False, **kwargs):
        if self.render_queue is None or requeue:
            self._queue_render(root)
        
        if "p_matrix" not in kwargs:
            kwargs['p_matrix'] = self.projection
        if "modelview" not in kwargs:
            kwargs['modelview'] = self.modelview
        if "light_direction" not in kwargs:
            kwargs['light_direction'] = self.light_direction
        
        if shader is not None:
            if apply_default: # apply this shader to all models that don't have a program specified
                self.programs[shader].draw(self, self.render_queue['default'], **kwargs)
            else: # apply this shader to only the models that have this shader specified
                self.programs[shader].draw(self, self.render_queue[shader], **kwargs)
        else:
            for name, program in list(self.programs.items()):
                program.draw(self, self.render_queue[name], **kwargs)

        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL Error: {error}")
    
    def draw_done(self):
        self.reset_texunits()

class Renderer2D(Renderer):

    def __init__(self, screen_cm):
        shaders = dict()
        shaders['passthru'] = GL_VERTEX_SHADER, "passthrough2d.v.glsl"
        shaders['default'] = GL_FRAGMENT_SHADER, "none.f.glsl"
        programs = dict()
        programs['default'] = "passthru", "default"
        programs['ui'] = "passthru", "default"
        super().__init__(screen_cm, np.nan, 1, 1024, shaders=shaders, programs=programs)
        w, h = screen_cm
        self.projection = orthographic(w, h, 1, 1024)

def test():
    import pygame
    pygame.init()
    pygame.display.set_mode((100,100), pygame.OPENGL | pygame.DOUBLEBUF)

    return Renderer(
                shaders=dict(
                    passthru=(GL_VERTEX_SHADER, "passthrough.v.glsl"),
                    phong=(GL_FRAGMENT_SHADER, "default.f.glsl", "phong.f.glsl")), 
                programs=dict(
                    default=("passthru", "phong"),
                )
            )