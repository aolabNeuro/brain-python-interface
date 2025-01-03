'''Needs docs'''


import numpy as np
from OpenGL.GL import *
import pygame
from OpenGL.GL.EXT.texture_filter_anisotropic import *

from .models import Model

textypes = {GL_UNSIGNED_BYTE:np.uint8, GL_FLOAT:np.float32}
class Texture(object):
    def __init__(self, tex, size=None,
        magfilter=GL_LINEAR, minfilter=GL_LINEAR, 
        wrap_x=GL_CLAMP_TO_EDGE, wrap_y=GL_CLAMP_TO_EDGE,
        iformat=GL_RGBA8, exformat=GL_RGBA, dtype=GL_UNSIGNED_BYTE,
        mipmap=False, mipmap_filter=GL_LINEAR_MIPMAP_LINEAR,
        anisotropic_filtering=0):

        self.opts = dict(
            magfilter=magfilter, minfilter=minfilter, 
            wrap_x=wrap_x, wrap_y=wrap_y,
            iformat=iformat, exformat=exformat, dtype=dtype,
            mipmap=mipmap, mipmap_filter=mipmap_filter,
            anisotropic_filtering=anisotropic_filtering)

        if isinstance(tex, np.ndarray):
            if tex.max() <= 1:
                tex *= 255
            if len(tex.shape) < 3:
                tex = np.tile(tex, [3, 1, 1]).T
            if tex.shape[-1] == 3:
                tex = np.dstack([tex, np.ones(tex.shape[:-1])])
            size = tex.shape[:2]
            tex = tex.astype(np.uint8).tobytes()
        elif isinstance(tex, str):
            im = pygame.image.load(tex)
            size = im.get_size()
            tex = pygame.image.tostring(im, 'RGBA', True)
        
        self.texstr = tex
        self.size = size
        self.tex = None

    def init(self):
        if self.tex is not None:
            print(f"Texture already initialized: {self.tex}")
            return

        gltex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, gltex)
            
        # Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, self.opts['wrap_x'])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, self.opts['wrap_y'])
        
        # Set filter parameters based on mipmap option
        if self.opts['mipmap']:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self.opts['mipmap_filter'])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self.opts['magfilter'])
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, self.opts['minfilter'])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, self.opts['magfilter'])
        
        # Apply anisotropic filtering if requested
        if self.opts['anisotropic_filtering'] > 0:
            max_anisotropy = glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT)
            anisotropy = min(self.opts['anisotropic_filtering'], max_anisotropy)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, anisotropy)

        # Ensure width and height are integers
        width, height = int(self.size[0]), int(self.size[1])
        
        # Create and fill texture
        glTexImage2D(
            GL_TEXTURE_2D, 0,
            self.opts['iformat'],
            width, height, 0,
            self.opts['exformat'], self.opts['dtype'],
            self.texstr
        )
        
        # Generate mipmaps if requested
        if self.opts['mipmap']:
            glGenerateMipmap(GL_TEXTURE_2D)
        
        error = glGetError()
        if error != GL_NO_ERROR:
            print(f"OpenGL error after texture creation: {error}")
        else:
            print(f"Texture initialized successfully: {gltex}")
        
        self.tex = gltex
    
    def set(self, idx):
        glActiveTexture(GL_TEXTURE0+idx)
        glBindTexture(GL_TEXTURE_2D, self.tex)
    
    def get(self, filename=None):
        current = glGetInteger(GL_TEXTURE_BINDING_2D)
        glBindTexture(GL_TEXTURE_2D, self.tex)
        texstr = glGetTexImage(GL_TEXTURE_2D, 0, self.opts['exformat'], self.opts['dtype'])
        glBindTexture(GL_TEXTURE_2D, current)
        im = np.fromstring(texstr, dtype=textypes[self.opts['dtype']])
        im.shape = (self.size[1], self.size[0], -1)
        if filename is not None:
            np.save(filename, im)
        return im
    
    def delete(self):
        if self.tex is not None:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDeleteTextures(1, [self.tex])
            error = glGetError()
            if error != GL_NO_ERROR:
                print(f"Error after deleting texture: {error}")

class MultiTex(object):
    '''This is not ready yet!'''
    def __init__(self, textures, weights):
        raise NotImplementedError
        assert len(textures) < max_multitex
        self.texs = textures
        self.weights = weights

class TexModel(Model):
    def __init__(self, tex=None, **kwargs):
        if tex is not None:
            kwargs['color'] = (0,0,0,1)
        super(TexModel, self).__init__(**kwargs)
        
        self.tex = tex
    
    def init(self):
        super(TexModel, self).init()
        if self.tex.tex is None:
            self.tex.init()
        
    def render_queue(self, shader=None, **kwargs):
        if shader is not None:
            yield shader, self.draw, self.tex
        else:
            yield self.shader, self.draw, self.tex