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
                tex = (tex * 255).astype(np.uint8)
            else:
                tex = tex.astype(np.uint8)
            if tex.ndim == 2:
                tex = np.stack([tex]*3, axis=-1)  # grayscale → RGB
            elif tex.shape[-1] == 1:
                tex = np.repeat(tex, 3, axis=-1)
            if size is None:
                size = (tex.shape[1], tex.shape[0])
            tex = np.ascontiguousarray(tex.astype(np.uint8)).tobytes()
        elif isinstance(tex, str):
            im = pygame.image.load(tex)
            size = im.get_size()
            tex = pygame.image.tostring(im, 'RGBA', True)
        
        self.texstr = tex
        self.size = size
        self.tex = None

    def update(self, tex, size=None):
        if not isinstance(tex, np.ndarray):
            raise TypeError("Texture.update expects a numpy ndarray")

        if tex.max() <= 1:
            tex = (tex * 255).astype(np.uint8)
        else:
            tex = tex.astype(np.uint8)

        if tex.ndim == 2:
            tex = np.stack([tex] * 3, axis=-1)
        elif tex.shape[-1] == 1:
            tex = np.repeat(tex, 3, axis=-1)

        if size is None:
            size = (tex.shape[1], tex.shape[0])

        tex = np.ascontiguousarray(tex.astype(np.uint8))
        tex_bytes = tex.tobytes()

        if self.tex is None:
            self.size = size
            self.texstr = tex_bytes
            self.init()
            return

        width, height = int(size[0]), int(size[1])
        needs_realloc = tuple(size) != tuple(self.size)

        glBindTexture(GL_TEXTURE_2D, self.tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        if needs_realloc:
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                self.opts['iformat'],
                width,
                height,
                0,
                self.opts['exformat'],
                self.opts['dtype'],
                tex_bytes,
            )
            self.size = size
        else:
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                width,
                height,
                self.opts['exformat'],
                self.opts['dtype'],
                tex_bytes,
            )

        if self.opts['mipmap']:
            glGenerateMipmap(GL_TEXTURE_2D)

        self.texstr = tex_bytes

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

        # Avoid row-stride artifacts for RGB textures whose row width is not 4-byte aligned.
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        
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

    def release(self):
        self.tex.delete()

    def replace_texture(self, new_tex):
        self.tex.delete()
        self.tex = new_tex
        self.tex.init()