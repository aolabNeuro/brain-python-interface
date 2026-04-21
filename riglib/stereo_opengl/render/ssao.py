'''
Screen Space Ambient Occlusion (SSAO) Renderer

This module implements a Screen Space Ambient Occlusion renderer, which is a technique
used to approximate ambient occlusion in real-time. SSAO adds depth and realism to 3D
scenes by darkening areas that are occluded or surrounded by other geometry.

The SSAO class extends FBOrender to provide a multi-pass rendering pipeline:
1. Render scene to obtain normal and depth information
2. Calculate ambient occlusion based on the normal and depth data
3. Apply the ambient occlusion to the final render
'''

import numpy as np
from OpenGL.GL import *

from .render import Renderer
from .fbo import FBOrender, FBO
from ..textures import Texture

class SSAO(FBOrender):
    def __init__(self, *args, sf=1, **kwargs):
        super(SSAO, self).__init__(*args, **kwargs)
        self.sf = sf
        w, h = self.size[0] / self.sf, self.size[1] / self.sf
        
        self.normdepth = FBO(["color0", "depth"], size=(w,h))
        self.ping = FBO(['color0'], size=(w,h))
        self.pong = FBO(["color0"], size=(w,h))

        self.add_shader("fsquad", GL_VERTEX_SHADER, "fsquad.v.glsl")
        self.add_shader("ssao_pass1", GL_FRAGMENT_SHADER, "ssao_pass1.f.glsl")
        self.add_shader("ssao_pass2", GL_FRAGMENT_SHADER, "ssao_pass2.f.glsl")
        self.add_shader("ssao_pass3", GL_FRAGMENT_SHADER, "ssao_pass3.f.glsl", "phong.f.glsl")
        self.add_shader("hblur", GL_FRAGMENT_SHADER, "hblur.f.glsl")
        self.add_shader("vblur", GL_FRAGMENT_SHADER, "vblur.f.glsl")

        #override the default shader with this passthru + ssao_pass1 to store depth
        self.add_program("ssao_pass1", ("passthru", "ssao_pass1"))
        self.add_program("ssao_pass2", ("fsquad", "ssao_pass2"))
        self.add_program("hblur", ("fsquad", "hblur"))
        self.add_program("vblur", ("fsquad", "vblur"))
        self.add_program("ssao_pass3", ("passthru", "ssao_pass3"))

        # Some debug shaders
        # self.add_shader("fsquad", GL_VERTEX_SHADER, "fsquad.v.glsl")
        # self.add_shader("fsquad_frag", GL_FRAGMENT_SHADER, "fsquad.f.glsl")
        # self.add_program("none", ("fsquad", "fsquad_frag"))

        randtex = np.random.rand(3, int(w), int(h))
        randtex /= randtex.sum(0)
        self.rnm = Texture(randtex.T, wrap_x=GL_REPEAT, wrap_y=GL_REPEAT, 
            magfilter=GL_NEAREST, minfilter=GL_NEAREST)
        self.rnm.init()

        self.clips = args[2], args[3]

    def draw(self, root, **kwargs):
        # Save the current viewport
        original_viewport = glGetIntegerv(GL_VIEWPORT)
        original_framebuffer = glGetIntegerv(GL_FRAMEBUFFER_BINDING)

        # Set the new viewport for SSAO calculations
        new_viewport = (0, 0, self.size[0]//self.sf, self.size[1]//self.sf)
        glViewport(*new_viewport)

        # First, draw the whole scene, but only read the normals and depth into ssao
        self.draw_to_fbo(self.normdepth, root, shader="ssao_pass1", apply_default=True, **kwargs)
        
        # Now, do the actual ssao calculations, and draw it into pong
        self.draw_fsquad_to_fbo(self.pong, "ssao_pass2", rnm=self.rnm,
            normalMap=self.normdepth['color0'], depthMap=self.normdepth['depth'],
            nearclip=self.clips[0], farclip=self.clips[1] )
        
        # Blur the textures
        self.draw_fsquad_to_fbo(self.ping, "hblur", tex=self.pong['color0'], blur=1./(self.size[0]/self.sf))
        self.draw_fsquad_to_fbo(self.pong, "vblur", tex=self.ping['color0'], blur=1./(self.size[0]/self.sf))
        
        # glViewport(*original_viewport)
        # glBindFramebuffer(GL_FRAMEBUFFER, original_framebuffer)
        # self.draw_fsquad("none", tex=self.pong['color0'])
        # return

        # Restore the original viewport
        glViewport(*original_viewport)
        glBindFramebuffer(GL_FRAMEBUFFER, original_framebuffer)
        super(SSAO, self).draw(root, shader="ssao_pass3", apply_default=True, shadow=self.pong['color0'], 
            window=[float(i) for i in original_viewport], **kwargs)
        super(SSAO, self).draw(root, shader="ui", **kwargs)
        
    def clear(self):
        self.normdepth.clear()
        self.ping.clear()
        self.pong.clear()
    
    def draw_done(self):
        super(SSAO, self).draw_done()
        self.clear()
