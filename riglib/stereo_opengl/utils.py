
'''Needs docs'''
import numpy as np
from .textures import Texture

def frustum(l, r, t, b, n, f):
    '''
    This function emulates glFrustum: https://www.opengl.org/sdk/docs/man2/xhtml/glFrustum.xml
    A frustum is a solid cut by planes, e.g., the planes representing the viewable area of a screen.

    Parameters
    ----------
    l: float
        Distance to the left plane of the screen
    r: float
        Distance to the right plane of the screen
    t: float
        Distance to the top plane of the screen
    b: float
        Distance to the bottom plane of the screen
    n: float
        Distance to the near plane of the screen
    f: float
        Distance to the far plane of the screen

    Returns
    -------
    Projection matrix to apply to solid to truncate

    '''
    rl, nrl = r + l, r - l
    tb, ntb = t + b, t - b
    fn, nfn = f + n, f - n
    return np.array([[2*n / nrl,     0,     rl / nrl,       0],
                     [    0,     2*n / ntb, tb / ntb,       0],
                     [    0,         0,    -fn / nfn, -2*f*n / nfn],
                     [    0,         0,       -1,           0]])

def perspective(angle, aspect, near, far):
    '''Generates a perspective transform matrix'''
    f = 1./ np.tan(np.radians(angle) / 2)
    fn = far + near
    nfn = far - near
    return np.array([[f/aspect, 0,    0,               0],
                     [0,        f,    0,               0],
                     [0,        0, -fn/nfn, -2*far*near/nfn],
                     [0,        0,   -1,               0]])

def orthographic(w, h, near, far):
    fn = far + near
    nfn = far - near
    return np.array([[2/w, 0,   0,      0],
                     [0,   2/h, 0,      0],
                     [0,   0,   -2/nfn, -fn/nfn],
                     [0,   0,   0,       1]])

def offaxis_frusta(winsize, fov, near, far, focal_dist, iod, flip=False, flip_z=False):
    aspect = winsize[0] / winsize[1]
    top = near * np.tan(np.radians(fov) / 2)
    right = aspect*top
    fshift = (iod/2) * near / focal_dist

    # calculate the perspective matrix for the left eye and for the right eye
    left = frustum(-right+fshift, right+fshift, top, -top, near, far)
    right = frustum(-right-fshift, right-fshift, top, -top, near, far)
    
    # multiply in the iod (intraocular distance) modelview transform
    lxfm, rxfm = np.eye(4), np.eye(4)
    lxfm[:3,-1] = [0.5*iod, 0, 0]
    rxfm[:3,-1] = [-0.5*iod, 0, 0]
    flip_mat = np.eye(4)


    if flip:
        flip_mat[0,0] = -1
    if flip_z:
        flip_mat[1,1] = -1

    return np.dot(flip_mat, np.dot(left, lxfm)), np.dot(flip_mat, np.dot(right, rxfm))

    #return np.dot(left, lxfm), np.dot(right, rxfm)

def cloudy_tex(size=(512,512), alpha=0.5):
    '''Generates 1/f distributed noise and puts it into a texture. Looks like clouds'''
    im = np.random.randn(*size)
    grid = np.mgrid[-1:1:size[0]*1j, -1:1:size[1]*1j]
    mask = 1/(grid**2).sum(0)
    fim = np.fft.fftshift(np.fft.fft2(im))
    im = np.abs(np.fft.ifft2(np.fft.fftshift(mask * fim)))
    im -= im.min()
    im /= im.max()
    im = np.tile(im, (4,1,1)).T
    im[:,:,3] = alpha
    return Texture(im)

def create_grid_texture(size=800, density=50, thickness=3, line_color=[0.5, 0.5, 0.5, 1], 
                        background_color=[0.1, 0.1, 0.1, 1]):
    """
    Create a grid texture.
    """
    size = int(size)
    thickness = int(thickness)
    grid_texture = np.ones((size, size, 4)) * background_color  # RGB image
    num_lines = (size + 1)//density
    line_spacing = size//num_lines

    # Draw horizontal grid lines
    for r in range(num_lines):
        start = int(r * line_spacing - thickness//2)
        grid_texture[start:start+thickness, :, :] = line_color
        grid_texture[:, start:start+thickness, :] = line_color

    return Texture(grid_texture/np.max(grid_texture), mipmap=True, anisotropic_filtering=2)

def look_at(eye, target, up):
    # Convert inputs to numpy arrays
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)

    # Calculate forward direction
    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    # Calculate right direction
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # Recalculate up direction
    up = np.cross(right, forward)

    # Create rotation matrix
    rotation = np.array([
        [right[0], right[1], right[2], 0],
        [up[0], up[1], up[2], 0],
        [-forward[0], -forward[1], -forward[2], 0],
        [0, 0, 0, 1]
    ])

    # Create translation matrix
    translation = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ])

    # Combine rotation and translation
    return np.dot(rotation, translation)