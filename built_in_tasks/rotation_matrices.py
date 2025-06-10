import numpy as np

rotations = dict(
    yzx = np.array(    # names come from rows (optitrack), but screen coords come from columns:
        [[0, 1, 0, 0], # x goes into second column (y-coordinate, coming out of screen)
        [0, 0, 1, 0],  # y goes into third column (z-coordinate, up)
        [1, 0, 0, 0],  # z goes into first column (x-coordinate, right)
        [0, 0, 0, 1]]
    ),
    zyx = np.array(
        [[0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [1, 0, 0, 0], 
        [0, 0, 0, 1]]
    ),
    xzy = np.array(
        [[1, 0, 0, 0],
        [0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1]]
    ),
    xyz = np.identity(4),
)

baseline_rotations = dict(
    none = np.identity(4), # vertical workspace
    horizontal_workspace = np.array(
        [[1, 0, 0, 0], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1]]
    ),    
)

exp_rotations = dict(
    none = np.identity(4),
    about_x_90 = np.array(
        [[1, 0, 0, 0], 
        [0, 0, 1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1]]
    ),
    about_x_minus_90 = np.array(
        [[1, 0, 0, 0], 
        [0, 0, -1, 0], 
        [0, 1, 0, 0], 
        [0, 0, 0, 1]]
    ),
    oop_xy_45 = np.array(
        [[ 0.707,  0.5  ,  0.5  , 0.],
         [ 0.   ,  0.707, -0.707, 0.],
         [-0.707,  0.5  ,  0.5  , 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ),
    oop_xy_minus_45 = np.array(
        [[ 0.707,  0.5  , -0.5  , 0.],
         [ 0.   ,  0.707,  0.707, 0.],
         [ 0.707, -0.5  ,  0.5  , 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ),
    oop_xy_20 = np.array(
        [[ 0.94 ,  0.117,  0.321, 0.],
         [-0.   ,  0.94 , -0.342, 0.],
         [-0.342,  0.321,  0.883, 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ),
    oop_xy_minus_20 = np.array(
        [[ 0.94 ,  0.117, -0.321, 0.],
         [-0.   ,  0.94 ,  0.342, 0.],
         [ 0.342, -0.321,  0.883, 0.],
         [ 0.,     0.   ,  0.,    1.]]
    ),
    upright_xz_45 = np.array(
        [[ 0.707, 0.5,   0.5,   0 ],
        [-0.707, 0.5,   0.5 ,   0 ],
        [ 0.,   -0.707, 0.707 , 0 ],
        [ 0.,    0.,    0.,     1 ]]
    ),
    flat_xz_45 = np.array(
        [[ 0.707, 0.5 ,  0.5 , 0. ],
        [-0.707 ,0.5  , 0.5  , 0. ],
        [ 0.   ,-0.707, 0.707, 0. ],
        [ 0.   , 0.   , 0.   , 1. ]]
    ),
)
