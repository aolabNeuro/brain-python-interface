import numpy as np

def calculate_square_positions(half_height, window_size, n, square_size):
    '''
    Generate the x,y positions of squares tangent to the window perimeter
    '''

    pixel_to_cm_ratio =  2 * half_height / window_size[1]
    width = window_size[0] * pixel_to_cm_ratio - square_size
    height = window_size[1] * pixel_to_cm_ratio - square_size
    n_rounded = 4 * np.ceil(n / 4) # make spacing always a multiple of 4
    horizontal_spacing = (2 * width) / (n_rounded / 2)
    vertical_spacing = (2 * height) / (n_rounded / 2)

    print(horizontal_spacing, vertical_spacing)

    centers = []
    current_distance = 0

    for _ in range(n):
        if current_distance < width:                    # Top edge
            x = current_distance - width / 2 - square_size / 2
            y = height / 2 - square_size / 2
            print('top', current_distance, x, y)
            current_distance += horizontal_spacing
        elif current_distance < width + height:         # Right edge
            x = width / 2 - square_size / 2
            y = height / 2 - (current_distance - width) - square_size / 2
            print('right', current_distance, x, y)
            current_distance += vertical_spacing
        elif current_distance < 2 * width + height:     # Bottom edge
            x = width / 2 - (current_distance - width - height) - square_size / 2
            y = - height / 2 - square_size / 2
            print('bottom', current_distance, x, y)
            current_distance += horizontal_spacing
        else:                                           # Left edge
            x = - width / 2 - square_size / 2
            y = - height / 2 + (current_distance - 2 * width - height) - square_size / 2
            print('left', current_distance, x, y)
            current_distance += vertical_spacing

        centers.append((x, y))

    return centers