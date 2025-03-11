import numpy as np
# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


def direction_to_change(direction):
    if direction==0:
        change = [0,0]
    if direction==1:
        change = [0,-1]
    if direction==2:
        change = [1,0]
    if direction==3:
        change = [0,1]
    if direction==4:
        change = [-1,0]
    return np.array(change)