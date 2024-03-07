import numpy as np


def stimulus_std_schedule(N: int, init_std: float = 0.01):
    std = np.zeros((N, ))
    std[0] = init_std
    
    for i in range(1, N):
        if std[i-1] < 0.25:
            std[i] = std[i-1] * 1.5
        else:
            std[i] = std[i-1] * 1.25
    
    return std
