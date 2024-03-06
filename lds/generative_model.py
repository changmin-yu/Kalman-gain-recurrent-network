import numpy as np


def lds_generative(
    transition_std: float, 
    obs_std: float, 
    init_std: float, 
    stimuli: np.ndarray, 
):
    N, D = stimuli.shape
    
    w = [np.random.normal(0, scale=init_std, size=(D, ))]
    r = []
    
    for t in range(N):
        x = stimuli[t]
        w_temp = w[-1].copy()
        w.append(np.copy(w_temp + np.random.normal(0, scale=transition_std, size=(D, ))))
        r.append(np.copy(w_temp.dot(x) + np.random.normal(0, scale=obs_std)))
    
    return w, r
