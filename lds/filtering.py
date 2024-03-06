import numpy as np


def lds_filtering(
    obs: np.ndarray, 
    stimuli: np.ndarray, 
    init_std: float, 
    obs_std: float, 
    transition_std: float
):
    N, D = stimuli.shape
    
    mu, Sigma = [np.zeros(D)], [np.eye(D) * (init_std ** 2)]
    
    for t in range(N):
        x = stimuli[t]
        r = obs[t]
        mu_temp = mu[-1].copy()
        
        mu.append(
            np.copy(mu_temp + (Sigma[-1].dot(x) / (np.dot(x, np.dot(Sigma_temp, x)) + obs_std ** 2)) * (r - mu_temp.dot(x)))
        )
        
        # scheme 1 (same as the paper)
        # Sigma_temp = Sigma[-1].copy() 
        # Sigma.append(
        #     np.copy(Sigma_temp + np.eye(D) * (transition_std ** 2) - np.dot(Sigma_temp, np.dot(np.outer(x, x), Sigma_temp)) / (np.dot(x, np.dot(Sigma_temp, x)) + obs_std ** 2))
        # )
        
        # scheme 2, including the latent transition covariance
        Sigma_temp = Sigma[-1].copy() + (transition_std ** 2) * np.eye(D)
        Sigma.append(
            np.copy(Sigma_temp - np.matmul(Sigma_temp, np.matmul(np.outer(x, x), Sigma_temp)) / (np.dot(x, np.dot(Sigma_temp, x)) + obs_std ** 2))
        )
        
    return mu, Sigma
