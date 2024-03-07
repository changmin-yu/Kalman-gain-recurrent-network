import numpy as np


def lds_filtering(
    obs: np.ndarray, 
    stimuli: np.ndarray, 
    init_std: float, 
    obs_std: float, 
    transition_std: float
):
    N, D = stimuli.shape
    
    mu = np.zeros((N+1, D))
    Sigma = np.zeros((N+1, D, D))
    
    mu[0, :] = np.zeros((D, ))
    Sigma[0, :] = np.eye(D) * (init_std ** 2)
    
    for t in range(N):
        x = stimuli[t]
        r = obs[t]
        
        mu[t+1] = mu[t] + (Sigma[t].dot(x) / (np.dot(x, np.dot(Sigma[t], x)) + obs_std ** 2)) * (r - mu[t].dot(x))
        
        # scheme 1 (same as the paper)
        Sigma[t+1] = Sigma[t] + np.eye(D) * (transition_std ** 2) - np.matmul(Sigma[t], np.matmul(np.outer(x, x), Sigma[t])) / (np.dot(x, np.dot(Sigma[t], x)) + obs_std ** 2)
        
        # # scheme 2, including the latent transition covariance
        # Sigma_temp = Sigma[t] + (transition_std ** 2) * np.eye(D)
        # Sigma[t+1] = np.copy(Sigma_temp - np.matmul(Sigma_temp, np.matmul(np.outer(x, x), Sigma_temp)) / (np.dot(x, np.dot(Sigma_temp, x)) + obs_std ** 2))
        
    return mu, Sigma
