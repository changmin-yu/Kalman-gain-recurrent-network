import numpy as np


class KalmanRecurrentNetwork:
    def __init__(
        self, 
        D: int, 
        learning_rate: float, 
        dt: float, 
        threshold: float = 1e-5, 
    ):
        
        self.B = np.random.uniform(size=(D, D))
        self.w = np.random.uniform(size=(2, ))
    
        self.y = np.zeros((D, ))
        
        self.learning_rate = learning_rate
        self.dt = dt
        self.threshold = threshold
        
        self.converged = False
        
    def step(self, x: np.ndarray):
        dy = (-self.y + x + self.B.dot(self.y)) / self.dt
        new_y = self.y + dy
        if np.sqrt(np.sum(np.square(new_y - self.y))) < self.threshold:
            self.converged = True
        
        self.y = new_y
        
        self.update_recurrent(x)
    
    def update_recurrent(self, x: np.ndarray):
        dB = -np.outer(x, self.y) + np.eye(x.shape[-1]) - self.B
        self.B += self.learning_rate * dB
    
    def update_feedforward(self, r: float, x: np.ndarray):
        dw = self.y * (r - np.dot(self.w, x)) # assume y has converged to y(\infty)
        self.w += self.learning_rate * dw


def train_network(
    x: np.ndarray, 
    r: np.ndarray, 
    learning_rate: float, 
    dt: float, 
    threshold: float = 1e-5, 
    max_iters: int = 100, 
):
    N, D = x.shape
    network = KalmanRecurrentNetwork(D, learning_rate, dt, threshold)
    
    w_history = np.zeros((N, D))
    B_history = np.zeros((N, D, D))
    
    for i in range(N):
        steps = 0
        while not network.converged or steps < max_iters:
            network.step(x[i])
            steps += 1
        
        network.update_feedforward(r[i], x[i])
        
        w_history[i] = network.w
        B_history[i] = network.B
    
    return network, w_history, B_history
