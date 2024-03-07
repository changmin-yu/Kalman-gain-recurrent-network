import numpy as np


class KalmanRecurrentNetwork:
    def __init__(
        self, 
        D: int, 
        learning_rate: float, 
        dt: float, 
        threshold: float = 1e-5, 
    ):
        
        # initialise parameters
        # B = np.random.randn(D, D) * 0.1
        # self.B = B + B.T
        self.B = np.zeros((D, D))
        self.w = np.zeros((2, ))

        # initialise internal states in recurrent network
        self.y = np.zeros((D, ))
        
        self.learning_rate = learning_rate
        self.dt = dt
        self.threshold = threshold
        
        self.converged = False
    
    def step(self, x: np.ndarray, r: float):
        D = x.shape[-1]
        self.y = np.matmul(np.linalg.inv(np.eye(D) - self.B), x)
        
        self.update_feedforward(r, x)
        self.update_recurrent(x)
        
    def recurrent_step(self, x: np.ndarray):
        dy = (-self.y + x + self.B.dot(self.y)) / self.dt
        new_y = self.y + dy
        if np.sqrt(np.sum(np.square(new_y - self.y))) < self.threshold:
            self.converged = True
        
        self.y = new_y
        
        self.update_recurrent(x)
    
    def update_recurrent(self, x: np.ndarray):
        dB = -np.outer(x, self.y) + np.eye(x.shape[-1]) - self.B
        self.B += self.learning_rate * dB
        self.B = 0.5 * (self.B + self.B.T)
    
    def update_feedforward(self, r: float, x: np.ndarray):
        dw = self.y * (r - np.dot(self.w, x)) # assume y has converged to y(\infty)
        self.w += self.learning_rate * dw


def train_network(
    x: np.ndarray, 
    r: np.ndarray, 
    learning_rate: float, 
    dt: float, 
    threshold: float = 1e-5, 
):
    N, D = x.shape
    network = KalmanRecurrentNetwork(D, learning_rate, dt, threshold)
    
    w_history = np.zeros((N, D))
    B_history = np.zeros((N, D, D))
    
    for i in range(N):
        network.step(x[i], r[i])
        
        w_history[i] = network.w
        B_history[i] = network.B
    
    return network, w_history, B_history
