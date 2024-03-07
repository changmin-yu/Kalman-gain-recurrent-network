Python implementation of [Dayan & Kakade, 2002](https://github.com/changmin-yu/Kalman-gain-recurrent-network/tree/main).

Two major components of the implementation.
- Relationship between Kalman filter (specifically the mean covariance matrix) and the whitening filter for the input correlation matrix. See this [notebook](Kalman_filters.ipynb).
- Augmented delta rule for training feedforward weights in a recurrent network based on Kalman gain matrix. See this [notebook](network_implementation.ipynb).
