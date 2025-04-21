import numpy as np

class KalmanBeta:
    """
    EMA-initialised Kalman filter for online beta estimation between two series.
    """
    def __init__(self, ema_alpha=0.01, q=1e-5, r=1e-2):
        self.ema_alpha = ema_alpha
        self.q = q  # process noise
        self.r = r  # measurement noise
        self.beta = 0.0
        self.p = 1.0
        self.initialized = False
    def fit(self, x, y):
        # x, y: arrays for initial OLS/EMA
        if len(x) < 2:
            self.beta = 1.0
        else:
            cov = np.cov(x, y)
            self.beta = cov[0,1] / (cov[0,0] + 1e-8)
        self.initialized = True
    def update(self, x, y):
        if not self.initialized:
            self.fit(x, y)
        # Kalman update for last point
        if len(x) < 1:
            return self.beta
        x_t, y_t = x[-1], y[-1]
        pred = self.beta * x_t
        e = y_t - pred
        k = self.p * x_t / (self.r + self.p * x_t**2)
        self.beta += k * e
        self.p = (1 - k * x_t) * self.p + self.q
        return self.beta
    def get_beta(self):
        return self.beta
