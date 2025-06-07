import numpy as np

class GradientDescent:
    def __init__(self, X,w_0,method,lr):
        self.w = w_0
        self.method = method
        self.lr = lr
        self.X = X

    def vanilla(self):
        tol = 1e-4

        loss = np.sum([(self.X[i]-self.w)**2 for i in range(self.X.shape[0])])

        while(loss > tol):
            self.w -= self.lr * np.gradient(loss)
            self.lr /= 2
            loss = np.sum([(self.X[i]-self.w)**2 for i in range(self.X.shape[0])])
        
    