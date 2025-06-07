import numpy as np

class LogisticRegressor:
    def __init__(self):
        self.n_classes = 2
    
    def fit(self, X_train, y_train):
        self.n_classes = len(set(y_train))
        self.w = np.zeros(tuple(X_train.shape[1:]+1))

        

        if(self.n_classes == 2):
            self.fn = lambda x: 1/(1+np.exp(np.sum(-self.w * np.array([1]+list(x)))))
        
        else:
            self.fn = lambda x : np.exp(self.w * np.array([1]+list(x))) / np.sum(np.exp(self.w * np.array([1]+list(x))))
        
