import numpy as np

class NBClassifer:

    def __init__(self):
        self.n_classes = 2
        self.dist = lambda x, mu, sig : (1/(np.sqrt(2*np.pi)*sig)) * np.exp((x-mu)**2 / (2*sig**2))


    def fit(self, X_train, y_train):
        self.n_classes = len(set(y_train))
        self.mu = np.mean(X_train, axis = 0)
        self.sig = np.std(X_train, axis = 0)
        self.prior = np.zeros(self.n_classes)

        for i in range(X_train.shape[0]):
            self.prior[y_train[i]] += 1
        
        self.prior /= X_train.shape[0]

    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])

        for i in range(X_test.shape[0]):
            y_pred[i] = np.argmin(-np.log(self.dist(X_test[i],self.mu,self.sig) * self.prior))
        
        return y_pred