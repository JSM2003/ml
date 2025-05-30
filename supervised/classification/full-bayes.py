import numpy as np

class BayesClassifer:

    def __init__(self):
        self.n_classes = 2
        self.dist = lambda x, mu, cov : (1/(np.sqrt(2*np.pi*np.linalg.det(cov)))) * np.exp((x-mu).T@np.linalg.inv(cov)@(x-mu))


    def fit(self, X_train, y_train):
        self.n_classes = len(set(y_train))
        self.mu = np.mean(X_train, axis = 0).reshape((self.n_classes,1))
        self.cov = np.cov(X_train.T)
        self.prior = np.zeros(self.n_classes).reshape((self.n_classes,1))

        for i in range(X_train.shape[0]):
            self.prior[y_train[i]] += 1
        
        self.prior /= X_train.shape[0]

    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])

        for i in range(X_test.shape[0]):

            y_pred[i] = np.argmin(-np.log(self.dist(X_test[i].reshape((self.n_classes,1)),self.mu,self.cov) * self.prior))
        
        return y_pred