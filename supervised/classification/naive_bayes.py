import numpy as np

class NBClassifer:

    def __init__(self):
        self.n_classes = 2
        self.dist = lambda x, mu, var : np.prod((1/(np.sqrt(2*np.pi*(var)))) * np.exp(-(x-mu)**2 / (2*var)))


    def fit(self, X_train, y_train):
        self.n_classes = len(set(y_train))
        self.mu = np.empty(tuple([self.n_classes]+list(X_train.shape[1:])))
        self.var = np.empty(tuple([self.n_classes]+list(X_train.shape[1:])))
        self.prior = np.zeros((self.n_classes,1))

        for i in range(self.n_classes):
            self.mu[i] = np.mean(X_train[y_train==i], axis=0)
            self.var[i] = np.var(X_train[y_train==i], axis=0)

        for i in range(X_train.shape[0]):
            self.prior[y_train[i]] += 1
        
        self.prior /= X_train.shape[0]

    def predict(self, X_test):
        y_pred = np.empty(X_test.shape[0])

        for i in range(X_test.shape[0]):
            min_nll = np.inf
            min_nll_class = 0

            for j in range(self.n_classes):
                nll_j = -np.log(self.dist(X_test[i], self.mu[j], self.var[j]) * self.prior[j])
                if (nll_j) < min_nll:
                    min_nll = nll_j
                    min_nll_class = j
                
            y_pred[i] = min_nll_class
        
        return y_pred