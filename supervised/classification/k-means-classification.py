import numpy as np

class KMeansClassifier:
    def __init__(self):
        self.n_classes = 2
    

    def fit(self,X_train, y_train):
        self.n_classes = len(set(y_train))
        k_means = np.zeros((self.n_classes,X_train.shape[1:]))

        for i in range(X_train.shape[0]):
            k_means[y_train[i]] += X_train[i,:]
        
        k_means /= X_train.shape[0]

        self.k_means = k_means

    def predict(self,x_test):
        y_pred = np.empty(x_test.shape[0])

        for i in range(x_test.shape[0]):
            y_pred[i] = np.argmin((x_test[i]-self.k_means)**2)
        
        return y_pred