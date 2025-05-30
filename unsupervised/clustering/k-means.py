import numpy as np

class KMeans:
    def __init__(self, k):
        self.n_classes = k
    

    def fit(self,X_train):
        k_means = np.random.randint(0,X_train.shape[0], self.n_clusters, dtype=int)
        while(len(k_means) != len(set(k_means))):
            k_means = np.random.randint(0,X_train.shape[0], self.n_clusters, dtype=int)

        self.k_means = k_means

        return k_means

    def predict(self,x_test):
        y_pred = np.empty(x_test.shape[0])

        for i in range(x_test.shape[0]):
            y_pred[i] = np.argmin((x_test[i,:]-self.k_means)**2)
        
        return y_pred