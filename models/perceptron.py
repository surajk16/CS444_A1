"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, d: int, n:int, decay_mode: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            d: number of dimensions of input data
            n: number of data points
            decay_mode: learning rate decay mode
        """
        self.w =   np.random.rand(n_class, d)
        self.n = n
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay_mode = decay_mode

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

      
        N = X_train.shape[0]
        D = X_train.shape[1]
        X = np.transpose(X_train)
        print(y_train.shape,N)
        eta = self.lr
        for epoch in range(self.epochs):
            pred = self.w @ X
            Z = np.zeros((self.n_class, N))
            for i in range(N):
                Z[:,i] = pred[y_train[i],i]
            k = pred - Z
            for pt in range(N):
                incorrect_pts = 0
                for clas in range(self.n_class):
                    if clas == y_train[pt]:
                        continue
                    elif k[clas][pt] > 0:
                        self.w[clas,:] = self.w[clas,:] - eta*X[:,pt]
                        incorrect_pts+=1
                self.w[y_train[pt], :] += eta*incorrect_pts*X[:,pt]
            print(f'Epoch {epoch+1} done')
            print(f'Accuracy {self.getaccuracy(self.predict(X_train), y_train)}')
            eta = self.decayed_eta(eta, epoch,self.decay_mode)
        
    def decayed_eta(self, eta, epoch, mode):
        if mode == 2:
            if epoch != 0 and epoch % 5 == 0:
                eta = eta - eta/5
            if eta < 0.05:
                eta = 0.05
        elif mode == 1:
            if epoch !=0 and epoch % 5 == 0:
                eta = eta - eta/5
        return eta
    
    def getaccuracy(self, y_pred, y_test):
        return np.sum(y_test == y_pred) / len(y_test) * 100
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        X = np.transpose(X_test)
        pred = self.w @ X
        y_test = np.argmax(pred, axis=0)
        
        return y_test
