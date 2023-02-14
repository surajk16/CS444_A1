"""Softmax model."""

import numpy as np
import random
import math

class Softmax:
    def __init__(self, n_class: int, lr: float, batch_size: int, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            n_dimension: dimension of input data
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.n_dimension = None
        self.n_sample = None

    def softmax_prob(self, scores):
        max_fc = np.max(scores, axis=1)
        output = scores - max_fc[:, None]
        output /= 100000
        # output /= (1e5 * np.mean(abs(output), axis=1))[:, None]
        # output /= (np.max(abs(output)) * 1e4)

        softmax = np.exp(output)
        prob = softmax / np.sum(softmax, axis=1)[:, None]
        
        return prob

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """

        output = X_train @ self.w.T
        softmax = self.softmax_prob(output)

        # data_loss = 0
        # for i in range(self.batch_size):
        #     data_loss += -math.log(softmax[i][y_train[i]])
        
        #data_loss /= self.batch_size
        #weight_loss = (np.sqrt(np.sum(self.w ** 2)) * self.batch_size * self.reg_const) / (2 * self.n_sample)
        #total_loss = data_loss + weight_loss

        data_loss_grad = np.zeros(self.w.shape)

        for i in range(self.batch_size):
            for j in range(self.n_class):
                if (y_train[i] != j):
                    data_loss_grad[j] += softmax[i][j] * X_train[i]
                else:
                    data_loss_grad[j] += (softmax[i][j]-1) * X_train[i]

        data_loss_grad /= self.batch_size
        weight_loss_grad = (self.reg_const * self.batch_size) / (self.n_sample) * (self.w)

        total_grad = data_loss_grad + weight_loss_grad

        return total_grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        self.n_sample, self.n_dimension = X_train.shape

        self.w = np.random.rand(self.n_class, self.n_dimension+1)

        X_train_copy = np.c_[X_train, np.ones(self.n_sample)]

        eta = self.lr

        for i in range(self.epochs):
            # for i in range(0, self.n_sample, self.batch_size):

            indices = random.sample(range(self.n_sample), self.batch_size)
            X_batch = X_train_copy[indices]
            y_batch = y_train[indices]

            grad = self.calc_gradient(X_batch, y_batch)
            self.w = self.w - eta * grad

            eta = self.decayed_eta(eta, i, 2)

            if (i % 100 == 0):
                print("Epoch {}/{} Accuracy: {}".format(i+1, self.epochs, self.get_acc(self.predict(X_train), y_train)))

        return
    
    def decayed_eta(self, eta, epoch, mode):
        if mode == 2:
            if epoch != 0 and epoch % 5 == 0:
                eta = eta - eta/10
            if eta < 0.05:
                eta = 0.05
        elif mode == 1:
            if epoch !=0 and epoch % 5 == 0:
                eta = eta - eta/5
        return eta

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
        
        X_test = np.c_[X_test, np.ones(X_test.shape[0])]
        pred = X_test @ self.w.T
        y_test = np.argmax(pred, axis=1)

        return y_test
    
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100