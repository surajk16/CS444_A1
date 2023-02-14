"""Logistic regression model."""

import numpy as np
import random
import math

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float, batch_size: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.batch_size = batch_size
        self.reg_const = reg_const
        self.n_class = 2

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        
        return 1 / (1 + np.exp(-z))
    
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

        # output = X_train @ self.w.T
        # # print(y_train.shape, output.shape)
        # # print((np.array(output).squeeze() * y_train).shape)

        # data_loss = np.sum(-np.log(self.sigmoid(np.array(output).squeeze() * y_train)))
        # data_loss /= self.batch_size
        # weight_loss = (np.sqrt(np.sum(self.w ** 2)) * self.batch_size * self.reg_const) / (2 * self.n_sample)
        # total_loss = data_loss + weight_loss

        # data_loss_grad = -np.sum(self.sigmoid(-np.array(output).squeeze() * y_train)) * (y_train.T @ X_train)
        # data_loss_grad /= self.batch_size
        # print(data_loss_grad)
        # weight_loss_grad = (self.reg_const * self.batch_size) / (self.n_sample) * (self.w)

        # total_grad = data_loss_grad + weight_loss_grad

        # #return total_loss, total_grad
        # return data_loss, data_loss_grad

        output = X_train @ self.w.T
        y_train = y_train.reshape(-1, 1)

        #data_loss = -np.mean(np.log(self.sigmoid(y_train * output)))
        weight_loss = (np.sqrt(np.sum(self.w ** 2)) * self.reg_const) / (2 * self.batch_size)
        #total_loss = data_loss + weight_loss

        data_loss_grad = 0
        for i in range(X_train.shape[0]):
            data_loss_grad += -self.sigmoid(-y_train[i] * output[i]) * (y_train[i] * X_train[i])

        data_loss_grad /= self.batch_size
          
        # data_loss_grad = np.sum(-self.sigmoid(-y_train * output)) * (y_train.T @ X_train)
        # data_loss_grad /= self.batch_size

        weight_loss_grad = (self.reg_const) / (self.batch_size) * (self.w)
        total_grad = data_loss_grad + weight_loss_grad

        return 0, total_grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        
        self.n_sample, self.n_dimension = X_train.shape

        self.w = np.random.rand(1, self.n_dimension+1)
        #self.w = np.zeros((1, self.n_dimension+1))

        X_train_copy = np.c_[X_train, np.ones(self.n_sample)]
        y_train_copy = y_train.copy()
        y_train_copy[y_train==0] = -1

        eta = self.lr

        # for i in range(self.epochs):

        #     indices = random.sample(range(self.n_sample), self.batch_size)
        #     X_batch = X_train_copy[indices]
        #     y_batch = y_train_copy[indices]

        #     loss, grad = self.calc_gradient(X_batch, y_batch)
        #     self.w = self.w - eta * grad

        #     eta = self.decayed_eta(eta, i, 2)

        #     if (i % 1 == 0):
        #         print("Epoch {}/{} Accuracy: {}".format(i+1, self.epochs, self.get_acc(self.predict(X_train), y_train)))

        for j in range(self.epochs):
        
            for i in range(0, self.n_sample, self.batch_size):
                st = i
                en = st + self.batch_size
                loss, grad = self.calc_gradient(X_train_copy[st:en], y_train_copy[st:en])
                self.w = self.w - eta * grad

            eta = self.decayed_eta(eta, i, 2)

            if (i % 1 == 0):
                print("Epoch {}/{} Accuracy: {}".format(j+1, self.epochs, self.get_acc(self.predict(X_train), y_train)))

        return
    

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
        pred = self.sigmoid(X_test @ self.w.T)

        pred[pred > self.threshold] = 1
        pred[pred < self.threshold] = 0

        return np.array(pred).squeeze()
    
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
