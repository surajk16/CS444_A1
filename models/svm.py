"""Support Vector Machine (SVM) model."""

import numpy as np

class SVM:
    def __init__(self, n_class: int, d):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            d: number of dimensions of the data
        """
        
        self.w = np.random.rand(n_class, d)  
        self.n_class = n_class
        
    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        
        grad = np.zeros_like(self.w)
        N = X_train.shape[0]
        X = np.transpose(X_train)
        
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
                    elif k[clas][pt] > -1:
                        grad[clas,:] += X[:,pt]
                        incorrect_pts+=1
                grad[y_train[pt], :] -= incorrect_pts*X[:,pt]  
        return grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray, lr, epochs, reg_const, decay_mode, mini_batch, batch_size):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
            lr: learning rate
            epochs: number of epochs
            reg_const: regularisation constant
            decay_mode: learning rate decay mode
            mini_batch: False means strict stochastic gradient descent True means mini batch SGD
            batch_size: size of mini batch
        """
        print("epochs",epochs,"reg_val",reg_const,"learning rate",lr,"mini_batch fraction", batch_size,"mini batch", mini_batch)
        
        N = X_train.shape[0]
        D = X_train.shape[1]

        eta = lr
        indx = np.array(range(0,N))
        shuffled = random.shuffle(indx)
        for epoch in range(epochs):

            # Regularization update(once per epoch)
            self.w = (1 - eta*reg_const/N)*self.w
            
          
            
            if not mini_batch: 
                for data_pt in shuffled:
                    grad = self.calc_gradient(X_train[data_pt], Y_train[data_pt]) 
                    self.w = self.w - eta*grad 


            else:
                for i in range(0,N,batch_size):
                    st = i
                    en = st + batch_size
                    grad = self.calc_gradient(X_train[st:en], y_train[st:en])
                    self.w = self.w - eta*grad
                    
            print(f'Epoch {epoch+1} done')
            print(f'Accuracy {self.getaccuracy(self.predict(X_train), y_train)}')
            eta = self.decayed_eta(eta, epoch,decay_mode)    
        return

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