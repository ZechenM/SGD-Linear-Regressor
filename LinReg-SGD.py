import time
import random
from random import randrange
import numpy as np
import numpy.linalg as npla
import pandas as pd
import sys


class SGDSolver():

    def __init__(self, path):
        """load input dataset specified in path and split data into train and validation.
        Hint: you can store both training and testing features and target vector as
        class variables"""

        # read the input
        dataset = pd.read_csv(path)
        # numpy array w/ size: n*7
        self.x = dataset.iloc[:, :-1].values
        # make the first column all ones because b = w[0]
        self.x[:, 0] = 1
        print("x is ", self.x)
        # numpy array w/ size n*1
        self.y = dataset.iloc[:, -1].values

        # initialize the mse by setting y_hat = 0
        # self.y = y_GT
        # e_mse = 1/n * sum(y_gt - y_regress)**2
        self.mse = np.sum((self.y - 0)**2) / len(self.y)

    def training(self, alpha, lam, nepoch, epsilon):
        """Training process of linear regression will happen here. User will provide
        learning rate alpha, regularization term lam, specific number of training epoches,
        and a variable epsilon to specify pre-mature end condition,
        ex. if error < epsilon, training stops. Hint: You can store both weight and
        bias as class variables, so other functions can directly use them"""

        # find the best w and b until error < epsilon

        # 2-D grid search - outer: alpha; inner: lam
        # lr: learning rate
        # rw: regularization weight
        lr = alpha[0]
        while lr <= alpha[1]:
            rw = lam[0]
            while rw <= lam[1]:

                # initialize class variables for weight and bias
                # w: w
                # b: w[0]
                # self.x.shape[1] = # of features + 1
                w = np.random.normal(0, 1, self.x.shape[1])
                w[0] = 0

                # for specified number of epochs
                for i in range(nepoch):

                    # initialize y_regression
                    y_hat = []

                    # run through all the samples
                    # in this training dataset, there are 360 of them (self.x.shape[0] = 360)
                    for j in range(self.x.shape[0]):
                        # model prediction
                        y_tmp = self.x[j].dot(w)    # y_tmp is a value
                        y_hat.append(y_tmp)

                        # # loss value
                        # L = 0.5 * (self.y[j] - y_tmp)**2 + \
                        #     0.5 * rw * npla.norm(w, 2)**2

                        # update weight vector
                        for k in range(len(w)):
                            # run through each element by the indexing variable k
                            w[k] = w[k] + lr * self.x[j][k] * \
                                (self.y[j] - self.x[j].dot(w)) - lr * rw * w[k]

                    # at the end of every epoch, calculate the mean squared error
                    # self.y = y_GT; y_hat = y_regression
                    # e_mse = 1/n * sum(y_gt - y_regress)**2
                    # sigmoid(z) = w*x_test
                    # y_test - y
                    # cross_entropy = np.sum(-{y_test*Log(y) + (1-y_test)* log(1-y)})
                    mse_new = np.sum((self.y - y_hat)**2) / len(self.y)

                    # want to minimize the mse
                    # and memorize the best w
                    if mse_new < self.mse:
                        self.w = w
                        self.mse = mse_new

                    # stop if error < epsilon
                    if self.mse < epsilon:
                        return

                # increment regularization weight lambda
                rw *= 10

            # increment learning rate alpha
            lr *= 10

    def testing(self, testX):
        """Use your trained weight and bias to compute the predicted y values,
        return the n*1 y vector"""

        # testX dimensions
        m, n = testX.shape

        # add one more column
        testX_copy = np.ones((m, n+1))
        # make the first column all ones and the rest the same as testX
        testX_copy[:, 1:] = testX

        # compute the predicted y value
        # testX_copy: n*(# of attributes + 1)
        # self.w: (# of attributes + 1)*1
        # y: n*1
        y = testX_copy.dot(self.w).reshape(m, 1)

        # return the y vector
        return y


""" Training Process: You only need to modify nepoch, epsilon of training method,
this is for auto-testing """
model = SGDSolver('train.csv')
# model = SGDSolver(sys.argv[1])
# Compute the time to do grid search on training
start = time.time()

model.training([10**-10, 10], [1, 1e10], 10, 0)


# test = np.array([[310, 108, 5, 3.5, 3.5, 8.56, 0],
#                  [329, 113, 4, 4.0, 4.5, 9.1, 1]])
# print("chance of admit: ", model.testing(
#     test))

end = time.time()
