import numpy as np


class Adaline(object):

    def __init__(self, eta=0.0025, precision=0.000001, epoch=10000):
        self.eta = eta # learning rate
        self.precision = precision # exit condition
        self.epoch = epoch # max iters

    def fit(self, X, y):
        self.weight_ = np.random.uniform(0, 1, X.shape[1] + 1) # weights vector
        print('Pesos iniciais: ', self.weight_)
        self.error_ = [] # errors vector

        for e in range(self.epoch):

            output = self.activation_function(X)
            errors = y - output

            self.weight_[0] += self.eta * sum(errors)
            self.weight_[1:] += self.eta * X.T.dot(errors)

            # EQM
            cost = (errors ** 2).sum() /  X.shape[0]

            if (len(self.error_) > 0):
                if (abs(cost - self.error_[-1]) <= self.precision):
                    print("Padrão aprendido em " + str(e) + " epocas")
                    print('EQM: ', str(cost))
                    print('Pesos finais: ', self.weight_)
                    break

            self.error_.append(cost)

        return self

    def net_input(self, X):
        return np.dot(X, self.weight_[1:]) + self.weight_[0]

    def activation_function(self, X):
        # g(z) function
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation_function(X) >= 0.0, 1, -1)