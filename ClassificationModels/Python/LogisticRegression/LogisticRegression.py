import numpy as np

def sigmoid(x):
    # return sigmoid function to use in code
    return 1/(1+np.exp(-x))

class LogisticRegression():
    def __init__(self, lr=0.001, nIters=1000):
        self.lr = lr
        self.nIters = nIters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        nSamples, nFeatures = X.shape

        # initialize weights and bias as zero
        self.weights = np.zeros(nFeatures)
        self.bias = 0

        for _ in range(self.nIters):
            
            # define formula
            linearPreds = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linearPreds)

            # calc gradients (we need to get the transpose of X hence X.T)
            dw = (1/nSamples) * np.dot(X.T, (predictions - y))
            db = (1/nSamples) * np.sum(predictions - y)

            # update weights
            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

            # once this is done, my logistic regression algorithm is trained

    def predict(self, X):
        # define formula
            linearPreds = np.dot(X, self.weights) + self.bias
            yPred = sigmoid(linearPreds)

            # sigmoid return values between 0 and 1 so we need to make labels for the values
            classPredictions = [0 if y<=0.5 else 1 for y in yPred]
            return classPredictions