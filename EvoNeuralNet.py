import numpy as np
np.seterr(all='ignore')

class EvoNN:
    def __init__(self, startWeights=None):
        self.weights = startWeights

        # constants (?)
        self.a = 64
        self.b = 32
        self.c = 1

    def set_weights(self, weights):
        self.weights = weights

    def set_params(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def forward(self, X):
        # Add bias to input cases
        cases = np.hstack((np.ones((X.shape[0], 1)), X))
        # Reshape weights for layers
        w1_end = (cases.shape[1]) * self.a
        w2_end = w1_end + (self.a + 1) * self.b
        w1 = self.weights[:w1_end].reshape(cases.shape[1], self.a)
        w2 = self.weights[w1_end:w2_end].reshape(self.a + 1, self.b)
        w3 = self.weights[w2_end:].reshape(self.b + 1, 1)
        # First layer: input to hidden layer 1
        prediction = cases @ w1
        prediction = np.where(prediction > 0, prediction, 0.01 * prediction)  # Leaky ReLU
        # Add bias to the first hidden layer output
        prediction = np.hstack((np.ones((prediction.shape[0], 1)), prediction))
        # Second layer: hidden layer 1 to hidden layer 2
        prediction = prediction @ w2
        prediction = np.where(prediction > 0, prediction, 0.01 * prediction)  # Leaky ReLU
        # Add bias to the second hidden layer output
        prediction = np.hstack((np.ones((prediction.shape[0], 1)), prediction))
        # Output layer: hidden layer 2 to output
        prediction = prediction @ w3
        prediction = 1 / (1 + np.exp(-prediction))  # Sigmoid activation
        # Thresholding
        prediction = (prediction > 0.5).astype(int)    
        return prediction
    

    def accuracy(self, X, y):
        accuracy = (self.forward(X) == np.array(y).reshape(-1,1)).sum() / len(np.array(y))
        return accuracy

    def proportion(self, X1, X2):
        if len(X1) == 0 or len(X2) == 0:
            return None, None
        probx = self.forward(X1).mean()
        proby = self.forward(X2).mean()
        return probx, proby


    def grab_X_split(self, X, c):
        ''' Grabs the splits of X based on the comparisons '''
        if "GENDER" in c:
            male = X[X['gender']==1]
            female = X[X['gender']==0]
            return male.to_numpy(), female.to_numpy()

        elif "AGE" in c:
            old = X[X['age']==1]
            young = X[X['age']==0]
            return old.to_numpy(), young.to_numpy()

        elif "RACE" in c:
            white = X[X['race']==1]
            other = X[X['race']==0]
            return white.to_numpy(), other.to_numpy()

    def proportion2(self, X, comp):
        ''' Returns the proportion of predictions '''
        numerator, denominator = self.grab_X_split(X, comp)

        prob = (self.forward(numerator).mean()) / (self.forward(denominator).mean() + np.finfo(float).eps)

        # if prob > 2:
        #     print(comp)
        #     print(f"Proportion: {prob}")
        #     print(f"Numerator size: {len(numerator)} [avg: {self.forward(numerator).mean()}]")
        #     print(f"Denominator size: {len(denominator)} [avg: {self.forward(denominator).mean()}]")

        return prob

    def run(self, X, y, comps):
        return self.accuracy(X,y), (self.proportion2(X, comps[0]), self.proportion2(X, comps[1]))