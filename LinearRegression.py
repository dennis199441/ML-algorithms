class LinearRegression:

    def __init__(self, learning_rate, n_epoch):
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch

    def predict(self, features):
        y_hat = self.coef[0]
        for i in range(self.n_features):
            y_hat += self.coef[i+1] * features[i]

        return y_hat

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_features = len(X[0])

        self.coef = [0.0 for i in range(self.n_features + 1)]

        for epoch in range(self.n_epoch):
            print("Epoh: %s" % epoch)
            print("Coefficient: %s" % self.coef)
            for i in range(len(X)):
                y_hat = self.predict(X[i])
                error = y_hat - y[i][0]
                self.coef[0] = self.coef[0] - self.learning_rate * error
                for j in range(self.n_features):
                    self.coef[j+1] = self.coef[j+1] - self.learning_rate * error * X[i][j]