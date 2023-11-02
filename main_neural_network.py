import numpy as np


class NeuralNetwork():

    def __init__(self):
        self.W1 = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0],
                            [-0.3, -0.4, 0.2, 0.2, 0.4]])
        self.W2 = np.array([[1.0],
                            [1.0],
                            [1.0],
                            [1.0],
                            [1.0]])

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, sigmoid):
        return sigmoid * (1 - sigmoid)

    def backpropagation(self, X_train, Y_train, learning_rate, epoch):
        iteration = 1
        error = []
        accuracy = []

        for iteration in range(epoch + 1):
            X = X_train
            sum1 = np.dot(X, self.W1)
            H1 = self.sigmoid(sum1)

            sum2 = np.dot(H1, self.W2)
            Y_pred = self.sigmoid(sum2)

            error = Y_train - Y_pred
            average = np.mean(abs(error))
            result = self.predict(X)
            accuracy = np.mean(Y_train == result)
            acc = accuracy

            if iteration % 10000 == 0:
                print('Epoch: ' + str(iteration) + ' Loss: ' + str(average) + ' Accuracy: ' + str(
                    acc))
                np.append(error, average)
                np.append(accuracy, acc)
            iteration += 1

            derivate_W2 = np.dot(H1.T, error * self.sigmoid_derivative(Y_pred))
            self.W2 += derivate_W2 * learning_rate
            derivate_W1 = np.dot(X.T,
                                 np.dot(error * self.sigmoid_derivative(Y_pred), self.W2.T) * self.sigmoid_derivative(
                                     H1))
            self.W1 += derivate_W1 * learning_rate

    def predict_value(self, X):
        H1 = self.sigmoid(np.dot(X, self.W1))
        Y_pred = self.sigmoid(np.dot(H1, self.W2))
        return Y_pred

    def predict(self, X):
        result = (self.predict_value(X) > 0.5).astype(int)
        return result

    def prediction(self, X, Y):
        error = 0
        correct = 0

        predictions = (self.predict_value(X) > 0.5).astype(int)
        print('\n')
        for i in range(len(X)):
            print(i + 1, X[i], predictions[i], Y[i])
            if (predictions[i].any() != Y[i].any()):
                error = error + 1
            elif (predictions[i].all() == Y[i].all()):
                correct = correct + 1
        print("Number of errors: ", error)
        print("Error percentage: ", (error / len(X)) * 100, "%")
        print("Number of correct: ", correct)
        print("Correct percentage: ", (correct / len(X)) * 100, "%")
