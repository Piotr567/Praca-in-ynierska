import pandas as pd
import joblib
from main_neural_network import *

if __name__ == "__main__":
    neural_network_windows = NeuralNetwork()

    print("Starting weights: ")
    print(neural_network_windows.W1)
    print(neural_network_windows.W2)

    raw_dataset_windows = pd.read_csv('trainingWindows.csv', delimiter=',')
    dataset_windows = raw_dataset_windows.copy()
    testing_dataset_windows = pd.read_csv('trainingWindowsTest.csv', delimiter=',')
    testing_windows = testing_dataset_windows.copy()

    dataset_windows = (dataset_windows - dataset_windows.min()) / (dataset_windows.max() - dataset_windows.min())
    print(dataset_windows.min())
    print(dataset_windows.max())

    X_train = np.array([dataset_windows['inside_temperature'], dataset_windows['air_conditioner']]).T

    Y_train = np.array([dataset_windows['windows']]).T

    neural_network_windows.backpropagation(X_train, Y_train, 0.02, 10000000)
    print(neural_network_windows.predict_value(X_train))
    print(neural_network_windows.prediction(X_train, Y_train))

    print(testing_windows.min())
    print(testing_windows.max())

    testing_windows = (testing_windows - testing_windows.min()) / (testing_windows.max() - testing_windows.min())

    X_test = np.array([testing_windows['inside_temperature'], testing_windows['air_conditioner']]).T

    Y_test = np.array([testing_windows['windows']]).T

    print(neural_network_windows.prediction(X_test, Y_test))

    print("Weights after training: ")
    print(neural_network_windows.W1)
    print(neural_network_windows.W2)

    max_temperature = 30
    min_temperature = 15

    inside_temperature = float(input("Input 1: "))
    air_conditioner = int(input("Input 2: "))

    print("New situation: input data = ", inside_temperature, air_conditioner)
    inside_temperature = (inside_temperature - min_temperature) / (max_temperature - min_temperature)

    print("New situation: input data = ", inside_temperature, air_conditioner)
    print("Output data: ")
    print(neural_network_windows.predict_value(np.array([inside_temperature, air_conditioner])))
    print(neural_network_windows.predict(np.array([inside_temperature, air_conditioner])))

    filename_3 = "model_windows.joblib"
    joblib.dump(neural_network_windows, filename_3)
