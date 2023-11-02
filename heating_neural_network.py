import pandas as pd
import joblib
from main_neural_network import *

if __name__ == "__main__":
    neural_network_heating = NeuralNetwork()

    print("Starting weights: ")
    print(neural_network_heating.W1)
    print(neural_network_heating.W2)

    raw_dataset_heating = pd.read_csv('trainingHeating.csv', delimiter=',')
    dataset_heating = raw_dataset_heating.copy()
    testing_dataset_heating = pd.read_csv('trainingHeatingTest.csv', delimiter=',')
    testing_heating = testing_dataset_heating.copy()

    dataset_heating = (dataset_heating - dataset_heating.min()) / (dataset_heating.max() - dataset_heating.min())
    print(dataset_heating.min())
    print(dataset_heating.max())

    X_train = np.array([dataset_heating['inside_temperature'], dataset_heating['previous_heating_status']]).T

    Y_train = np.array([dataset_heating['heating']]).T

    neural_network_heating.backpropagation(X_train, Y_train, 0.02, 10000000)
    print(neural_network_heating.predict_value(X_train))
    print(neural_network_heating.prediction(X_train, Y_train))

    print(testing_heating.min())
    print(testing_heating.max())

    testing_heating = (testing_heating - testing_heating.min()) / (testing_heating.max() - testing_heating.min())

    X_test = np.array([testing_heating['inside_temperature'], testing_heating['previous_heating_status']]).T

    Y_test = np.array([testing_heating['heating']]).T

    print(neural_network_heating.prediction(X_test, Y_test))

    print("Weights after training: ")
    print(neural_network_heating.W1)
    print(neural_network_heating.W2)

    max_temperature = 30
    min_temperature = 15

    inside_temperature = float(input("Input 1: "))
    previous_heating_status = int(input("Input 2: "))

    print("New situation: input data = ", inside_temperature, previous_heating_status)
    inside_temperature = (inside_temperature - min_temperature) / (max_temperature - min_temperature)

    print("New situation: input data = ", inside_temperature, previous_heating_status)
    print("Output data: ")
    print(neural_network_heating.predict_value(np.array([inside_temperature, previous_heating_status])))
    print(neural_network_heating.predict(np.array([inside_temperature, previous_heating_status])))

    filename_4 = "model_heating.joblib"
    joblib.dump(neural_network_heating, filename_4)

