import pandas as pd
import joblib
from main_neural_network import *


if __name__ == "__main__":
    neural_network_air_conditioner = NeuralNetwork()

    print("Starting weights: ")
    print(neural_network_air_conditioner.W1)
    print(neural_network_air_conditioner.W2)

    raw_dataset_air_conditioner = pd.read_csv('trainingAirConditioner.csv', delimiter=',')
    dataset_air_conditioner = raw_dataset_air_conditioner.copy()
    testing_dataset_air_conditioner = pd.read_csv('trainingAirConditionerTest.csv', delimiter=',')
    testing_air_conditioner = testing_dataset_air_conditioner.copy()

    dataset_air_conditioner = (dataset_air_conditioner - dataset_air_conditioner.min()) / (
            dataset_air_conditioner.max() - dataset_air_conditioner.min())
    print(dataset_air_conditioner.min())
    print(dataset_air_conditioner.max())

    X_train = np.array(
        [dataset_air_conditioner['inside_temperature'], dataset_air_conditioner['previous_air_conditioner_status']]).T

    Y_train = np.array([dataset_air_conditioner['air_conditioner']]).T

    neural_network_air_conditioner.backpropagation(X_train, Y_train, 0.02, 10000000)
    print(neural_network_air_conditioner.predict_value(X_train))
    print(neural_network_air_conditioner.prediction(X_train, Y_train))

    print(testing_air_conditioner.min())
    print(testing_air_conditioner.max())

    testing_air_conditioner = (testing_air_conditioner - testing_air_conditioner.min()) / (
            testing_air_conditioner.max() - testing_air_conditioner.min())

    X_test = np.array(
        [testing_air_conditioner['inside_temperature'], testing_air_conditioner['previous_air_conditioner_status']]).T

    Y_test = np.array([testing_air_conditioner['air_conditioner']]).T

    print(neural_network_air_conditioner.prediction(X_test, Y_test))

    print("Weights after training: ")
    print(neural_network_air_conditioner.W1)
    print(neural_network_air_conditioner.W2)

    max_temperature = 30
    min_temperature = 15

    inside_temperature = float(input("Input 1: "))
    previous_air_conditioner_status = int(input("Input 2: "))

    print("New situation: input data = ", inside_temperature, previous_air_conditioner_status)
    inside_temperature = (inside_temperature - min_temperature) / (max_temperature - min_temperature)

    print("New situation: input data = ", inside_temperature, previous_air_conditioner_status)
    print("Output data: ")
    print(
        neural_network_air_conditioner.predict_value(np.array([inside_temperature, previous_air_conditioner_status])))
    print(neural_network_air_conditioner.predict(np.array([inside_temperature, previous_air_conditioner_status])))

    filename_2 = "model_air_conditioner.joblib"
    joblib.dump(neural_network_air_conditioner, filename_2)
