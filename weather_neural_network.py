import pandas as pd
import joblib
from main_neural_network_weather import *


if __name__ == "__main__":
    neural_network_weather = NeuralNetworkWeather()

    print("Starting weights: ")
    print(neural_network_weather.W1)
    print(neural_network_weather.W2)

    raw_dataset_weather = pd.read_csv('trainingWeather.csv', delimiter=',')
    dataset_weather = raw_dataset_weather.copy()
    testing_dataset_weather = pd.read_csv('testWeather.csv', delimiter=',')
    testing_weather = testing_dataset_weather.copy()

    print(dataset_weather.min())
    print(dataset_weather.max())

    dataset_weather = (dataset_weather - dataset_weather.min()) / (dataset_weather.max() - dataset_weather.min())
    print(dataset_weather.min())
    print(dataset_weather.max())

    X_train = np.array(
        [dataset_weather['precipitation'], dataset_weather['no_precipitation'], dataset_weather['wind_power']]).T

    Y_train = np.array([dataset_weather['output']]).T

    neural_network_weather.backpropagation(X_train, Y_train, 0.02, 10000000)
    print(neural_network_weather.predict_value(X_train))
    print(neural_network_weather.prediction(X_train, Y_train))

    print(testing_weather.min())
    print(testing_weather.max())

    testing_weather = (testing_weather - testing_weather.min()) / (testing_weather.max() - testing_weather.min())

    X_test = np.array(
        [testing_weather['precipitation'], testing_weather['no_precipitation'], testing_weather['wind_power']]).T

    Y_test = np.array([testing_weather['output']]).T

    print(neural_network_weather.prediction(X_test, Y_test))

    print("Weights after training: ")
    print(neural_network_weather.W1)
    print(neural_network_weather.W2)

    max_wind_power = 100
    min_wind_power = 0

    precipitation = int(input("Input 1: "))
    no_precipitation = int(input("Input 2: "))
    wind_power = int(input("Input 3: "))

    print("New situation: input data = ", precipitation, no_precipitation, wind_power)
    wind_power = (wind_power - min_wind_power) / (max_wind_power - min_wind_power)

    print("New situation: input data = ", precipitation, no_precipitation, wind_power)
    print("Output data: ")
    print(
        neural_network_weather.predict_value(np.array([precipitation, no_precipitation, wind_power])))
    print(neural_network_weather.predict(np.array([precipitation, no_precipitation, wind_power])))

    filename_1 = "model_weather.joblib"
    joblib.dump(neural_network_weather, filename_1)
