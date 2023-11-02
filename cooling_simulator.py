import time

import matplotlib.pyplot as plt

from heating_neural_network import *

filename_1 = "model_weather.joblib"
model_weather = joblib.load(filename_1)
filename_2 = "model_air_conditioner.joblib"
model_air_conditioner = joblib.load(filename_2)
filename_3 = "model_windows.joblib"
model_windows = joblib.load(filename_3)


class CoolingSimulator():
    def __init__(self):
        self.duration_time = 60
        self.precipitation = [1, 1, 0, 0, 0, 0, 0, 0, 0]
        self.no_precipitation = [0, 0, 1, 1, 1, 1, 1, 1, 1]
        self.wind_power = [30, 45, 40, 35, 30, 42, 33, 25, 20]
        self.outside_temperature = [18, 20, 21, 22, 23, 24, 24, 24, 25]
        self.greenhouse_area = 62.41
        self.greenhouse_area_with_open_windows = 59.09
        self.open_windows_area = 3.32
        self.heat_transfer_coefficient = 2.94
        self.heat_transfer_coefficient_of_a_greenhouse_with_open_windows_and_doors = 3.13
        self.heat_transfer_coefficient_of__open_windows_and_doors = 12.82
        self.greenhouse_area_in_ft = self.greenhouse_area * 10.76391042
        self.greenhouse_area_with_open_windows_in_ft = self.greenhouse_area_with_open_windows * 10.76391042
        self.open_windows_area_in_ft = self.open_windows_area * 10.76391042
        self.intensity_of_solar_radiation = [220, 250, 280, 320, 350, 370, 320, 250, 220]
        self.earth_radiation = [300, 300, 310, 320, 320, 320, 310, 310, 300]
        self.conversion_to_watts = 0.29307107
        self.air_conditioner_power = 2500
        self.greenhouse_heat_capacity = 54270
        self.thermal_resistance = 0.1
        self.greenhouse_resistance = 0.34
        self.temperature = [20.0]
        self.air_condition = [0]
        self.start = time.time()
        self.timer = 0
        self.list = []
        self.air_conditioner = []
        self.windows = [0]
        self.compressor_temperature = 30
        self.compressor_heat_capacity = 2760

        self.condenser_temperature = 30
        self.condenser_heat_capacity = 1760

        self.capillaries_temperature = 30
        self.capillaries_heat_capacity = 380

        self.evaporator_temperature = 30
        self.evaporator_heat_capacity = 1760

        self.air_conditioner_system_heat_capacity = 6780
        super().__init__()

    def cooling_simulator(self):
        for i in range(len(self.outside_temperature)):
            for self.timer in range(self.duration_time):

                self.timer = int(time.time() - self.start)
                time.sleep(1)
                self.list.append(self.timer)
                print("Minuta:", self.timer)

                if i == 0 and self.timer == 0:
                    self.previous_air_conditioner_status = int(self.air_condition[-1])
                    inside_temperature_in_greenhouse = self.temperature[-1]
                    self.inside_temperature = inside_temperature_in_greenhouse

                max_wind_power = 100
                min_wind_power = 0

                print("Dane wejściowe pogody = ", self.precipitation[i], self.no_precipitation[i], self.wind_power[i])
                wind_power = self.wind_power[i]
                wind_power = (wind_power - min_wind_power) / (max_wind_power - min_wind_power)
                print("Dane wyjściowe pogody = ", model_weather.predict(np.array([self.precipitation[i], self.no_precipitation[i], wind_power])))
                output = model_weather.predict(np.array([self.precipitation[i], self.no_precipitation[i], wind_power]))

                max_temperature = 30
                min_temperature = 15

                self.previous_air_conditioner_status = int(self.air_condition[-1])

                print("Dane wejściowe klimatyzacji = ", self.inside_temperature, self.previous_air_conditioner_status)

                self.inside_temperature = (self.inside_temperature - min_temperature) / (max_temperature - min_temperature)

                print("Dane wejściowe klimatyzacji zmodyfikowane = ", self.inside_temperature, self.previous_air_conditioner_status)

                print("Dane wyjściowe klimatyzacji = ", model_air_conditioner.predict_value(
                    np.array([self.inside_temperature, self.previous_air_conditioner_status])))
                air_conditioner = model_air_conditioner.predict(
                    np.array([self.inside_temperature, self.previous_air_conditioner_status]))

                self.air_condition.append(int(air_conditioner))

                print("Poprzedni status klimatyzacji = ", int(self.previous_air_conditioner_status), "Aktualny status_klimatyzacji = ", int(air_conditioner))

                print("Dane wyjściowe klimatyzacji = ", int(air_conditioner))

                air_conditioner = int(air_conditioner)
                print("Dane wejściowe okien zmodyfikowane = ", self.inside_temperature, air_conditioner)

                print("Dane wyjściowe okien = ", model_windows.predict_value(np.array([self.inside_temperature, air_conditioner])))
                windows = model_windows.predict(np.array([self.inside_temperature, air_conditioner]))
                if (windows == 1 and output == 0) or (windows == 0 and output == 0) or (windows == 0 and output == 1):
                    windows = 0
                self.windows.append(int(windows))

                print("Status okien = ", int(windows))

                if air_conditioner == 0 and windows == 0:
                    heating_power = (self.intensity_of_solar_radiation[i] + self.earth_radiation[i]) * self.greenhouse_area
                    inside_temperature_greenhouse = (heating_power / (self.greenhouse_area_in_ft * self.heat_transfer_coefficient * self.conversion_to_watts) - 32) * 5 / 9 + self.outside_temperature[i]
                    inside_temperature_in_greenhouse = inside_temperature_in_greenhouse + (inside_temperature_greenhouse - inside_temperature_in_greenhouse) / 60

                if windows == 1 and air_conditioner == 0:
                    heating_power = ((self.intensity_of_solar_radiation[i] + self.earth_radiation[i]) * self.greenhouse_area_with_open_windows) - ((self. intensity_of_solar_radiation[i] + self.earth_radiation[i]) * self.open_windows_area)
                    inside_temperature_greenhouse = (heating_power / ((self.greenhouse_area_in_ft * self.heat_transfer_coefficient_of_a_greenhouse_with_open_windows_and_doors +
                                                                       self.open_windows_area_in_ft * self.heat_transfer_coefficient_of__open_windows_and_doors) * self.conversion_to_watts) - 32) * 5 / 9 + self.outside_temperature[i]
                    inside_temperature_in_greenhouse = inside_temperature_in_greenhouse + (inside_temperature_greenhouse - inside_temperature_in_greenhouse) / 60


                    print("Okna otwarte:", windows)

                if air_conditioner == 1 and windows == 0:
                    self.compressor_temperature += self.air_conditioner_power / self.compressor_heat_capacity
                    air_conditioning_compressor_temperature_change = (self.compressor_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.compressor_temperature -= (air_conditioning_compressor_temperature_change / self.compressor_heat_capacity)
                    inside_temperature_in_greenhouse += (air_conditioning_compressor_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.condenser_temperature += self.air_conditioner_power / self.condenser_heat_capacity
                    air_conditioning_condenser_temperature_change = (self.condenser_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.condenser_temperature += (air_conditioning_condenser_temperature_change / self.condenser_heat_capacity)
                    inside_temperature_in_greenhouse -= (air_conditioning_condenser_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.capillaries_temperature += self.air_conditioner_power / self.capillaries_heat_capacity
                    air_conditioning_capillaries_temperature_change = (self.capillaries_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.capillaries_temperature += (air_conditioning_capillaries_temperature_change / self.capillaries_heat_capacity)
                    inside_temperature_in_greenhouse -= (air_conditioning_capillaries_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.evaporator_temperature += self.air_conditioner_power / self.evaporator_heat_capacity
                    air_conditioning_evaporator_temperature_change = (self.evaporator_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.evaporator_temperature += (air_conditioning_evaporator_temperature_change / self.evaporator_heat_capacity)
                    inside_temperature_in_greenhouse -= (air_conditioning_evaporator_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.compressor_temperature += self.air_conditioner_power / self.compressor_heat_capacity
                    air_conditioning_compressor_temperature_change = (self.compressor_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.compressor_temperature -= (air_conditioning_compressor_temperature_change / self.compressor_heat_capacity)
                    inside_temperature_in_greenhouse += (air_conditioning_compressor_temperature_change / self.air_conditioner_system_heat_capacity)

                    greenhouse_temperature_change = (inside_temperature_in_greenhouse - self.outside_temperature[i]) / self.greenhouse_resistance
                    inside_temperature_in_greenhouse -= (greenhouse_temperature_change / self.greenhouse_heat_capacity)


                if air_conditioner != 1:
                    self.compressor_temperature = 30
                    self.condenser_temperature = 30
                    self.capillaries_temperature = 30
                    self.evaporator_temperature = 30


                self.inside_temperature = inside_temperature_in_greenhouse

                self.temperature.append(round(inside_temperature_in_greenhouse, 1))

            print('Temperatura: ', self.temperature)
            print('Czas: ', self.list)
            print('Okna: ', self.windows)
            print('Klimatyzacja: ', self.air_conditioner)

        self.temperature.pop()
        plt.subplots(figsize=(18, 6))
        plt.title('Temperatura wewnętrzna w czasie')
        plt.grid(True)
        plt.xlabel("Czas")
        plt.ylabel('Temperatura')
        plt.plot(self.list, self.temperature)
        plt.legend(['Temperatura wewnętrzna'], loc='lower right')
        plt.show()

        self.air_condition.pop()
        plt.subplots(figsize=(18, 6))
        plt.title('Status klimatyzacji w czasie')
        plt.grid(True)
        plt.xlabel("Czas")
        plt.ylabel('Status klimatyzacji')
        plt.plot(self.list, self.air_condition)
        plt.legend(['Temperatura wewnętrzna'], loc='lower right')
        plt.show()

        self.windows.pop()
        plt.subplots(figsize=(18, 6))
        plt.title('Status okien w czasie')
        plt.grid(True)
        plt.xlabel("Czas")
        plt.ylabel('Status okien')
        plt.plot(self.list, self.windows)
        plt.legend(['Temperatura wewnętrzna'], loc='lower right')
        plt.show()

