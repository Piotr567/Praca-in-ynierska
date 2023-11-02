import time

import matplotlib.pyplot as plt

from heating_neural_network import *

filename_3 = "model_heating.joblib"
model_heating = joblib.load(filename_3)


class HeatingSimulator():
    def __init__(self):
        self.duration_time = 60
        self.outside_temperature = [8, 9, 10, 10, 11, 11, 12, 12, 12]
        self.greenhouse_area = 62.41
        self.greenhouse_area_in_ft = self.greenhouse_area * 10.76391042
        self.heat_transfer_coefficient = 2.94
        self.intensity_of_solar_radiation = [20, 30, 40, 50, 55, 60, 50, 40, 20]
        self.earth_radiation = [300, 300, 310, 320, 320, 320, 310, 310, 300]
        self.conversion_to_watts = 0.29307107
        self.air_conditioner_power = 3300
        self.greenhouse_heat_capacity = 54270
        self.thermal_resistance = 0.1
        self.greenhouse_resistance = 0.34
        self.temperature = [20.0]
        self.air_condition = [0]
        self.start = time.time()
        self.timer = 0
        self.list = []
        self.air_conditioner = []
        self.heating = [0]
        self.compressor_temperature = 20
        self.compressor_heat_capacity = 2760

        self.condenser_temperature = 20
        self.condenser_heat_capacity = 1760

        self.capillaries_temperature = 20
        self.capillaries_heat_capacity = 380

        self.evaporator_temperature = 20
        self.evaporator_heat_capacity = 1760

        self.air_conditioner_system_heat_capacity = 6780
        super().__init__()

    def heating_simulator(self):
        for i in range(len(self.outside_temperature)):
            for self.timer in range(self.duration_time):

                self.timer = int(time.time() - self.start)
                time.sleep(1)
                self.list.append(self.timer)
                print("Minuta:", self.timer)

                if i == 0 and self.timer == 0:
                    self.previous_heating_status = int(self.heating[-1])
                    inside_temperature_in_greenhouse = self.temperature[-1]
                    self.inside_temperature = inside_temperature_in_greenhouse

                max_temperature = 30
                min_temperature = 15

                self.previous_heating_status = int(self.heating[-1])

                print("Dane wejściowe klimatyzacji = ", self.inside_temperature, self.previous_heating_status)

                self.inside_temperature = (self.inside_temperature - min_temperature) / (max_temperature - min_temperature)

                print("Dane wejściowe klimatyzacji zmodyfikowane = ", self.inside_temperature, self.previous_heating_status)

                print("Dane wyjściowe klimatyzacji = ",
                      model_heating.predict_value(np.array([self.inside_temperature, self.previous_heating_status])))
                heating = model_heating.predict(np.array([self.inside_temperature, self.previous_heating_status]))

                self.heating.append(int(heating))

                print("Poprzedni status klimatyzacji = ", int(self.previous_heating_status),
                      "Aktualny status_klimatyzacji = ", int(heating))

                print("Dane wyjściowe klimatyzacji = ", int(heating))

                if heating == 0:
                    heating_power = (self.intensity_of_solar_radiation[i] + self.earth_radiation[i]) * self.greenhouse_area
                    inside_temperature_greenhouse = (heating_power / (self.greenhouse_area_in_ft * self.heat_transfer_coefficient * self.conversion_to_watts) - 32) * 5 / 9 + self.outside_temperature[i]
                    inside_temperature_in_greenhouse = inside_temperature_in_greenhouse + (inside_temperature_greenhouse - inside_temperature_in_greenhouse) / 60
                    print(inside_temperature_in_greenhouse, inside_temperature_greenhouse)


                if heating == 1:
                    self.compressor_temperature += self.air_conditioner_power / self.compressor_heat_capacity
                    air_conditioning_compressor_temperature_change = (self.compressor_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.compressor_temperature += (air_conditioning_compressor_temperature_change / self.compressor_heat_capacity)
                    inside_temperature_in_greenhouse -= (air_conditioning_compressor_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.condenser_temperature += self.air_conditioner_power / self.condenser_heat_capacity
                    air_conditioning_condenser_temperature_change = (self.condenser_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.condenser_temperature -= (air_conditioning_condenser_temperature_change / self.condenser_heat_capacity)
                    inside_temperature_in_greenhouse += (air_conditioning_condenser_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.capillaries_temperature += self.air_conditioner_power / self.capillaries_heat_capacity
                    air_conditioning_capillaries_temperature_change = (self.capillaries_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.capillaries_temperature -= (air_conditioning_capillaries_temperature_change / self.capillaries_heat_capacity)
                    inside_temperature_in_greenhouse += (air_conditioning_capillaries_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.evaporator_temperature += self.air_conditioner_power / self.evaporator_heat_capacity
                    air_conditioning_evaporator_temperature_change = (self.evaporator_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.evaporator_temperature -= (air_conditioning_evaporator_temperature_change / self.evaporator_heat_capacity)
                    inside_temperature_in_greenhouse += (air_conditioning_evaporator_temperature_change / self.air_conditioner_system_heat_capacity)

                    self.compressor_temperature += self.air_conditioner_power / self.compressor_heat_capacity
                    air_conditioning_compressor_temperature_change = (self.compressor_temperature - inside_temperature_in_greenhouse) / self.thermal_resistance
                    self.compressor_temperature += (air_conditioning_compressor_temperature_change / self.compressor_heat_capacity)
                    inside_temperature_in_greenhouse -= (air_conditioning_compressor_temperature_change / self.air_conditioner_system_heat_capacity)

                    greenhouse_temperature_change = (inside_temperature_in_greenhouse - self.outside_temperature[i]) / self.greenhouse_resistance
                    inside_temperature_in_greenhouse += (greenhouse_temperature_change / self.greenhouse_heat_capacity)

                if heating != 1:
                    self.compressor_temperature = 20
                    self.condenser_temperature = 20
                    self.capillaries_temperature = 20
                    self.evaporator_temperature = 20

                self.inside_temperature = inside_temperature_in_greenhouse
                self.temperature.append(round(inside_temperature_in_greenhouse, 1))

            print(self.temperature)
            print(self.list)
            print(self.heating)

        self.temperature.pop()
        plt.subplots(figsize=(18, 6))
        plt.title('Temperatura wewnętrzna w czasie')
        plt.grid(True)
        plt.xlabel("Czas")
        plt.ylabel('Temperatura')
        plt.plot(self.list, self.temperature)
        plt.legend(['Temperatura wewnętrzna'], loc='lower right')
        plt.show()

        self.heating.pop()
        plt.subplots(figsize=(18, 6))
        plt.title('Status ogrzewania w czasie')
        plt.grid(True)
        plt.xlabel("Czas")
        plt.ylabel('Status ogrzewania')
        plt.plot(self.list, self.heating)
        plt.legend(['Temperatura wewnętrzna'], loc='lower right')
        plt.show()
