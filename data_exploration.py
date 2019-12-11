import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import distance
import xgboost


class ProjectDataExploration:
    data = None
    data_train = None
    data_test = None

    def __init__(self, data_path='LocTreino_Equipe_3.csv', log_dir='log_mlp/'):
        self.data = pd.read_csv(data_path)
        self.preproccess_data()
        self.analisys()
        self.drop_outliers()

    def preproccess_data(self):
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()

    def analisys(self):
        fig, ax = plt.subplots()
        ax.boxplot(self.data.lat)
        ax.set_title('BoxPlot Latitude')
        fig.savefig('boxplot_lat.png')

        fig, ax = plt.subplots()
        ax.boxplot(self.data.lon)
        ax.set_title('BoxPlot Longitude')
        fig.savefig('boxplot_lon.png')

        print(f'Latitude Média: {np.mean(self.data.lat)}\nLatitude Mínima: {min(self.data.lat)}\n\
Latitude Máxima: {max(self.data.lat)}\nDesvio Padrão Lat: {np.std(self.data.lat)}')
        print()
        print(f'Longitude Média: {np.mean(self.data.lon)}\nLongitude Mínima: {min(self.data.lon)}\n\
Longitude Máxima: {max(self.data.lon)}\nDesvio Padrão Lon: {np.std(self.data.lon)}')

    def drop_outliers(self):
        self.data = self.data[self.data['lat'] <= np.std(self.data.lat)]
        self.data = self.data[self.data['lon'] <= np.std(self.data.lon)]
        self.data = self.data[self.data['rssi_1_1']
                              <= np.std(self.data.rssi_1_1)]
        self.data = self.data[self.data['rssi_1_2']
                              <= np.std(self.data.rssi_1_2)]
        self.data = self.data[self.data['rssi_1_3']
                              <= np.std(self.data.rssi_1_3)]
        self.data = self.data[self.data['rssi_2_1']
                              <= np.std(self.data.rssi_2_1)]
        self.data = self.data[self.data['rssi_2_2']
                              <= np.std(self.data.rssi_2_2)]
        self.data = self.data[self.data['rssi_2_3']
                              <= np.std(self.data.rssi_2_3)]
        self.data = self.data[self.data['rssi_3_1']
                              <= np.std(self.data.rssi_3_1)]
        self.data = self.data[self.data['rssi_3_2']
                              <= np.std(self.data.rssi_3_2)]
        self.data = self.data[self.data['rssi_3_3']
                              <= np.std(self.data.rssi_3_3)]

        self.data.to_csv('new_dataset.csv', index=False)


if __name__ == "__main__":
    ProjectDataExploration()
