import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from geopy.distance import distance


class ProjectXGBOOST:
    data = None
    data_train = None
    data_test = None
    model = None
    optimizer = None
    loss_function = None
    model_loss = 10000
    early_stop = 0

    def __init__(self, data_path='new_dataset.csv', ft=False):
        self.data = pd.read_csv(data_path)
        self.preproccess_data()
        self.split_data()
        self.declare_model()
        self.ft = ft

    def preproccess_data(self):
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()

    def split_data(self):
        self.data_train = self.data_test = self.data
        self.data_train = self.data_train[:int(0.9 * len(self.data_train))]
        self.data_test = self.data_test[int(0.9 * len(self.data_test)):]

    def declare_model(self):
        self.model_lat = xgboost.XGBRegressor(gamma=0,
                                              learning_rate=0.01,
                                              max_depth=30,
                                              n_estimators=100000,
                                              objective='reg:squarederror',
                                              n_jobs=-1)
        self.model_lon = xgboost.XGBRegressor(gamma=0,
                                              learning_rate=0.01,
                                              max_depth=30,
                                              n_estimators=100000,
                                              objective='reg:squarederror',
                                              n_jobs=-1)

    def train(self):
        batch = self.data_test if not self.ft else self.data
        x_batch = batch[[
            x for x in batch.columns if x not in ('lat', 'lon')]].values
        y_batch = batch[[
            x for x in batch.columns if x in ('lat', 'lon')]].values

        self.model_lat.fit(x_batch, batch.lat)
        self.model_lon.fit(x_batch, batch.lon)

    def test(self):
        x_batch = self.data_test[[
            x for x in self.data_test.columns if x not in ('lat', 'lon')]].values
        y_batch = self.data_test[[
            x for x in self.data_test.columns if x in ('lat', 'lon')]].values

        out_lat = self.model_lat.predict(x_batch)
        out_lon = self.model_lon.predict(x_batch)
        out = list(zip(out_lat, out_lon))
        return out, y_batch

    def test_arx(self, path):
        test = pd.read_csv(path)
        x_batch = test[[
            x for x in test.columns if x not in ('lat', 'lon')]].values
        y_batch = test[[
            x for x in test.columns if x in ('lat', 'lon')]].values

        out_lat = self.model_lat.predict(x_batch)
        out_lon = self.model_lon.predict(x_batch)

        df = pd.DataFrame()
        df['pontoId'] = test.pontoId
        df['lat'] = y_batch[:, 0]
        df['lon'] = y_batch[:, 1]
        df['lat_pred'] = out_lat
        df['lon_pred'] = out_lon
        df.to_csv('Resultados_Equipe3_Metodo_XGBoost.csv', index=False)

    def result_analisys(self, out, y_batch):
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in y_batch], [x[1]
                                          for x in y_batch], 'o', label='Real')
        ax.plot([x[0] for x in out], [x[1] for x in out], 'x', label='XGBoost')
        ax.legend()
        ax.set_title('Comparação')
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        fig.savefig('xgboost_map.png')

        dists = []
        for i in range(len(out)):
            dist = distance(y_batch[i], out[i]).km*1000
            dists.append(dist)

        fig, ax = plt.subplots()
        ax.boxplot(dists)
        ax.set_title('XGBoost BoxPlot Dists')
        fig.savefig('xgboost_boxplot_dists.png')

        fig, ax = plt.subplots()
        ax.hist(dists)
        ax.set_title('XGBoost Histogram Dists')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Distance (M)')
        fig.savefig('xgboost_hist_dists.png')

        print(f'Erro Médio: {np.mean(dists)}\nErro Mínimo: {min(dists)}\n\
Erro Máximo: {max(dists)}\nDesvio Padrão: {np.std(dists)}')


if __name__ == "__main__":
    xgmodel = ProjectXGBOOST(ft=True)
    xgmodel.train()
    with open('xgboost.bin', 'wb') as knnf:
        pickle.dump(xgmodel, knnf)
    # result = xgmodel.test()
    # xgmodel.result_analisys(*result)
