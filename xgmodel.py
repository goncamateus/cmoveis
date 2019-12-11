import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import distance
import xgboost 

class ProjectXGBOOST:
    data = None
    data_train = None
    data_test = None
    model = None
    optimizer = None
    loss_function = None
    model_loss = 10000
    early_stop = 0

    def __init__(self, data_path='LocTreino_Equipe_3.csv'):
        self.data = pd.read_csv(data_path)
        self.preproccess_data()
        self.split_data()
        self.declare_model()

    def preproccess_data(self):
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()

    def split_data(self):
        self.X = self.data[[
            x for x in self.data.columns if x not in ('lat', 'lon')]].values
        self.y = self.data[[
            x for x in self.data.columns if x in ('lat', 'lon')]].values

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
        batch = self.data_train
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
    

    def result_analisys(self, out, y_batch):
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in y_batch], [x[1]
                                          for x in y_batch], 'o', label='Real')
        ax.plot([x[0] for x in out], [x[1] for x in out], 'x', label='XGBoost')
        ax.legend()
        ax.set_title('Comparação')
        fig.savefig('xgboost_map.png')

        dists = []
        for i in range(len(out)):
            dist = distance(y_batch[i], out[i]).km*1000
            dists.append(dist)
        
        fig, ax = plt.subplots()
        ax.boxplot(dists)
        ax.set_title('BoxPlot Dists')
        fig.savefig('xgboost_boxplot_dists.png')

        fig, ax = plt.subplots()
        ax.hist(dists)
        ax.set_title('Histogram Dists')
        fig.savefig('xgboost_hist_dists.png')


        print(f'Erro Médio: {np.mean(dists)}\nErro Mínimo: {min(dists)}\n\
Erro Máximo: {max(dists)}\nDesvio Padrão: {np.std(dists)}')


if __name__ == "__main__":
    xgmodel = ProjectXGBOOST()
    xgmodel.train()
    result = xgmodel.test()
    xgmodel.result_analisys(*result)