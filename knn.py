import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from geopy.distance import distance
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor


class ProjectKNN:
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
        # params = {'n_neighbors': [1, 3, 5, 8, 13],
        #           'metric': ['euclidian', 'manhattan'],
        #           'weights': ['uniform', 'distance'],
        #           'leaf_size': [30, 33, 55, 88]
        #           }
        # knn = KNeighborsRegressor()
        # self.model = GridSearchCV(knn, params,
        #                           verbose=1, cv=3, n_jobs=-1)
        self.model = KNeighborsRegressor(algorithm='auto', leaf_size=30,
                                         metric='manhattan',
                                         metric_params=None,
                                         n_jobs=-1, n_neighbors=1,
                                         p=2, weights='distance')

    def train(self):
        batch = self.data_test if not self.ft else self.data
        x_batch = batch[[
            x for x in batch.columns if x not in ('lat', 'lon', 'pontoId')]].values
        y_batch = batch[[
            x for x in batch.columns if x in ('lat', 'lon')]].values

        self.model.fit(x_batch, y_batch)
        # self.model = knn.model.best_estimator_
        # print(self.model)

    def test(self):
        x_batch = self.data_test[[
            x for x in self.data_test.columns if x not in ('lat', 'lon', 'pontoId')]].values
        y_batch = self.data_test[[
            x for x in self.data_test.columns if x in ('lat', 'lon')]].values

        out = self.model.predict(x_batch)
        return out, y_batch

    def test_arx(self, path):
        test = pd.read_csv(path)
        x_batch = test[[
            x for x in test.columns if x not in ('lat', 'lon', 'pontoId')]].values
        y_batch = test[[
            x for x in test.columns if x in ('lat', 'lon')]].values
        out = self.model.predict(x_batch)
        df = pd.DataFrame()
        df['pontoId'] = test.pontoId
        df['lat_pred'] = out[:, 0]
        df['lon_pred'] = out[:, 1]
        df.to_csv('Resultados_Equipe3_Metodo_KNN.csv', index=False)

    def result_analisys(self, out, y_batch):
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in y_batch], [x[1]
                                          for x in y_batch], 'o', label='Real')
        ax.plot([x[0] for x in out], [x[1] for x in out], 'x', label='KNN')
        ax.legend()
        ax.set_title('Comparação')
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        fig.savefig('knn_map.png')

        dists = []
        for i in range(len(out)):
            dist = distance(y_batch[i], out[i]).km*1000
            dists.append(dist)

        fig, ax = plt.subplots()
        ax.boxplot(dists)
        ax.set_title('KNN BoxPlot Dists')
        fig.savefig('knn_boxplot_dists.png')

        fig, ax = plt.subplots()
        ax.hist(dists)
        ax.set_title('KNN Histogram Dists')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Distance (M)')
        fig.savefig('knn_hist_dists.png')

        print(f'Erro Médio: {np.mean(dists)}\nErro Mínimo: {min(dists)}\n\
Erro Máximo: {max(dists)}\nDesvio Padrão: {np.std(dists)}')


if __name__ == "__main__":
    knn = ProjectKNN(ft=True)
    knn.train()
    with open('knn.bin', 'wb') as knnf:
        pickle.dump(knn, knnf)

    # result = knn.test()
    # knn.result_analisys(*result)
