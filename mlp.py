import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from geopy.distance import distance
from tqdm import tqdm, trange


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.layers(x)


class ProjectMLP:
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
        self.data = self.data.drop(
            columns=["pontoId"])
        self.X = self.data[[
            x for x in self.data.columns if x not in ('lat', 'lon')]].values
        self.y = self.data[[
            x for x in self.data.columns if x in ('lat', 'lon')]].values
        self.data_train = self.data_test = self.data
        self.data_train = self.data_train[:int(0.9 * len(self.data_train))]
        self.data_test = self.data_test[int(0.9 * len(self.data_test)):]

    def declare_model(self):
        self.model_x = MLP(
            self.X[0].shape[0], self.y[0].shape[0]).cuda()
        self.optimizer_x = optim.Adam(self.model_x.parameters(), lr=1e-3)
        self.model_y = MLP(
            self.X[0].shape[0], self.y[0].shape[0]).cuda()
        self.optimizer_y = optim.Adam(self.model_y.parameters(), lr=1e-3)
        self.loss_function = F.mse_loss

    def train(self, epochs=50):
        losses_x = losses_y = list()
        t = trange(epochs, desc='MLP training', leave=True)
        for epoch in t:
            try:
                for _ in range(1000):
                    batch = self.data_test if not self.ft else self.data
                    x_batch = batch[[
                        x for x in batch.columns if x not in ('lat', 'lon', 'pontoId')]].values
                    y_batch_x = batch[[
                        x for x in batch.columns if x in ('lat')]].values
                    y_batch_y = batch[[
                        x for x in batch.columns if x in ('lon')]].values

                    x_batch = torch.tensor(
                        x_batch).float().cuda()
                    y_batch_x = torch.tensor(
                        y_batch_x).float().cuda()
                    y_batch_y = torch.tensor(
                        y_batch_y).float().cuda()

                    out_x = self.model_x(x_batch)
                    out_y = self.model_y(x_batch)

                    loss_x = self.loss_function(out_x, y_batch_x)
                    self.optimizer_x.zero_grad()
                    loss_x.backward()
                    self.optimizer_x.step()
                    losses_x.append(loss_x.item())

                    loss_y = self.loss_function(out_y, y_batch_y)
                    self.optimizer_y.zero_grad()
                    loss_y.backward()
                    self.optimizer_y.step()
                    losses_y.append(loss_y.item())
            except KeyboardInterrupt:
                break
            t.set_description(
                f"loss_lat {np.average(losses_x)}; loss_lon {np.average(losses_y)}")
            t.refresh()

    def test(self):
        x_batch = self.data_test[[
            x for x in self.data_test.columns if x not in ('lat', 'lon', 'pontoId')]].values
        y_batch_x = self.data_test[[
            x for x in self.data_test.columns if x in ('lat')]].values
        y_batch_y = self.data_test[[
            x for x in self.data_test.columns if x in ('lon')]].values
        y_batch = self.data_test[[
            x for x in self.data_test.columns if x in ('lat', 'lon')]].values

        x_batch = torch.tensor(x_batch).float().cuda()
        y_batch_x = torch.tensor(y_batch_x).float().cuda()
        y_batch_y = torch.tensor(y_batch_y).float().cuda()

        self.model_x.eval()
        out_x = self.model_x(x_batch)
        out_x = out_x.detach().cpu().numpy()

        self.model_y.eval()
        out_y = self.model_y(x_batch)
        out_y = out_y.detach().cpu().numpy()

        out = list(zip(out_x, out_y))
        return out, y_batch

    def test_arx(self, path):
        test = pd.read_csv(path)
        x_batch = test[[
            x for x in test.columns if x not in ('lat', 'lon', 'pontoId')]].values
        y_batch_x = test[[
            x for x in test.columns if x in ('lat')]].values
        y_batch_y = test[[
            x for x in test.columns if x in ('lon')]].values

        x_batch = torch.tensor(x_batch).float().cuda()
        y_batch_x = torch.tensor(y_batch_x).float().cuda()
        y_batch_y = torch.tensor(y_batch_y).float().cuda()

        self.model_x.eval()
        out_x = self.model_x(x_batch)
        out_x = out_x.detach().cpu().numpy()

        self.model_y.eval()
        out_y = self.model_y(x_batch)
        out_y = out_y.detach().cpu().numpy()

        df = pd.DataFrame()
        df['pontoId'] = test.pontoId
        df['lat'] = y_batch_x.cpu().numpy()
        df['lon'] = y_batch_y.cpu().numpy()
        df['lat_pred'] = out_x
        df['lon_pred'] = out_y
        df.to_csv('Resultados_Equipe3_Metodo_MLP.csv', index=False)

    def result_analisys(self, out, y_batch):
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in y_batch], [x[1]
                                          for x in y_batch], 'o', label='Real')
        ax.plot([x[0] for x in out], [x[1] for x in out], 'x', label='MLP')
        ax.legend()
        ax.set_title('Comparação')
        ax.set_ylabel('Latitude')
        ax.set_xlabel('Longitude')
        fig.savefig('mlp_map.png')

        dists = []
        for i in range(len(out)):
            dist = distance(y_batch[i], out[i]).km*1000
            dists.append(dist)

        fig, ax = plt.subplots()
        ax.boxplot(dists)
        ax.set_title('BoxPlot Dists')
        fig.savefig('mlp_boxplot_dists.png')

        fig, ax = plt.subplots()
        ax.hist(dists)
        ax.set_title('Histogram Dists')
        ax.set_ylabel('Frequency')
        ax.set_xlabel('Distance (M)')
        fig.savefig('mlp_hist_dists.png')

        print(f'Erro Médio: {np.mean(dists)}\nErro Mínimo: {min(dists)}\n\
Erro Máximo: {max(dists)}\nDesvio Padrão: {np.std(dists)}')


if __name__ == "__main__":
    mlp = ProjectMLP(ft=True)
    mlp.train()
    with open('mlp.bin', 'wb') as knnf:
        pickle.dump(mlp, knnf)
    # result = mlp.test()
    # mlp.result_analisys(*result)
