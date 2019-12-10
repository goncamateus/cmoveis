import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.distance import distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
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

    def __init__(self, data_path='LocTreino_Equipe_3.csv', log_dir='log_mlp/'):
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
        self.data_train = self.data_train[:int(0.7 * len(self.data_train))]
        self.data_test = self.data_test[int(0.7 * len(self.data_test)):]

    def declare_model(self):
        self.model = MLP(
            self.X[0].shape[0], self.y[0].shape[0])
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_function = F.mse_loss

    def train(self, epochs=500):
        for epoch in range(epochs):
            for _ in range(100):
                batch = self.data_train.sample(128)
                x_batch = batch[[
                    x for x in batch.columns if x not in ('lat', 'lon')]].values
                y_batch = batch[[
                    x for x in batch.columns if x in ('lat', 'lon')]].values

                x_batch = torch.tensor(
                    x_batch).float()
                y_batch = torch.tensor(
                    y_batch).float()

                out = self.model(x_batch)
                loss = self.loss_function(out, y_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if abs(self.model_loss - loss.item()) < 0.1:
                    if self.early_stop < 10:
                        self.model_loss = loss.item()
                        self.early_stop += 1
                    else:
                        break
                else:
                    self.model_loss = loss.item()
                    self.early_stop = 0

    def test(self):
        x_batch = self.data_test[[
            x for x in self.data_test.columns if x not in ('lat', 'lon')]].values
        y_batch = self.data_test[[
            x for x in self.data_test.columns if x in ('lat', 'lon')]].values

        x_batch = torch.tensor(x_batch).float()
        y_batch = torch.tensor(y_batch).float()

        self.model.eval()
        out = self.model(x_batch)
        out = out.detach().numpy()
        return out, y_batch
    

    def result_analisys(self, out, y_batch):
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in y_batch], [x[1]
                                          for x in y_batch], 'o', label='Real')
        ax.plot([x[0] for x in out], [x[1] for x in out], 'x', label='MLP')
        ax.legend()
        ax.set_title('Comparação')
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
        fig.savefig('mlp_hist_dists.png')


        print(f'Erro Médio: {np.mean(dists)}\nErro Mínimo: {min(dists)}\n\
Erro Máximo: {max(dists)}\nDesvio Padrão: {np.std(dists)}')


if __name__ == "__main__":
    mlp = ProjectMLP()
    mlp.train()
    result = mlp.test()
    mlp.result_analisys(*result)