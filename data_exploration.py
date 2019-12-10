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

    def preproccess_data(self):
        self.data = self.data.drop_duplicates()
        self.data = self.data.dropna()

    def analisys(self):   
        
        
        fig, ax = plt.subplots()
        ax.boxplot(dists)
        ax.set_title('BoxPlot Dists')
        fig.savefig('xgboost_boxplot_dists.png')

        print(f'Erro Médio: {np.mean(dists)}\nErro Mínimo: {min(dists)}\n\
Erro Máximo: {max(dists)}\nDesvio Padrão: {np.std(dists)}')


if __name__ == "__main__":
    ProjectDataExploration()