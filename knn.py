import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        self.data_train = self.data_train[:int(0.9 * len(self.data_train))]
        self.data_test = self.data_test[int(0.9 * len(self.data_test)):]

    def declare_model(self):
        params = {'n_neighbors': [1, 3, 5, 8, 13],
                  'metric': ['euclidian', 'manhattan'],
                  'weights': ['uniform', 'distance'],
                  'leaf_size': [30, 33, 55, 88]
                  }
        knn = KNeighborsRegressor()
        self.model = GridSearchCV(knn, params,
                                  verbose=1, cv=3, n_jobs=-1)

    def train(self):
        batch = self.data_train
        x_batch = batch[[
            x for x in batch.columns if x not in ('lat', 'lon')]].values
        y_batch = batch[[
            x for x in batch.columns if x in ('lat', 'lon')]].values

        self.model.fit(x_batch, y_batch)
        self.model = knn.model.best_estimator_
        print(self.model)

    def test(self):
        x_batch = self.data_test[[
            x for x in self.data_test.columns if x not in ('lat', 'lon')]].values
        y_batch = self.data_test[[
            x for x in self.data_test.columns if x in ('lat', 'lon')]].values

        out = self.model.predict(x_batch)
        plt.plot([x[0] for x in y_batch], [x[1] for x in y_batch], 'o', label='Real')
        plt.plot([x[0] for x in out], [x[1] for x in out], 'x', label='KNN')
        plt.legend()
        plt.set_title('Mapa de comparação')
        plt.savefig('knn_map.png')

        error = y_batch - out
        errors_x = error[:][0]
        errors_y = error[:][1]
        
        plt.clf()
        plt.boxplot(errors_x)
        plt.set_title('Box plot X')
        plt.savefig('knn_boxplot_x.png')

        plt.clf()
        plt.boxplot(errors_y)
        plt.set_title('Box plot Y')
        plt.savefig('knn_boxplot_y.png')

        errors_x = np.average(errors_x)
        errors_y = np.average(errors_y)

        return errors_x, errors_y


if __name__ == "__main__":
    knn = ProjectKNN()
    knn.train()
    result = knn.test()
    print(result)
