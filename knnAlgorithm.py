import numpy as np

class Knn:
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def euclidean(self, val1, val2):
        distance = 0.0
        for index in range(len(val1)):
            distance += (val1[index] - val2[index]) ** 2
            euclidean_dist = np.sqrt(distance)
        return euclidean_dist

    # funkcija, ki vraca neighbors
    def getNeighbors(self, X_train, y_train):
        eucl_dist = []
        for row in X_train:
            dist = self.euclidean(row, y_train)
            eucl_dist.append(dist)

        neighbors = np.array(eucl_dist).argsort()[:self.n_neighbors]

        get_knn_val = []

        for i in range(len(neighbors)):
            n_index = neighbors[i]
            e_dist = eucl_dist[i]
            get_knn_val.append(n_index, e_dist)

        return get_knn_val

    def fit_alg(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict_knn(self, X):
        prediction = []
        for i in range(len(X)):
            euclidean = []
            for row in self.X_train:
                # za vsako vrstico v X_train, pronajdi eucl_distance za X[i]
                eucl = self.euclidean(row, X[i])
                euclidean.append(eucl)
            # sortiraj euclidian_distances v narascajocem redu da bo najblizji na zacetku
            neighbors = np.array(euclidean).argsort()[: self.n_neighbors]

            count_neigh = {}
            for n in neighbors:
                # y_train[n] potem vraca 0 ali 1 in to vstavimo v prediction
                if self.y_train[n] in count_neigh:
                    count_neigh[self.y_train[n]] += 1
                else:
                    count_neigh[self.y_train[n]] = 1
            # use majority class labels of those closest points to predict the label of the test point
            # count the occurencies of values, return the most frequent
            prediction.append(max(count_neigh, key=count_neigh.get))
        return prediction


