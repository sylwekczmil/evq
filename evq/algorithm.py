import numpy as np


class EVQ:
    """
    Note that this algorithm requires normalized data in range [0,1]
    """

    def __init__(self, number_of_classes: int, vigilance: float, *args, **kwargs):
        self.number_of_classes = number_of_classes
        self.vigilance = vigilance  # [0, 1]
        self.clusters = None  # 2d array
        self.data_points_per_cluster = None  # 1d array
        self.hit_matrix = None  # 2d array

    def _init_using_first_sample(self, x, y):
        self.clusters = np.atleast_2d(x)
        self.data_points_per_cluster = np.array([1])
        cluster_hits = np.zeros(self.number_of_classes)
        cluster_hits[y] = 1
        self.hit_matrix = np.atleast_2d(cluster_hits)

    def _add_cluster(self, x, y):
        self.clusters = np.vstack((self.clusters, x))
        self.data_points_per_cluster = np.append(self.data_points_per_cluster, 1)
        cluster_hits = np.zeros(self.number_of_classes)
        cluster_hits[y] = 1
        self.hit_matrix = np.vstack((self.hit_matrix, cluster_hits))

    def _match_cluster(self, x):
        diff = np.absolute(self.clusters - x)
        distances = np.linalg.norm(diff, axis=1, ord=2)
        min_distance = np.min(distances)
        if min_distance < self.vigilance:
            return np.argmin(distances)
        return -1

    def _update_cluster(self, x, y, winning_cluster):
        eta = 0.5 / self.data_points_per_cluster[winning_cluster]
        self.clusters[winning_cluster] = self.clusters[winning_cluster] + (eta * (x - self.clusters[winning_cluster]))
        self.data_points_per_cluster[winning_cluster] += 1
        self.hit_matrix[winning_cluster, y] += 1

    def fit(self, x, y, epochs=10, permute=True):
        samples = np.atleast_2d(np.array(x, dtype=float))
        labels = np.atleast_1d(y)

        if self.clusters is None:
            self._init_using_first_sample(samples[0], labels[0])

        _samples = samples
        _labels = labels
        for epoch in range(epochs):
            if permute:
                idx = np.random.permutation(np.arange(len(samples)))
                _samples = samples[idx]
                _labels = labels[idx]

            for _sample, _label in zip(_samples, _labels):
                winning_cluster = self._match_cluster(_sample)
                if winning_cluster == -1:
                    self._add_cluster(_sample, _label)
                else:
                    self._update_cluster(_sample, _label, winning_cluster)

        return self

    def partial_fit(self, x, y):
        return self.fit(x, y, epochs=1, permute=False)

    def _predict_sample_variant_a(self, x):
        diff = np.absolute(self.clusters - x)
        distances = np.linalg.norm(diff, axis=1, ord=2)
        closest_cluster = np.argmin(distances)
        return np.argmax(self.hit_matrix[closest_cluster, :])

    def _predict_sample_variant_b(self, x):
        diff = np.absolute(self.clusters - x)
        distances = np.linalg.norm(diff, axis=1, ord=2)
        if len(distances) > 1:
            # closest and second closest
            c1, c2 = np.argpartition(distances, 1)[:2]
            i = 1 - (distances[c1] / distances[c2])
            w_i_1 = 0.5 + i
            w_i_2 = 0.5 - i
            conf_1 = w_i_1 * self.hit_matrix[c1] / self.data_points_per_cluster[c1]
            conf_2 = w_i_2 * self.hit_matrix[c2] / self.data_points_per_cluster[c2]
            conf = conf_1 + conf_2
            return np.argmax(conf)
        return -1

    def predict(self, x, variant='b'):
        samples = np.atleast_2d(x)
        labels = np.full(len(samples), -1)

        if self.clusters is None:
            return labels

        predict_method = self._predict_sample_variant_a if variant == 'a' else self._predict_sample_variant_b

        for i, sample in enumerate(samples):
            labels[i] = predict_method(sample)
        return labels
