import numpy as np

class SOM:
    def __init__(self, input_dim, map_size):
        self.input_dim = input_dim
        self.map_size = map_size
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)

    def train(self, data, num_epochs):
        for epoch in range(num_epochs):
            # Shuffle the data randomly
            np.random.shuffle(data)

            # Iterate over each data point
            for x in data:
                # Find the best matching unit (BMU) for the data point
                bmu, bmu_idx = self.find_bmu(x)

                # Update the weights of the BMU and its neighbors
                self.update_weights(x, bmu, bmu_idx, epoch, num_epochs)

    def find_bmu(self, x):
        # Compute the Euclidean distance between x and each neuron's weights
        dists = np.sum((self.weights - x) ** 2, axis=2)

        # Find the index of the neuron with the smallest distance to x
        bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
        bmu = self.weights[bmu_idx[0], bmu_idx[1]]

        return bmu, bmu_idx

    def update_weights(self, x, bmu, bmu_idx, epoch, num_epochs):
        # Compute the learning rate and neighborhood radius
        lr = 1.0 - (epoch / float(num_epochs))
        r = self.map_size[0] / 2.0 * lr

        # Compute the distance between each neuron and the BMU
        dists = np.sqrt(np.sum((np.indices(self.map_size).T - np.array(bmu_idx)) ** 2, axis=1))

        # Compute the neighborhood function
        h = np.exp(-(dists ** 2) / (2 * r ** 2))

        # Update the weights of the BMU and its neighbors
        delta = lr * h[:, np.newaxis] * (x - self.weights)
        self.weights += delta

    def cluster(self, data):
        labels = []
        for x in data:
            # Find the best matching unit (BMU) for the data point
            bmu, bmu_idx = self.find_bmu(x)

            # Assign the data point to the cluster of the BMU
            labels.append(bmu_idx)

        return np.array(labels)
