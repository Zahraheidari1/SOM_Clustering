import numpy as np

class SOM:
    def __init__(self, input_dim, map_size):
        self.input_dim = input_dim
        self.map_size = map_size
        self.weights = np.random.rand(*map_size, input_dim)

    def train(self, data, num_epochs):
        for epoch in range(num_epochs):
            np.random.shuffle(data)

            for x in data:
                # Find the best matching unit (BMU) 
                bmu, bmu_idx = self.find_bmu(x)

                # Update the weights of the BMU and its neighbors
                self.update_weights(x, bmu, bmu_idx, epoch, num_epochs)


    def find_bmu(self, x):
        dists = np.sum((self.weights - x) ** 2, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
        bmu = self.weights[bmu_idx]

        return bmu, bmu_idx

    def update_weights(self, x, bmu, bmu_idx, epoch, num_epochs):
        
        lr = 1.0 - (epoch / float(num_epochs))
        r = self.map_size[0] / 2.0 * lr
    
        indices = np.indices(self.map_size)
        dists = np.sqrt(np.sum((indices - np.array(bmu_idx)[:, np.newaxis, np.newaxis]) ** 2, axis=0))
    
        h = np.exp(-(dists ** 2) / (2 * r ** 2))
    
        # Compute the delta for updating the weights
        delta = lr * h[..., np.newaxis] * (x - self.weights)
    
        self.weights += delta

    def cluster(self, data):
        labels = []
    
        for x in data:
            # Find the best matching unit (BMU) for the data point
                bmu, bmu_idx = self.find_bmu(x)

            # Assign the data point to the cluster of the BMU
                labels.append(bmu_idx)

        return labels

