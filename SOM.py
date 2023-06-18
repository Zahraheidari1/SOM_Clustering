import numpy as np

class SOM:
    def __init__(self, input_dim, map_size):
        self.input_dim = input_dim
        self.map_size = map_size
        self.weights = np.random.rand(*map_size, input_dim)
        self.umatrix = None

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

        # Calculate U-Matrix after training
        self.calculate_umatrix()

    def find_bmu(self, x):
        # Compute the Euclidean distance between x and each neuron's weights
        dists = np.sum((self.weights - x) ** 2, axis=-1)

        # Find the index of the neuron with the smallest distance to x
        bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
        bmu = self.weights[bmu_idx]

        return bmu, bmu_idx

    def update_weights(self, x, bmu, bmu_idx, epoch, num_epochs):
        # Compute the learning rate and neighborhood radius
        lr = 1.0 - (epoch / float(num_epochs))
        r = self.map_size[0] / 2.0 * lr
    
        # Compute the distance between each neuron and the BMU
        indices = np.indices(self.map_size)
        dists = np.sum(np.abs(indices - np.array(bmu_idx)[:, np.newaxis, np.newaxis]), axis=0)
    
        # Compute the neighborhood function
        h = np.exp(-(dists ** 2) / (2 * r ** 2))
    
        # Compute the delta for updating the weights
        delta = lr * h[..., np.newaxis] * (x - self.weights)
    
        # Update the weights using element-wise addition
        self.weights += delta

    def calculate_umatrix(self):
        # Initialize the U-Matrix array with zeros
        umatrix = np.zeros((2 * self.map_size[0] - 1, 2 * self.map_size[1] - 1))

        # Iterate over each neuron in the SOM
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                # Get the current neuron's weights
                current_weights = self.weights[i, j]

                # Calculate the U-Matrix value for the current neuron
                if i > 0:
                    # Calculate the U-Matrix value for the left neighbor
                    left_weights = self.weights[i - 1, j]
                    umatrix[2 * i - 1, 2 * j] = np.linalg.norm(current_weights - left_weights)

                if i < self.map_size[0] - 1:
                    # Calculate the U-Matrix value for the right neighbor
                    right_weights = self.weights[i + 1, j]
                    umatrix[2 * i + 1, 2 * j] = np.linalg.norm(current_weights - right_weights)

                if j > 0:
                    # Calculate the U-Matrix value for the top neighbor
                    top_weights = self.weights[i, j - 1]
                    umatrix[2 * i, 2 * j - 1] = np.linalg.norm(current_weights - top_weights)

                if j < self.map_size[1] - 1:
                    # Calculate the U-Matrix value for the bottom neighbor
                    bottom_weights = self.weights[i, j + 1]
                    umatrix[2 * i, 2 * j + 1] = np.linalg.norm(current_weights - bottom_weights)

        self.umatrix = umatrix
        
    def get_u_matrix(self):
        return self.umatrix    

    def quantization_error(self, data):
        total_error = 0.0

        for x in data:
            bmu, _ = self.find_bmu(x)
            error = np.linalg.norm(x - bmu)
            total_error += error

        mean_error = total_error / len(data)
        return mean_error
    
    def topographic_error(self, data):
        total_error = 0

        for x in data:
            bmu, bmu_idx = self.find_bmu(x)
            neighbors = self.find_neighbors(bmu_idx)

            if not any(np.array_equal(bmu_idx, neighbor) for neighbor in neighbors):
                total_error += 1

        error_rate = total_error / len(data)
        return error_rate
    
    
    def find_neighbors(self, idx):
        neighbors = []
        i, j = idx

        if i > 0:
            neighbors.append((i - 1, j))
        if i < self.map_size[0] - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < self.map_size[1] - 1:
            neighbors.append((i, j + 1))

        return neighbors


    def cluster(self, data):
        labels = []
    
        for x in data:
            # Find the best matching unit (BMU) for the data point
                bmu, bmu_idx = self.find_bmu(x)

            # Assign the data point to the cluster of the BMU
                labels.append(bmu_idx)

        return labels

