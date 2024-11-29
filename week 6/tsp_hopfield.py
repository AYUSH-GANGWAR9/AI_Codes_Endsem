import numpy as np

class TSPHopfield:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.neurons = num_cities * num_cities
        self.weights = np.zeros((self.neurons, self.neurons))
        self.distance_matrix = None

    def initialize_weights(self, distances, A=500, B=500, C=200, D=200):
        self.distance_matrix = distances
        N = self.num_cities

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    for l in range(N):
                        neuron_1 = i * N + j
                        neuron_2 = k * N + l
                        if i == k and j != l:
                            self.weights[neuron_1][neuron_2] -= A
                        if i != k and j == l:
                            self.weights[neuron_1][neuron_2] -= B
                        if i == k and j == l:
                            self.weights[neuron_1][neuron_2] -= C
                        if (i + 1) % N == k:
                            self.weights[neuron_1][neuron_2] -= D * distances[j][l]

    def optimize_state(self, initial_state, iterations=100):
        state = np.copy(initial_state)
        for _ in range(iterations):
            for neuron in range(self.neurons):
                activation = np.dot(self.weights[neuron], state)
                state[neuron] = 1 if activation > 0 else 0
        return state

# Problem setup
num_cities = 10
distance_matrix = np.random.randint(1, 100, size=(num_cities, num_cities))

# Initialize and configure the Hopfield network for TSP
tsp_hopfield = TSPHopfield(num_cities)
tsp_hopfield.initialize_weights(distance_matrix)

# Generate an initial random state and optimize it
initial_state = np.random.choice([0, 1], size=(num_cities * num_cities))
optimized_state = tsp_hopfield.optimize_state(initial_state)

print("Optimized State:", optimized_state)
