import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def recall(self, input_pattern, iterations=10):
        state = np.copy(input_pattern)
        for _ in range(iterations):
            for i in range(self.size):
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
        return state

# Training patterns and test input
patterns = [np.array([1, -1, 1, -1, 1]), np.array([-1, 1, -1, 1, -1])]
test_pattern = np.array([1, -1, -1, -1, 1])

# Initialize and train the Hopfield network
hopfield_net = HopfieldNetwork(size=5)
hopfield_net.train(patterns)

# Recall the pattern based on the test input
recalled_pattern = hopfield_net.recall(test_pattern)
print("Recalled Pattern:", recalled_pattern)
