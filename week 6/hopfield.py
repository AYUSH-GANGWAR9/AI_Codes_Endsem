import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # Prevent self-connections

    def retrieve(self, pattern, max_iterations=100):
        state = pattern.copy()
        for _ in range(max_iterations):
            updated_state = np.sign(self.weights @ state)
            updated_state[updated_state == 0] = 1  # Handle zero activations
            if np.array_equal(state, updated_state):
                break
            state = updated_state
        return state

def generate_patterns(num_patterns, size):
    return [np.random.choice([-1, 1], size=size) for _ in range(num_patterns)]

if __name__ == "__main__":
    size = 100  # Number of neurons
    num_patterns = 15  # Number of patterns to store
    patterns = generate_patterns(num_patterns, size)

    # Initialize and train the Hopfield network
    hopfield_net = HopfieldNetwork(size)
    hopfield_net.train(patterns)

    # Test pattern retrieval
    correct_retrievals = 0
    for pattern in patterns:
        noisy_pattern = pattern.copy()
        noisy_pattern[:5] *= -1  # Introduce noise in the first 5 elements
        retrieved_pattern = hopfield_net.retrieve(noisy_pattern)
        if np.array_equal(retrieved_pattern, pattern):
            correct_retrievals += 1

    print(f"Successfully retrieved patterns: {correct_retrievals}/{num_patterns}")
