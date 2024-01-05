import numpy as np
import matplotlib.pyplot as plt


class Datapoints:
    def __init__(self, num_points):
        self.num_points = num_points
        self.points = []
        self.expected_outputs = []

    def generate_data(self):
        x_points = np.random.uniform(-np.pi, np.pi, self.num_points)
        y_points = np.random.uniform(-1, 1, self.num_points)

        cos_values = np.cos(x_points)

        for i in range(self.num_points):
            self.points.append((x_points[i], y_points[i]))
            if y_points[i] >= cos_values[i]:
                self.expected_outputs.append([1, 0])  # Above the curve
            else:
                self.expected_outputs.append([0, 1])  # Below the curve

    def plot_data(self):
        above_curve_x = [
            point[0]
            for i, point in enumerate(self.points)
            if self.expected_outputs[i] == [1, 0]
        ]
        above_curve_y = [
            point[1]
            for i, point in enumerate(self.points)
            if self.expected_outputs[i] == [1, 0]
        ]
        below_curve_x = [
            point[0]
            for i, point in enumerate(self.points)
            if self.expected_outputs[i] == [0, 1]
        ]
        below_curve_y = [
            point[1]
            for i, point in enumerate(self.points)
            if self.expected_outputs[i] == [0, 1]
        ]

        plt.figure(figsize=(8, 6))
        x_cos = np.linspace(-np.pi, np.pi, 100)
        y_cos = np.cos(x_cos)
        plt.plot(x_cos, y_cos, color="black", label="cos(x)")
        plt.scatter(above_curve_x, above_curve_y, c="blue", label="Above the curve")
        plt.scatter(below_curve_x, below_curve_y, c="red", label="Below the curve")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        plt.title("Points above and below cos(x)")
        plt.show()


import numpy as np
import matplotlib.pyplot as plt


# Activation function (ReLU)
def ActivationFunction(x):
    return np.maximum(0, x)


# L2 regularization term
def L2Regularization(weights, lambda_reg):
    return 0.5 * lambda_reg * np.sum(weights**2)


class Layer:
    def __init__(self, nodes_in, nodes_out, lambda_reg=0.01):
        self.nodes_in = nodes_in
        self.nodes_out = nodes_out
        self.weights = np.random.uniform(
            -1 / np.sqrt(nodes_in), 1 / np.sqrt(nodes_in), (nodes_in, nodes_out)
        )
        self.biases = np.zeros(nodes_out)
        self.lambda_reg = lambda_reg

    def output(self, inputs):
        outputs = np.dot(inputs, self.weights) + self.biases
        return ActivationFunction(outputs)

    def NodeCost(self, outputActivation, outputExpected):
        error = outputActivation - outputExpected
        return 0.5 * np.sum(error**2) + L2Regularization(
            self.weights, self.lambda_reg
        )

    def Backpropagate(self, dcost_da, inputs, learning_rate):
        dcost_dw = np.outer(inputs, dcost_da) + self.lambda_reg * self.weights
        dcost_db = dcost_da
        dcost_dinputs = np.dot(dcost_da, self.weights.T)
        self.weights -= learning_rate * dcost_dw
        self.biases -= learning_rate * dcost_db
        return dcost_dinputs


class NeuralNetwork:
    def __init__(self, layers_desc: list, lambda_reg=0.01):
        self.layers: list[Layer] = []
        for i in range(len(layers_desc) - 1):
            self.layers.append(Layer(layers_desc[i], layers_desc[i + 1], lambda_reg))
        self.cost_history = []  # List to store cost values for each epoch

    def ComputeOutputs(self, inputs):
        for layer in self.layers:
            inputs = layer.output(inputs)
        return inputs

    def NetworkOutput(self, inputs):
        outputs = self.ComputeOutputs(inputs)
        return np.argmax(outputs)

    def Cost(self, data: Datapoints):
        cost = 0
        for i in range(len(data.points)):
            outputs = self.ComputeOutputs(data.points[i])
            cost += np.sum((outputs - data.expected_outputs[i]) ** 2)
        return cost / len(data.points)

    def Train(self, data: Datapoints, epochs, learning_rate, batch_size, momentum=0.9):
        for epoch in range(epochs):
            total_cost = 0
            data_indices = list(range(len(data.points)))
            np.random.shuffle(data_indices)

            # Initialize momentum for weight updates
            layer_momentum = [np.zeros_like(layer.weights) for layer in self.layers]

            for start in range(0, len(data_indices), batch_size):
                end = min(start + batch_size, len(data_indices))
                batch_indices = data_indices[start:end]

                batch_cost = 0
                for i in batch_indices:
                    inputs = data.points[i]
                    expected_outputs = data.expected_outputs[i]

                    # Forward pass
                    layer_outputs = [inputs]
                    for layer in self.layers:
                        layer_outputs.append(layer.output(layer_outputs[-1]))

                    # Compute cost and backpropagate
                    cost = np.sum((layer_outputs[-1] - expected_outputs) ** 2)
                    batch_cost += cost
                    dcost_da = 2 * (layer_outputs[-1] - expected_outputs)
                    for j in range(len(self.layers) - 1, -1, -1):
                        dcost_da = self.layers[j].Backpropagate(
                            dcost_da, layer_outputs[j], learning_rate
                        )

                total_cost += batch_cost / len(batch_indices)

            # Store the cost for this epoch
            self.cost_history.append(total_cost / (len(data.points) / batch_size))

            # Update weights using momentum
            for i, layer in enumerate(self.layers):
                layer.weights += momentum * layer_momentum[i]

            # Print the cost for this epoch
            print(f"Epoch {epoch + 1}/{epochs}, Cost: {self.cost_history[-1]}")


if __name__ == "__main__":
    # Create a Datapoints object and generate data
    data = Datapoints(num_points=10000)
    data.generate_data()

    # Train the neural network with Mini-Batch Gradient Descent and Momentum
    network = NeuralNetwork([2, 32, 32, 2], lambda_reg=0.0001)
    network.Train(data, epochs=100, learning_rate=0.001, batch_size=100, momentum=0.85)

    # Plot the cost curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(network.cost_history) + 1), network.cost_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.title("Cost vs. Epoch")
    plt.show()

    # Plot the network classification and data points
    # Define the resolution for the graph
    x_resolution = 400
    y_resolution = 400

    # Generate grid points within the specified range
    x_grid = np.linspace(-np.pi, np.pi, x_resolution)
    y_grid = np.linspace(-1, 1, y_resolution)

    # Create an empty grid to store the predictions
    prediction_grid = np.zeros((y_resolution, x_resolution))

    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            # Predict whether the point is above or below the cos function
            inputs = np.array([x, y])
            prediction = network.NetworkOutput(inputs)
            prediction_grid[i, j] = prediction

    # Plot the grid
    plt.figure(figsize=(10, 6))
    plt.imshow(
        prediction_grid,
        extent=(-np.pi, np.pi, -1, 1),
        cmap="coolwarm",
        origin="lower",
        alpha=0.6,
    )
    plt.colorbar()

    # Plot the data points
    above_curve_x = [
        point[0]
        for i, point in enumerate(data.points[0:1000])
        if data.expected_outputs[0:1000][i] == [1, 0]
    ]
    above_curve_y = [
        point[1]
        for i, point in enumerate(data.points[0:1000])
        if data.expected_outputs[0:1000][i] == [1, 0]
    ]
    below_curve_x = [
        point[0]
        for i, point in enumerate(data.points[0:1000])
        if data.expected_outputs[0:1000][i] == [0, 1]
    ]
    below_curve_y = [
        point[1]
        for i, point in enumerate(data.points[0:1000])
        if data.expected_outputs[0:1000][i] == [0, 1]
    ]

    plt.scatter(
        above_curve_x,
        above_curve_y,
        c="blue",
        marker="o",
        s=10,
        label="Above the curve",
    )
    plt.scatter(
        below_curve_x, below_curve_y, c="red", marker="o", s=10, label="Below the curve"
    )
    x_cos = np.linspace(-np.pi, np.pi, 100)
    y_cos = np.cos(x_cos)
    plt.plot(x_cos, y_cos, color="black", label="cos(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.title("Network Predictions and Data Points")
    plt.show()
