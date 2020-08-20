import numpy as np
# np.random.seed(0)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

epochs = 10000
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Random weights and bias initialization
hidden_weights = np.array([[0.1, 0.2], [0.3, 0.4]])
hidden_bias = np.array([[0.5, 0.6]])
output_weights = np.array([[0.1], [0.2]])
output_bias = np.array([[0.3]])

print("Initial hidden weights: ", end='')
print(*hidden_weights)
print("Initial hidden biases: ", end='')
print(*hidden_bias)
print("Initial output weights: ", end='')
print(*output_weights)
print("Initial output biases: ", end='')
print(*output_bias)


# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = predicted_output - expected_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    d_hidden_layer = error_hidden_layer * \
        sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights -= hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias -= np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights -= inputs.T.dot(d_hidden_layer) * lr
    hidden_bias -= np.sum(d_hidden_layer, axis=0, keepdims=True) * lr

print("========== FLAG ==========")
print(hidden_weights)
print(hidden_bias)
print(output_weights)
print(output_bias)

print("Final hidden weights: ", end='')
print(*hidden_weights)
print("Final hidden bias: ", end='')
print(*hidden_bias)
print("Final output weights: ", end='')
print(*output_weights)
print("Final output bias: ", end='')
print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*predicted_output)


def predict(i1, i2, hw, hb, ow, ob):
    single_point = np.array([[i1, i2]])
    in_h = np.dot(single_point, hw)
    in_h += hb
    out_h = sigmoid(in_h)

    in_o = np.dot(out_h, ow)
    in_o += ob
    out_o = sigmoid(in_o)
    return out_o


print("\nTESTING\n")
print("XOR of (0,0) (Expected: 0):")
print(predict(0, 0, hidden_weights, hidden_bias, output_weights, output_bias))

print("XOR of (0,1) (Expected: 1):")
print(predict(0, 1, hidden_weights, hidden_bias, output_weights, output_bias))

print("XOR of (1,0) (Expected: 1):")
print(predict(1, 0, hidden_weights, hidden_bias, output_weights, output_bias))

print("XOR of (1,1) (Expected: 0):")
print(predict(1, 1, hidden_weights, hidden_bias, output_weights, output_bias))
