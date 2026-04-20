import random
from itertools import pairwise
import copy

# goal is to make this as general as possible, so you can change the node sizes and etc and still have a valid model
# this is not optimized

# loss = MSE: L = mean((predictions - targets) ** 2)
# activation function = ReLU: f(x) = max(0, x)
# problem: XOR

def init_wi():
    return random.uniform(-1, 1)

# weights[i][j][k] will give you the kth output weight belonging to the jth node in the ith layer (0-indexed)
def get_weights_and_biases(layer_sizes):
    layer_pairs = [[x, y] for x, y in pairwise(layer_sizes)]
    weights = [[list(map(lambda x: init_wi(), [0]*i[1])) for j in range(i[0])] for i in layer_pairs]
    biases = [[0]*i for i in layer_sizes[1:]]
    return [weights, biases]

def ReLU(x):
    return max(0, x)

def forward_prop(weights, biases, inputs, layer_sizes):
    layer_vals = [inputs] + copy.deepcopy(biases)
 
    # iterates through layers
    for layer_index in range(len(layer_sizes)-1):
 
        # iterates through nodes
        for node_index in range(layer_sizes[layer_index]):

            # iterates through outputs
            for output_index in range(layer_sizes[layer_index+1]):
                layer_vals[layer_index+1][output_index] += (layer_vals[layer_index][node_index] * weights[layer_index][node_index][output_index])
            
        if layer_index+2 < len(layer_sizes):
            for output_index in range(layer_sizes[layer_index+1]):
                layer_vals[layer_index+1][output_index] = ReLU(layer_vals[layer_index+1][output_index])

    return layer_vals

def derivative_of_ReLU(x):
    if x > 0:
        return 1
    return 0

def derivative_of_loss(pred, target):
    return 2*(pred - target)

def loss(pred, target):
    return (pred-target)**2

# we need every node AND weight except the first one
def back_prop(weights, biases, layer_vals, layer_sizes, target, learning_rate):
    layer_pairs = [[x, y] for x, y in pairwise(layer_sizes)]
    gradients = [[[0]*i[1] for j in range(i[0])] for i in layer_pairs]
    intermediate_weights = [[[0]*i[1] for j in range(i[0])] for i in layer_pairs]
    bias_gradients = [[0]*i for i in layer_sizes]
    for layer_index in range(len(layer_sizes)-1, 0, -1):

        for node_index in range(layer_sizes[layer_index]):

            if (layer_index < len(layer_sizes) - 1):
                for weight_index in range(layer_sizes[layer_index+1]):
                    bias_gradients[layer_index][node_index] += intermediate_weights[layer_index][node_index][weight_index]
                bias_gradients[layer_index][node_index] *= derivative_of_ReLU(layer_vals[layer_index][node_index])

            else:
                bias_gradients[layer_index][node_index] = derivative_of_loss(layer_vals[layer_index][node_index], target[node_index])
            
            for weight_index in range(layer_sizes[layer_index-1]):
                intermediate_weights[layer_index-1][weight_index][node_index] += bias_gradients[layer_index][node_index]*weights[layer_index-1][weight_index][node_index]
                gradients[layer_index-1][weight_index][node_index] += bias_gradients[layer_index][node_index]*layer_vals[layer_index-1][weight_index]
    bias_gradients = bias_gradients[1:]
    for layer_index in range(len(layer_sizes)-1):
        for node_index in range(layer_sizes[layer_index]):
            for output_index in range(layer_sizes[layer_index+1]):
                weights[layer_index][node_index][output_index] -= gradients[layer_index][node_index][output_index]*learning_rate
    for layer_index in range(1, len(layer_sizes)):
        for node_index in range(layer_sizes[layer_index]):
            biases[layer_index-1][node_index] -= bias_gradients[layer_index-1][node_index]*learning_rate
    return [weights, biases]

def train(inputs, targets, layer_sizes, epochs, learning_rate, print_every):
    [weights, biases] = get_weights_and_biases(layer_sizes)
    for i in range(epochs):
        for j in range(len(targets)):
            layer_vals = forward_prop(weights, biases, inputs[j], layer_sizes)
            [weights, biases] = back_prop(weights, biases, layer_vals, layer_sizes, targets[j], learning_rate)

    return [weights, biases]

def test(weights, biases, inp, layer_sizes):
    return forward_prop(weights, biases, inp, layer_sizes)[-1]

def main():
    inputs = [[0, 0], [0, 1], [1,0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    layer_sizes = [2, 4, 1]

    [weights, biases] = train(inputs, targets, layer_sizes, 100, 0.1, 10)
    for i in range(len(inputs)):
        print('test ', i+1, test(weights, biases, inputs[i], layer_sizes))

if __name__ == "__main__":
    main()