import numpy as np, random as rd

# Misc functions
def sigmoid(x): # activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Network(object):

    def __init__(self, layer_sizes):
        '''Initialize network with random weights and biases.'''
        self.layer_sizes = layer_sizes
        self.biases = [np.random.randn(layer_size, 1) for layer_size in layer_sizes[1:]] # generate a column vector of biases for each layer except the input layer
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])] # generate matrix of weights between each layer. We use (y, x) instead of (x, y) as it is easier to index during backpropagation

    def feed_forward(self, layer_input):
        '''Get output of network, given input as a column vector.'''
        for weight_set, bias_set in zip(self.weights, self.biases):
            layer_input = sigmoid(np.dot(weight_set, layer_input) + bias_set) # the operation produces a column vector of the activations of the next layer
        return layer_input

    def stochastic_gradient_descent(self, training_data, training_labels, num_epochs, eta):
        '''Update the network using SGD. The training data is a 4-d numpy array containing mini-batches which in turn contain images represented as a column vector. Eta is the learning rate.'''
        for epoch in range(1, num_epochs + 1):
            random_order = np.random.permutation(len(training_data))
            training_data = training_data[random_order]
            training_labels = training_labels[random_order]
            count = 1
            for training_data_batch, training_label_batch in zip(training_data, training_labels):
                print("Training network... Batch {} in epoch {}.".format(count, epoch))
                self.update_network_using_mini_batch(zip(training_data_batch, training_label_batch), eta, len(training_data_batch))
                count += 1
            print("Epoch {} complete".format(epoch))

    def update_network_using_mini_batch(self, mini_batch, eta, mini_batch_size):
        '''Apply gradient descent to a minibatch.'''
        nabla_weights = [np.zeros(weight_set.shape) for weight_set in self.weights] # init zeros in the shape of all weights and biases as these will be added to and eventually averagaged
        nabla_biases = [np.zeros(bias_set.shape) for bias_set in self.biases]
        for x, y in mini_batch: # for each training sample in the mini-batch
            delta_nabla_weights, delta_nabla_biases = self.backpropagate(x, y) # gradient as computed using one sample
            nabla_weights = [nabla_weight_set + delta_nabla_weight_set for nabla_weight_set, delta_nabla_weight_set in zip(nabla_weights, delta_nabla_weights)] # add up each gradient for each sample in the mimi-batch
            nabla_biases = [nabla_bias_set + delta_nabla_bias_set for nabla_bias_set, delta_nabla_bias_set in zip(nabla_biases, delta_nabla_biases)]
        self.weights = [weight_set - eta * nabla_weight_set / mini_batch_size for weight_set, nabla_weight_set in zip(self.weights, nabla_weights)] # each weight in the network is adjusted by element-wise subtraction of the average gradient multiplied by the learning rate
        self.biases = [bias_set - eta * nabla_bias_set / mini_batch_size for bias_set, nabla_bias_set in zip(self.biases, nabla_biases)] # same for each bias
    
    def backpropagate(self, x, y):
        '''Returns a tuple of 2 arrays that represent the changes that need to be averaged and then applied to the weights and biases respectively. They are in the same shape as the weights and biases.'''
        nabla_weights = [None] * len(self.weights) # init blank lists for gradient of cost function
        nabla_biases = nabla_weights.copy()
        # calculate all activations in forward pass
        activation = x
        activations = [x]
        zs = []
        for weight_set, bias_set in zip(self.weights, self.biases):
            z = np.dot(weight_set, activation) + bias_set # similar to feedforward method
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # delta for the output layer. We define delta as the partial derivative of cost with respect to each "z" (the change in cost for a small change in each z)
        nabla_weights[-1] = np.dot(delta, activations[-2].transpose()) # results in the cross product of a row and column vector, creating a matrix in the dimension of the previos weight set, which is the partial derivative of the cost with respect to each weight in the set
        nabla_biases[-1] = delta # the partial derivative of cost with respect to each bias is conveniently the same as that with respect to each "z". 
        for layer in range(2, len(self.layer_sizes)): # iterate backwards from the second-last layer to the second
            z = zs[-layer]
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sigmoid_prime(z) # delta for the previous layer. Cross product that results in a column matrix in which each new delta is the weighted sum of delta after it in the next layer--all multiplied by the derivative of the sigmoid of each "z"
            nabla_weights[-layer] = np.dot(delta, activations[-layer - 1].transpose()) # same as before
            nabla_biases[-layer] = delta
        return (nabla_weights, nabla_biases)

    def cost_derivative(self, output_activations, y):
        '''The cost derivative of the quadratic cost function.'''
        return output_activations - y

    def evaluate_performance(self, test_images, test_labels):
        '''Pass all test images and see how many are correctly identified.'''
        how_many_correct = 0
        for x, y in zip(test_images, test_labels):
            how_many_correct += int(np.argmax(self.feed_forward(x)) == y) # pass the test images into the network and get the prediction for each. The decision is taken as the highest activation in the output layer. Compare these with the labels.
        return how_many_correct

    def predict(self, img_data):
        '''Take a 1-d array of image data and output prediction.'''
        img_data = img_data.reshape((28 * 28, 1)) / 255
        prediction = self.feed_forward(img_data)
        prediction = prediction / prediction.sum() # scale such that they sum to 1
        for i in range(len(prediction)):
            print("{}: {}% confidence".format(i, round(prediction[i, 0], 2) * 100))

