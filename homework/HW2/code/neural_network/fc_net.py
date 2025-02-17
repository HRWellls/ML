import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=1 * 28 * 28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Args:
          input_dim: An integer giving the size of the input
          hidden_dim: An integer giving the size of the hidden layer
          num_classes: An integer giving the number of classes to classify
          weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
          reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        # begin answer
        # Initialize weights and biases
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)
        # end answer

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Args:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # begin answer
        # Flatten the input X from (N, 1, 28, 28) to (N, 784)
        N=X.shape[0]
        X = X.reshape(X.shape[0], -1)  # Flatten the images to (N, 784)
        
        # Forward pass
        # First layer
        hidden_layer = X.dot(self.params['W1']) + self.params['b1']
        hidden_layer = np.maximum(0, hidden_layer)  # ReLU activation

        # Second layer (output scores)
        scores = hidden_layer.dot(self.params['W2']) + self.params['b2']

        if y is None:
            # If we are in test-time, return the scores only
            return scores

        # Compute the loss (softmax + regularization)
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)  # For numerical stability
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # Softmax probabilities

        correct_logprobs = -np.log(probs[range(N), y])  # Log loss for correct class
        loss = np.sum(correct_logprobs) / N  # Average loss
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))  # Add L2 regularization

        # Backward pass (gradient computation)
        grads = {}

        # Gradient of loss w.r.t. scores
        dscores = probs
        dscores[range(N), y] -= 1
        dscores /= N

        # Backprop to second layer
        grads['W2'] = hidden_layer.T.dot(dscores)  # Gradient of W2
        grads['b2'] = np.sum(dscores, axis=0)  # Gradient of b2

        # Backprop to first layer
        dhidden = dscores.dot(self.params['W2'].T)
        dhidden[hidden_layer <= 0] = 0  # ReLU backpropagation

        grads['W1'] = X.T.dot(dhidden)  # Gradient of W1
        grads['b1'] = np.sum(dhidden, axis=0)  # Gradient of b1

        # Add regularization to gradients
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {affine - relu} x (L - 1) - affine - softmax

    where the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=1 * 28 * 28, num_classes=10,
                 reg=0.0, weight_scale=1e-2,
                 dtype=np.float32):
        """
        Initialize a new FullyConnectedNet.

        Args:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        ############################################################################
        # begin answer

        # Create a list of layer sizes
        layer_sizes = [input_dim] + hidden_dims + [num_classes]

        # Initialize weights and biases for each layer
        for i in range(self.num_layers):
            self.params[f'W{i + 1}'] = np.random.normal(0, weight_scale, (layer_sizes[i], layer_sizes[i + 1]))
            self.params[f'b{i + 1}'] = np.zeros(layer_sizes[i + 1])

        # end answer

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        ############################################################################
        # begin answer

        # Forward pass
        caches = {}
        out = X

        for i in range(self.num_layers - 1):
            W, b = self.params[f'W{i + 1}'], self.params[f'b{i + 1}']
            out, cache = affine_relu_forward(out, W, b)
            caches[f'cache{i + 1}'] = cache
        
        # Last layer: affine - softmax
        W, b = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, cache = affine_forward(out, W, b)
        caches[f'cache{self.num_layers}'] = cache

        # end answer

        # If test mode return early
        if y is None:
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # begin answer

        # Compute softmax loss and gradient
        data_loss, dscores = softmax_loss(scores, y)
        loss = data_loss

        # Add L2 regularization loss
        for i in range(self.num_layers):
            W = self.params[f'W{i + 1}']
            loss += 0.5 * self.reg * np.sum(W * W)

        # Backward pass
        dout, dW, db = affine_backward(dscores, caches[f'cache{self.num_layers}'])
        grads[f'W{self.num_layers}'] = dW + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db

        for i in range(self.num_layers - 2, -1, -1):
            dout, dW, db = affine_relu_backward(dout, caches[f'cache{i + 1}'])
            grads[f'W{i + 1}'] = dW + self.reg * self.params[f'W{i + 1}']
            grads[f'b{i + 1}'] = db


        # end answer
        return loss, grads



    def affine_forward(x, w, b):
        out = np.dot(x, w) + b
        cache = (x, w, b)
        return out, cache

    def affine_backward(dout, cache):
        x, w, b = cache
        dx = np.dot(dout, w.T)
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        return dx, dw, db

    def relu_forward(x):
        out = np.maximum(0, x)
        cache = x
        return out, cache

    def relu_backward(dout, cache):
        x = cache
        dx = dout * (x > 0)
        return dx

    def softmax_loss(scores, y):
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        N = scores.shape[0]
        loss = -np.sum(np.log(probs[np.arange(N), y])) / N
        
        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N
        
        return loss, dscores

