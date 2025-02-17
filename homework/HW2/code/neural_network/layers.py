import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Args:
      x: (np.array) containing input data, of shape (N, d_1, ..., d_k)
      w: (np.array) weights, of shape (D, M)
      b: (np.array) biases, of shape (M,)

    Returns:
      out: output, of shape (N, M)
      cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    # begin answer


    # Step 1: Reshape the input `x` to a 2D array of shape (N, D)
    N = x.shape[0]
    D = np.prod(x.shape[1:])  # Product of dimensions d_1, d_2, ..., d_k
    x_reshaped = x.reshape(N, D)
    
    # Step 2: Compute the output of the affine layer: out = x_reshaped.dot(w) + b
    out = np.dot(x_reshaped, w) + b

    # end answer
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine layer.

    Args:
      dout: Upstream derivative, of shape (N, M)
      cache: Tuple of:
        x: Input data, of shape (N, d_1, ... d_k)
        w: Weights, of shape (D, M)

    Returns:
      dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
      dw: Gradient with respect to w, of shape (D, M)
      db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    # begin answer

    # Step 1: Reshape x to (N, D)
    N = x.shape[0]
    D = np.prod(x.shape[1:])  # Product of dimensions d_1, d_2, ..., d_k
    x_reshaped = x.reshape(N, D)
    
    # Step 2: Compute db (gradient with respect to b)
    db = np.sum(dout, axis=0)
    
    # Step 3: Compute dw (gradient with respect to w)
    dw = np.dot(x_reshaped.T, dout)
    
    # Step 4: Compute dx (gradient with respect to x)
    dx = np.dot(dout, w.T)
    
    # Step 5: Reshape dx back to the original input shape (N, d_1, ..., d_k)
    dx = dx.reshape(x.shape)

    # end answer
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Args:
      x: Inputs, of any shape

    Returns:
      out: Output, of the same shape as x
      cache: x
    """
    out = None
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    # begin answer
    out = np.maximum(0, x)
    # end answer
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Args:
      dout: Upstream derivatives, of any shape
      cache: Input x, of same shape as dout

    Returns:
      dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    # begin answer
    dx = dout * (x > 0)  # Use element-wise multiplication with a boolean mask
    # end answer
    return dx


def svm_loss(x, y):
    """Computes the loss and gradient using for multiclass SVM classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Args:
      x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
      y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns:
      loss: Scalar giving the loss
      dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
