import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))
    iters = 0
    max_iters = 1000
    learning_rate = 0.1
    X = np.vstack((np.ones((1, N)), X))
    # YOUR CODE HERE
    while iters < max_iters:
        flag = False
        iters += 1
        for i in range(N):
            if y[0,i] * np.dot(w.T, X[:, i].reshape(-1,1)) <= 0:
                w = w + learning_rate * y[0,i] * X[:, i].reshape(-1, 1)
                flag = True
        if not flag:
            break

    # begin answer
    # end answer
    
    return w, iters


