import numpy as np

def linear_regression(X, y):
    '''
    LINEAR_REGRESSION Linear Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
    '''
    
    #x=p*n y=1*n w=p+1*1 b=sum of every element in w


    P, N = X.shape
    w = np.zeros((P + 1, 1))
    # YOUR CODE HERE
    # begin answer
    newX=np.vstack([(np.ones(N)),X])
    w=np.linalg.inv(newX@newX.T)@newX@y.T
    # begin answer
    # end answer
    return w
