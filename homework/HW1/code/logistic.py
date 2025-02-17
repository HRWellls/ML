import numpy as np

def logistic(X, y):
    '''
    LR Logistic Regression using Gradient Descent.

    INPUT:  
        X: training sample features, P-by-N matrix.
        y: training sample labels, 1-by-N row vector.
        learning_rate: step size for gradient descent.
        num_iterations: number of iterations for gradient descent.

    OUTPUT: 
        w: learned parameters, (P+1)-by-1 column vector.
    '''
    learning_rate=0.1
    num_iterations=1000
    y = (y + 1) // 2  # 将 -1 转换为 0，1 保持不变
    P, N = X.shape
    # Add bias term
    X_bias = np.vstack([np.ones((1, N)), X])  # Adding a row of ones for bias term
    
    # Initialize parameters w
    w = np.zeros((P + 1, 1))
    
    # Gradient Descent
    for i in range(num_iterations):
        # Calculate predictions
        z = np.dot(w.T, X_bias)
        predictions = 1 / (1 + np.exp(-z))  # Sigmoid function
        
        # Gradient of the log-likelihood
        gradient = np.dot(X_bias, (predictions - y).T) / N
        
        # Update weights
        w -= learning_rate * gradient
    
    return w
