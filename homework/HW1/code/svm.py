import numpy as np
from scipy.optimize import minimize

def svm(X, y):
    '''
    使用 scipy.optimize 实现支持向量机（SVM）。

    输入:  
        X: 训练样本特征，P × N 矩阵。
        y: 训练样本标签，1 × N 行向量。

    输出: 
        w: 学习到的感知机参数，(P+1) × 1 列向量。
        num: 支持向量的数量
    '''
    P, N = X.shape
    X_aug = np.vstack([X, np.ones((1, N))])  # 将 X 增广，添加偏置项
    y = y.flatten()  # 确保 y 是 1D 数组
    
    # 定义目标函数（正则化项）
    def objective(w):
        return 0.5 * np.sum(w[:-1] ** 2)  # 目标函数是正则化项

    # 定义约束函数（y_i * (w^T * x_i + b) >= 1）
    def constraint(w):
        return y * (np.dot(w[:-1].T, X) + w[-1]) - 1
    
    # 初始猜测的 w（P+1 维度）
    w_init = np.zeros(P + 1)
    
    # 定义约束：不对优化变量设置具体边界
    bounds = [(None, None)] * (P + 1)  # 不对权重设置边界
    
    # 定义优化问题的约束条件
    cons = {'type': 'ineq', 'fun': constraint}
    
    # 调用 scipy.optimize.minimize 求解优化问题
    result = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=cons)
    
    # 获取优化后的权重
    w_opt = result.x
    
    # 学习到的感知机参数（w 和 b）
    w_final = w_opt[:-1].reshape(-1, 1)  # 权重
    b_final = w_opt[-1]  # 偏置

    # print("w_final: ", w_final)
    # print("b_final: ", b_final)
    
    # 计算支持向量（满足边界条件的样本）
    margins = y * (np.dot(X.T, w_final) + b_final)
    support_vectors = np.where(np.abs(margins - 1) < 1e-5)[0]  # 支持向量
    
    num_support_vectors = len(support_vectors)

    w=  np.vstack([b_final, w_final])  # 感知
    
    return w, num_support_vectors
