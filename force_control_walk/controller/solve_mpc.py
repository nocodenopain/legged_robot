import numpy as np
from scipy.linalg import expm
import quadprog  # 需要安装：pip install quadprog
from utils.util import *

def solveMpc(R_yaw, I_inv, rbf, x0, xd, dt, horizon, gait, mass):
    f_max = 140
    mu = 1/0.4
    alpha = 0.00002
    
    # 系统矩阵
    Ac = np.zeros((13, 13))
    Ac[0:3, 6:9] = R_yaw.T
    Ac[3:6, 9:12] = np.eye(3)
    Ac[11, 12] = 1
    
    I3 = np.eye(3)
    Bc = np.zeros((13, 12))
    
    # 摩擦锥约束
    f_block = np.array([
        [mu,   0, 1],
        [-mu,   0, 1],
        [0, mu, 1],
        [0, -mu, 1],
        [0,   0, 1]
    ])
    
    # 构建约束矩阵
    A = np.zeros((20*horizon, 12*horizon))
    for i in range(4*horizon):
        A[5*i:5*(i+1), 3*i:3*(i+1)] = f_block
    
    # 构建输入矩阵
    for i in range(4):
        print(rbf[:, i].shape)
        Bc[6:9, 3*i:3*(i+1)] = I_inv @ Skew(rbf[:, i])
        Bc[9:12, 3*i:3*(i+1)] = I3 / mass
    
    # 离散化系统
    Aqp, Bqp = c2qp(Ac, Bc, dt, horizon)
    
    # 权重矩阵
    full_weight = np.array([25, 25, 10, 2, 2, 100, 0, 0, 0.3, 10, 10, 20, 0])
    S = np.diag(np.tile(full_weight, horizon))
    
    # 构建QP问题
    H = 2 * (Bqp.T @ S @ Bqp + alpha * np.eye(12*horizon))
    g = 2 * Bqp.T @ S @ (Aqp @ x0 - xd)
    
    # 构建约束条件
    lbA = np.zeros(20*horizon)
    ubA = np.zeros(20*horizon)
    k = 0
    for i in range(horizon):
        for j in range(4):
            ubA[5*k + 4] = gait[4*i + j] * f_max  # 只有第5个约束有上限
            k += 1
    
    # 转换为quadprog需要的格式 (Gx <= h)
    G = np.vstack([A, -A])
    h = np.hstack([ubA, -lbA])  # lbA <= Ax <= ubA → Ax <= ubA AND -Ax <= -lbA
    
    # 调用quadprog求解
    x, _, _ = quadprog.solve_qp(H, g, A, G.T, h, 0)  # 注意quadprog需要G.T
    
    return x

def c2qp(A, B, dt, horizon):
    """离散化系统"""
    ABc = np.zeros((25, 25))
    ABc[0:13, 0:13] = A
    ABc[0:13, 13:25] = B
    ABc = dt * ABc
    expmm = expm(ABc)
    
    Adt = expmm[0:13, 0:13]
    Bdt = expmm[0:13, 13:25]
    
    powerMats = np.zeros((13, 13, horizon+1))
    powerMats[:, :, 0] = np.eye(13)
    
    for i in range(horizon):
        powerMats[:, :, i+1] = Adt @ powerMats[:, :, i]
    
    A_qp = np.zeros((13*horizon, 13))
    B_qp = np.zeros((13*horizon, 12*horizon))
    
    for m in range(horizon):
        A_qp[13*m:13*(m+1), :] = powerMats[:, :, m+1]
        for n in range(horizon):
            if m >= n:
                a_num = m - n
                B_qp[13*m:13*(m+1), 12*n:12*(n+1)] = powerMats[:, :, a_num+1] @ Bdt
    
    return A_qp, B_qp