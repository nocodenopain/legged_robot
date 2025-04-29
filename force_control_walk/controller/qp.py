import numpy as np
from qpsolvers import solve_qp

def Skew(v):
    """返回3D向量的斜对称矩阵"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def qp(rbf, b_control, flag):
    uf = 0.5
    n = 12  # 变量个数
    m = 20  # 约束条件个数
    
    # 初始化边界和约束条件
    ub = np.zeros(n)  # 变量上界
    lb = np.zeros(n)  # 变量下界
    ubA = np.array([0, 1e6, 0, 1e6, 160, 0, 1e6, 0, 1e6, 160, 
                    0, 1e6, 0, 1e6, 160, 0, 1e6, 0, 1e6, 160])  # 约束上界
    lbA = np.array([0, 1e6, 0, 1e6, 10, 0, 1e6, 0, 1e6, 10, 
                    0, 1e6, 0, 1e6, 10, 0, 1e6, 0, 1e6, 10])  # 约束下界
    
    # 构建A_control矩阵
    A_control = np.block([
        [flag[0] * np.eye(3), flag[1] * np.eye(3), flag[2] * np.eye(3), flag[3] * np.eye(3)],
        [flag[0] * Skew(rbf[:,0]), flag[1] * Skew(rbf[:,1]), flag[2] * Skew(rbf[:,2]), flag[3] * Skew(rbf[:,3])]
    ])
    
    # 构建Hessian矩阵和梯度向量
    S = np.diag([1, 1, 10, 50, 30, 10])  # 权重矩阵
    W = 0.001 * np.eye(12)  # 正则化矩阵
    alpha = 0.01  # 正则化系数
    
    H = 2 * A_control.T @ S @ A_control + 2 * alpha * W  # Hessian矩阵
    g = -2 * A_control.T @ S @ b_control  # 梯度向量
    
    # 构建约束矩阵C
    c = np.array([
        [1, 0, -uf],  # 约束条件1
        [1, 0, uf],   # 约束条件2
        [0, 1, -uf],  # 约束条件3
        [0, 1, uf],   # 约束条件4
        [0, 0, 1]     # 约束条件5
    ])
    O = np.zeros((5, 3))  # 零矩阵块
    
    # 完整的约束矩阵A
    A = np.block([
        [flag[0] * c, O, O, O],
        [O, flag[1] * c, O, O],
        [O, O, flag[2] * c, O],
        [O, O, O, flag[3] * c]
    ])
    
    # 根据flag更新边界条件
    for i in range(4):
        # 更新变量边界
        for j in range(3):
            ub[3 * i + j] = flag[i] * 100000
            lb[3 * i + j] = -flag[i] * 100000
        
        # 更新约束下界
        lbA[5 * i] = -flag[i] * 100000
        lbA[5 * i + 1] = 0
        lbA[5 * i + 2] = -flag[i] * 100000
        lbA[5 * i + 3] = 0
        lbA[5 * i + 4] = flag[i] * 10
        
        # 更新约束上界
        ubA[5 * i] = 0
        ubA[5 * i + 1] = flag[i] * 100000
        ubA[5 * i + 2] = 0
        ubA[5 * i + 3] = flag[i] * 100000
        ubA[5 * i + 4] = flag[i] * 160
    
    # 求解二次规划问题
    x = solve_qp(
        P=H,          # Hessian矩阵
        q=g,          # 梯度向量
        G=None,       # 不使用线性不等式约束
        h=None,       # 不使用线性不等式约束
        A=None,       # 没有线性等式约束
        b=None,       # 没有等式约束右侧向量
        lb=lb,        # 变量下界
        ub=ub,        # 变量上界
        # lbA=lbA,      # 约束下界
        # ubA=ubA,      # 约束上界
        solver='osqp'  # 指定使用qpOASES求解器
    )
    
    return x

# rbf = np.array([
#         [1.0, 0.0, 0.0],  # 第1个向量
#         [0.0, 1.0, 0.0],  # 第2个向量
#         [0.0, 0.0, 1.0],  # 第3个向量
#         [1.0, 1.0, 1.0]   # 第4个向量
#     ]).T  # 转置为3x4矩阵
    
# # 创建b_control向量 (6D向量)
# b_control = np.array([10, 20, 30, 40, 50, 60])
    
# # 设置flag (控制哪些部分激活)
# flag = np.array([1, 1, 1, 1])  # 全部激活

# x = qp(rbf, b_control, flag)
# print("QP解:", x)  # 打印QP解