import numpy as np

def quat_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def rotmat_to_euler(R):
    roll = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    yaw = np.arctan2(R[1,0], R[0,0])
    return np.array([roll, pitch, yaw])

def JacobianMatrix(q):
    # J1 matrix
    J1 = np.array([
        [0, 0.0819823 * np.cos(q[1]) - 0.192981 * np.sin(q[1] + q[2]) - 0.00268261 * np.cos(q[1] + q[2]) + 0.175826 * np.sin(q[1]),
         - 0.00268261 * np.cos(q[1] + q[2]) - 0.192981 * np.sin(q[1] + q[2])],
        [0.085 * np.sin(q[0]) + 0.0819823 * np.cos(q[0]) * np.cos(q[1]) + 0.175826 * np.cos(q[0]) * np.sin(q[1]) 
         - 0.00268261 * np.cos(q[1] + q[2]) * np.cos(q[0]) - 0.192981 * np.sin(q[1] + q[2]) * np.cos(q[0]),
         0.175826 * np.cos(q[1]) * np.sin(q[0]) - 0.0819823 * np.sin(q[0]) * np.sin(q[1]) - 0.192981 * np.cos(q[1] + q[2]) * np.sin(q[0]) 
         + 0.00268261 * np.sin(q[1] + q[2]) * np.sin(q[0]), 
         0.00268261 * np.sin(q[1] + q[2]) * np.sin(q[0]) - 0.192981 * np.cos(q[1] + q[2]) * np.sin(q[0])],
        [0.0819823 * np.cos(q[1]) * np.sin(q[0]) - 0.085 * np.cos(q[0]) + 0.175826 * np.sin(q[0]) * np.sin(q[1]) 
         - 0.00268261 * np.cos(q[1] + q[2]) * np.sin(q[0]) - 0.192981 * np.sin(q[1] + q[2]) * np.sin(q[0]),
         0.0819823 * np.cos(q[0]) * np.sin(q[1]) - 0.175826 * np.cos(q[0]) * np.cos(q[1]) + 0.192981 * np.cos(q[1] + q[2]) * np.cos(q[0]) 
         - 0.00268261 * np.sin(q[1] + q[2]) * np.cos(q[0]),
         0.192981 * np.cos(q[1] + q[2]) * np.cos(q[0]) - 0.00268261 * np.sin(q[1] + q[2]) * np.cos(q[0])]
    ])
    
    # J2 matrix
    J2 = np.array([
        [0, 0.0819823 * np.cos(q[4]) - 0.192981 * np.sin(q[4] + q[5]) - 0.00268261 * np.cos(q[4] + q[5]) + 0.175826 * np.sin(q[4]),
         - 0.00268261 * np.cos(q[4] + q[5]) - 0.192981 * np.sin(q[4] + q[5])],
        [0.0819823 * np.cos(q[3]) * np.cos(q[4]) - 0.085 * np.sin(q[3]) + 0.175826 * np.cos(q[3]) * np.sin(q[4]) 
         - 0.00268261 * np.cos(q[4] + q[5]) * np.cos(q[3]) - 0.192981 * np.sin(q[4] + q[5]) * np.cos(q[3]),
         0.175826 * np.cos(q[4]) * np.sin(q[3]) - 0.0819823 * np.sin(q[3]) * np.sin(q[4]) - 0.192981 * np.cos(q[4] + q[5]) * np.sin(q[3]) 
         + 0.00268261 * np.sin(q[4] + q[5]) * np.sin(q[3]), 
         0.00268261 * np.sin(q[4] + q[5]) * np.sin(q[3]) - 0.192981 * np.cos(q[4] + q[5]) * np.sin(q[3])],
        [0.085 * np.cos(q[3]) + 0.0819823 * np.cos(q[4]) * np.sin(q[3]) + 0.175826 * np.sin(q[3]) * np.sin(q[4]) 
         - 0.00268261 * np.cos(q[4] + q[5]) * np.sin(q[3]) - 0.192981 * np.sin(q[4] + q[5]) * np.sin(q[3]),
         0.0819823 * np.cos(q[3]) * np.sin(q[4]) - 0.175826 * np.cos(q[3]) * np.cos(q[4]) + 0.192981 * np.cos(q[4] + q[5]) * np.cos(q[3]) 
         - 0.00268261 * np.sin(q[4] + q[5]) * np.cos(q[3]),
         0.192981 * np.cos(q[4] + q[5]) * np.cos(q[3]) - 0.00268261 * np.sin(q[4] + q[5]) * np.cos(q[3])]
    ])

    # J3 matrix
    J3 = np.array([
        [0, 0.0819823 * np.cos(q[7]) - 0.192981 * np.sin(q[7] + q[8]) - 0.00268261 * np.cos(q[7] + q[8]) + 0.175826 * np.sin(q[7]),
         - 0.00268261 * np.cos(q[7] + q[8]) - 0.192981 * np.sin(q[7] + q[8])],
        [0.085 * np.sin(q[6]) + 0.0819823 * np.cos(q[6]) * np.cos(q[7]) + 0.175826 * np.cos(q[6]) * np.sin(q[7]) 
         - 0.00268261 * np.cos(q[7] + q[8]) * np.cos(q[6]) - 0.192981 * np.sin(q[7] + q[8]) * np.cos(q[6]),
         0.175826 * np.cos(q[7]) * np.sin(q[6]) - 0.0819823 * np.sin(q[6]) * np.sin(q[7]) - 0.192981 * np.cos(q[7] + q[8]) * np.sin(q[6]) 
         + 0.00268261 * np.sin(q[7] + q[8]) * np.sin(q[6]), 
         0.00268261 * np.sin(q[7] + q[8]) * np.sin(q[6]) - 0.192981 * np.cos(q[7] + q[8]) * np.sin(q[6])],
        [0.0819823 * np.cos(q[7]) * np.sin(q[6]) - 0.085 * np.cos(q[6]) + 0.175826 * np.sin(q[6]) * np.sin(q[7]) 
         - 0.00268261 * np.cos(q[7] + q[8]) * np.sin(q[6]) - 0.192981 * np.sin(q[7] + q[8]) * np.sin(q[6]),
         0.0819823 * np.cos(q[6]) * np.sin(q[7]) - 0.175826 * np.cos(q[6]) * np.cos(q[7]) + 0.192981 * np.cos(q[7] + q[8]) * np.cos(q[6]) 
         - 0.00268261 * np.sin(q[7] + q[8]) * np.cos(q[6]),
         0.192981 * np.cos(q[7] + q[8]) * np.cos(q[6]) - 0.00268261 * np.sin(q[7] + q[8]) * np.cos(q[6])]
    ])

    # J4 matrix
    J4 = np.array([
        [0, 0.0819823 * np.cos(q[10]) - 0.192981 * np.sin(q[10] + q[11]) - 0.00268261 * np.cos(q[10] + q[11]) + 0.175826 * np.sin(q[10]),
         - 0.00268261 * np.cos(q[10] + q[11]) - 0.192981 * np.sin(q[10] + q[11])],
        [0.0819823 * np.cos(q[9]) * np.cos(q[10]) - 0.085 * np.sin(q[9]) + 0.175826 * np.cos(q[9]) * np.sin(q[10]) 
         - 0.00268261 * np.cos(q[10] + q[11]) * np.cos(q[9]) - 0.192981 * np.sin(q[10] + q[11]) * np.cos(q[9]),
         0.175826 * np.cos(q[10]) * np.sin(q[9]) - 0.0819823 * np.sin(q[9]) * np.sin(q[10]) - 0.192981 * np.cos(q[10] + q[11]) * np.sin(q[9]) 
         + 0.00268261 * np.sin(q[10] + q[11]) * np.sin(q[9]), 
         0.00268261 * np.sin(q[10] + q[11]) * np.sin(q[9]) - 0.192981 * np.cos(q[10] + q[11]) * np.sin(q[9])],
        [0.085 * np.cos(q[9]) + 0.0819823 * np.cos(q[10]) * np.sin(q[9]) + 0.175826 * np.sin(q[9]) * np.sin(q[10]) 
         - 0.00268261 * np.cos(q[10] + q[11]) * np.sin(q[9]) - 0.192981 * np.sin(q[10] + q[11]) * np.sin(q[9]),
         0.0819823 * np.cos(q[9]) * np.sin(q[10]) - 0.175826 * np.cos(q[9]) * np.cos(q[10]) + 0.192981 * np.cos(q[10] + q[11]) * np.cos(q[9]) 
         - 0.00268261 * np.sin(q[10] + q[11]) * np.cos(q[9]),
         0.192981 * np.cos(q[10] + q[11]) * np.cos(q[9]) - 0.00268261 * np.sin(q[10] + q[11]) * np.cos(q[9])]
    ])

    # Concatenate the four matrices
    J = np.hstack([J1, J2, J3, J4])

    return J

def Skew(r):
    r = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])
    return r

def forwardKinematics(q):
    # Calculating rsf (frame 1 positions)
    rsf_x1 = 0.082 * np.sin(q[1]) - 0.175826 * np.cos(q[1]) + 0.192981 * np.cos(q[1] + q[2]) - 0.00268261 * np.sin(q[1] + q[2]) + 0.065
    rsf_y1 = 0.082 * np.cos(q[1]) * np.sin(q[0]) - 0.085 * np.cos(q[0]) + 0.175826 * np.sin(q[0]) * np.sin(q[1]) - 0.00268261 * np.cos(q[1] + q[2]) * np.sin(q[0]) - 0.192981 * np.sin(q[0]) * np.sin(q[1] + q[2])
    rsf_z1 = 0.00268261 * np.cos(q[0]) * np.cos(q[1] + q[2]) - 0.082 * np.cos(q[0]) * np.cos(q[1]) - 0.175826 * np.cos(q[0]) * np.sin(q[1]) - 0.085 * np.sin(q[0]) + 0.192981 * np.cos(q[0]) * np.sin(q[1] + q[2])

    # Calculating rsf (frame 2 positions)
    rsf_x2 = 0.082 * np.sin(q[4]) - 0.175826 * np.cos(q[4]) + 0.192981 * np.cos(q[4] + q[5]) - 0.00268261 * np.sin(q[4] + q[5]) + 0.065
    rsf_y2 = 0.085 * np.cos(q[3]) + 0.082 * np.cos(q[4]) * np.sin(q[3]) + 0.175826 * np.sin(q[3]) * np.sin(q[4]) - 0.00268261 * np.cos(q[4] + q[5]) * np.sin(q[3]) - 0.192981 * np.sin(q[3]) * np.sin(q[4] + q[5])
    rsf_z2 = 0.085 * np.sin(q[3]) - 0.082 * np.cos(q[3]) * np.cos(q[4]) - 0.175826 * np.cos(q[3]) * np.sin(q[4]) + 0.00268261 * np.cos(q[3]) * np.cos(q[4] + q[5]) + 0.192981 * np.cos(q[3]) * np.sin(q[4] + q[5])

    # Calculating rsf (frame 3 positions)
    rsf_x3 = 0.082 * np.sin(q[7]) - 0.175826 * np.cos(q[7]) + 0.192981 * np.cos(q[7] + q[8]) - 0.00268261 * np.sin(q[7] + q[8]) - 0.065
    rsf_y3 = 0.082 * np.cos(q[7]) * np.sin(q[6]) - 0.085 * np.cos(q[6]) + 0.175826 * np.sin(q[6]) * np.sin(q[7]) - 0.00268261 * np.cos(q[7] + q[8]) * np.sin(q[6]) - 0.192981 * np.sin(q[6]) * np.sin(q[7] + q[8])
    rsf_z3 = 0.00268261 * np.cos(q[6]) * np.cos(q[7] + q[8]) - 0.082 * np.cos(q[6]) * np.cos(q[7]) - 0.175826 * np.cos(q[6]) * np.sin(q[7]) - 0.085 * np.sin(q[6]) + 0.192981 * np.cos(q[6]) * np.sin(q[7] + q[8])

    # Calculating rsf (frame 4 positions)
    rsf_x4 = 0.082 * np.sin(q[10]) - 0.175826 * np.cos(q[10]) + 0.192981 * np.cos(q[10] + q[11]) - 0.00268261 * np.sin(q[10] + q[11]) - 0.065
    rsf_y4 = 0.085 * np.cos(q[9]) + 0.082 * np.cos(q[10]) * np.sin(q[9]) + 0.175826 * np.sin(q[9]) * np.sin(q[10]) - 0.00268261 * np.cos(q[10] + q[11]) * np.sin(q[9]) - 0.192981 * np.sin(q[9]) * np.sin(q[10] + q[11])
    rsf_z4 = 0.085 * np.sin(q[9]) - 0.082 * np.cos(q[9]) * np.cos(q[10]) - 0.175826 * np.cos(q[9]) * np.sin(q[10]) + 0.00268261 * np.cos(q[9]) * np.cos(q[10] + q[11]) + 0.192981 * np.cos(q[9]) * np.sin(q[10] + q[11])

    # Assembling rsf matrix
    rsf = np.array([[rsf_x1, rsf_x2, rsf_x3, rsf_x4],
                    [rsf_y1, rsf_y2, rsf_y3, rsf_y4],
                    [rsf_z1, rsf_z2, rsf_z3, rsf_z4]])

    # Calculating rbf matrix
    rbf = np.array([[rsf_x1 + 0.139, rsf_x2 + 0.139, rsf_x3 - 0.139, rsf_x4 - 0.139],
                    [rsf_y1 - 0.061, rsf_y2 + 0.061, rsf_y3 - 0.061, rsf_y4 + 0.061],
                    [rsf_z1, rsf_z2, rsf_z3, rsf_z4]])

    return rsf, rbf

def getMpcTable(iteration, nIterations, offsets, durations):
    # 创建一个全为零的数组
    mpctable = np.zeros(4 * nIterations)
    
    for i in range(1, nIterations + 1):
        # 计算迭代器
        iter = (i + iteration - 1) % nIterations
        progress = iter * np.array([1, 1, 1, 1]) - offsets
        
        for j in range(4):
            # 如果 progress(j) < 0, 则修正为正值
            if progress[j] < 0:
                progress[j] += nIterations
                
            # 判断是否小于持续时间
            if progress[j] < durations[j]:
                mpctable[(i - 1) * 4 + j] = 1
            else:
                mpctable[(i - 1) * 4 + j] = 0
    
    return mpctable

def getSwingState(phase, offsetsFloat, durationsFloat):
    # 计算偏移量
    swing_offset = offsetsFloat + durationsFloat
    
    # 如果偏移量大于 1，则减去 1
    for i in range(4):
        if swing_offset[i] > 1:
            swing_offset[i] -= 1
    
    # 计算摆动持续时间
    swing_duration = np.array([1, 1, 1, 1]) - durationsFloat
    
    # 计算摆动状态
    swingstate = phase - swing_offset

    for i in range(4):
        # 如果摆动状态小于 0，则加 1
        if swingstate[i] < 0:
            swingstate[i] += 1
        
        # 如果摆动状态大于持续时间，则归零
        if swingstate[i] > swing_duration[i]:
            swingstate[i] = 0
        else:
            # 否则，归一化摆动状态
            swingstate[i] /= swing_duration[i]
    
    return swingstate

def matrixLogRot(R):
    tmp = (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2
    if tmp >= 1:
        theta = 0
    elif tmp <= -1:
        theta = np.pi  # 3.1415926
    else:
        theta = np.arccos(tmp)
    
    omega = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    
    if theta > 1e-5:
        omega = omega * theta / (2 * np.sin(theta))
    else:
        omega = omega / 2
    
    return omega

def setIterations(n_iterations, current_iteration, iterations_between_mpc):
    iteration = (current_iteration // iterations_between_mpc) % n_iterations
    phase = (current_iteration % (iterations_between_mpc * n_iterations)) / (iterations_between_mpc * n_iterations)
    return iteration, phase
