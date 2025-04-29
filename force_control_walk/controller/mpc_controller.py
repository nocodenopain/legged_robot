import numpy as np
from scipy.linalg import expm, logm
from controller.solve_mpc import solveMpc
from controller.swing_trajectory import SwingTrajectoryBezier
from utils.util import *
from controller.qp import *

class MpcController:
    def __init__(self, horizon):
        # 初始化所有持久性变量
        self.wf = np.zeros(3)
        self.x_s = np.zeros(13)
        self.x_desire = np.zeros(13 * horizon)
        self.f = np.zeros(12 * horizon)
        self.state = 0
        self.timer = 0
        self.iterationcounter = 0
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0
        self.v0 = 0
        self.firstswing = np.array([0, 0, 0, 0])
        self.pf_init = np.zeros((3, 4))
        self.pf_final = np.zeros((3, 4))
        self.swingTimeRemaining = np.zeros(4)
        
    def compute_control(self, R, w, x, y, z, v, q1, q2, q3, q4, w1, w2, w3, w4, t, 
                       foot_senser, vx, vy, offsets, durations, bodyheight, d_roll, 
                       d_pitch, d_yaw, rbs, mass, I, offset, Kpcom, Kdcom, Kpbase, 
                       Kdbase, dt, IterationsBetweenMpc, stancetime, swingtime, 
                       height, horizon, Kp_cartesian, Kd_cartesian, vz, nIterations):
        
        # 初始化局部变量
        trajAll = np.zeros(12 * horizon)
        tao = np.zeros((3, 4))
        Fex = np.zeros((3, 4))
        p = np.zeros((3, 4))
        pv = np.zeros((3, 4))
        pa = np.zeros((3, 4))
        rsf_des = np.zeros((3, 4))
        vleg_des = np.zeros((3, 4))
        v_leg = np.zeros((3, 4))
        w_leg = np.array([w1, w2, w3, w4])
        
        dtMpc = dt * IterationsBetweenMpc
        offsetsFloat = offsets / nIterations
        durationsFloat = durations / nIterations
        
        # 计算姿态角
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
        yaw = np.arctan2(R[1, 0], R[0, 0])
        
        R_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        I_world = R @ I @ R.T
        I_inv = np.linalg.inv(I_world)
        
        # 设置参考速度和位置
        if t > 3 and t < 5:
            v_ref = np.array([0, 0, vz])
            r_ref = np.array([0, 0, 0.1 + vz * (t - 3)])
        else:
            if t == 5:
                self.x0, self.y0, self.z0 = x, y, z
            v_ref = np.array([vx, vy, 0])
            r_ref = np.array([self.x0, self.y0, bodyheight])
        
        q_ref = np.array([d_roll, d_pitch, d_yaw])
        w_ref = np.array([0, 0, 0])
        r = np.array([x, y, z])
        
        # 计算控制输入
        a = Kpcom * (r_ref - r) + Kdcom * (v_ref - v)
        wd = np.array([0, 0, 0])
        qw = matrixLogRot(R.T)
        aw = Kpbase * qw + Kdbase * (wd - w)
        F = mass * (a + np.array([0, 0, 9.81]))
        Tao = I_world @ aw
        b_control = np.concatenate([F, Tao])
        
        q = np.array([q1, q2, q3, q4])
        
        # 正向运动学
        rsf_body, rbf_body = forwardKinematics(q)
        rbf_world = R @ rbf_body
        J = JacobianMatrix(q)
        
        if self.state == 0:
            flag = np.array([1, 1, 1, 1])
            self.timer += 1
            
            # QP优化
            self.f = qp(rbf_world, b_control, flag)
            
            for i in range(4):
                tao[:, i] = -J[:, i*3-2:i*3+1].T @ R.T @ self.f[i*3-2:i*3+1]
            
            if self.timer > 1000:  # 站立过程(2s除以dt)
                self.state = 1
                self.timer = 0
                self.firstswing = np.array([0, 1, 1, 0])
        
        elif self.state == 1:
            for i in range(4):
                if self.firstswing[i] == 1:
                    self.swingTimeRemaining[i] = swingtime
                    self.pf_init[:, i] = np.array([x, y, z]) + rbf_world[:, i]
                    self.pf_init[2, i] = 0.015
                    self.wf = 0.5 * stancetime * np.array([v[0], v[1], 0])
                    if self.wf[0] > 0.35:
                        self.wf[0] = 0.35
                else:
                    self.swingTimeRemaining[i] -= dt
                
                if self.swingTimeRemaining[i] < 0:
                    self.swingTimeRemaining[i] = 0
                
                self.pf_final[:, i] = (np.array([x, y, z]) + v_ref * self.swingTimeRemaining[i] + 
                                      R @ offset[:, i] + self.wf)
                self.pf_final[2, i] = 0.015
            
            iteration, phase = setIterations(nIterations, self.iterationcounter, IterationsBetweenMpc)
            swingstate = getSwingState(phase, offsetsFloat, durationsFloat)
            mpctable = getMpcTable(iteration, nIterations, offsets, durations)
            
            if self.iterationcounter % IterationsBetweenMpc == 0:
                for i in range(horizon):
                    trajinit = np.concatenate([q_ref, [x, y, bodyheight], w_ref, v_ref])
                    for j in range(12):
                        trajAll[12*i + j] = trajinit[j]
                        if i == 0:
                            trajAll[12*i + 3] = trajinit[3] + dtMpc * v_ref[0]
                            trajAll[12*i + 4] = trajinit[4] + dtMpc * v_ref[1]
                            trajAll[12*i + 5] = trajinit[5] + dtMpc * v_ref[2]
                        else:
                            trajAll[12*i + 3] = trajAll[12*(i-1) + 3] + dtMpc * v_ref[0]
                            trajAll[12*i + 4] = trajAll[12*(i-1) + 4] + dtMpc * v_ref[1]
                            trajAll[12*i + 5] = trajAll[12*(i-1) + 5] + dtMpc * v_ref[2]
                
                self.x_s = np.concatenate([
                    [roll, pitch, yaw],
                    [x, y, z],
                    w, v,
                    [-9.8]
                ])
                
                for i in range(horizon):
                    for j in range(12):
                        self.x_desire[13*i + j] = trajAll[12*i + j]
                    self.x_desire[13*i + 12] = -9.8
                
                self.f = solveMpc(R_yaw, I_inv, rbf_world, self.x_s, self.x_desire, 
                                 dtMpc, horizon, mpctable, mass)
            
            self.iterationcounter += 1
            
            for i in range(4):
                if swingstate[i] > 0:
                    if self.firstswing[i] == 1:
                        self.firstswing[i] = 0
                    
                    p[:, i], pv[:, i], pa[:, i] = SwingTrajectoryBezier(
                        self.pf_init[:, i], self.pf_final[:, i], swingstate[i], swingtime, height
                    )
                    
                    rsf_des[:, i] = R.T @ (p[:, i] - np.array([x, y, z])) - rbs[:, i]
                    vleg_des[:, i] = R.T @ (pv[:, i] - v)
                    v_leg[:, i] = J[:, i*3-2:i*3+1] @ w_leg[:, i]
                    
                    Fex[:, i] = (
                        Kp_cartesian * (rsf_des[:, i] - rsf_body[:, i]) +
                        Kd_cartesian * (vleg_des[:, i] - v_leg[:, i])
                    )
                    tao[:, i] = J[:, i*3-2:i*3+1].T @ Fex[:, i]
                
                elif foot_senser[i] == 1:
                    self.firstswing[i] = 1
                    tao[:, i] = -J[:, i*3-2:i*3+1].T @ R.T @ self.f[i*3-2:i*3+1]
                else:
                    self.firstswing[i] = 1
                    tao[:, i] = -J[:, i*3-2:i*3+1].T @ R.T @ self.f[i*3-2:i*3+1]
        
        return tao[:, 0], tao[:, 1], tao[:, 2], tao[:, 3]