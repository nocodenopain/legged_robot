import mujoco
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from numpy import cos, sin

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0
        self.previous_error = 0

    def compute(self, error, velocity):
        self.integral += error * self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * (-velocity)
        self.previous_error = error
        return output

class GaitParams:
    def __init__(self):
        self.amplitude = 0.3
        self.frequency = 1
        self.phase_offset = np.pi / 2

# 加载模型
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 获取关节索引
joint_indices = [model.joint(name).id for name in [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
]]

actuator_indices = [model.actuator(name).id for name in [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf"
]]

foot_indices = [model.site(name).id for name in [
    "FR_foot_site", "FL_foot_site", "RR_foot_site", "RL_foot_site"
]]

joint_site_indices = [model.site(name).id for name in [
    "FR_hip_site", "FL_hip_site", "RR_hip_site", "RL_hip_site"
]]

def foot_trajectory(phase_time, swing_time, stance_time, step_height, step_length):
    """
    输入：
        phase_time: 当前相位时间（秒）
        swing_time: 腾空阶段总时间
        stance_time: 支撑阶段总时间
        step_height: 最大抬脚高度
        step_length: 一步前进的距离
    输出：
        相对于身体坐标系下的足端位置 [x, z]
    """
    total_time = swing_time + stance_time
    phase = phase_time / total_time  # 归一化相位 [0, 1)

    if phase <= swing_time / total_time:
        # === 腾空阶段（贝塞尔曲线） ===
        swing_phase = phase / (swing_time / total_time)

        P0 = np.array([0.0, 0.0])                  # 起点
        P1 = np.array([0.25 * step_length, step_height])
        P2 = np.array([0.75 * step_length, step_height])
        P3 = np.array([step_length, 0.0])          # 终点

        t = swing_phase
        foot_xy = (1 - t)**3 * P0 + 3*(1 - t)**2*t * P1 + 3*(1 - t)*t**2 * P2 + t**3 * P3
    else:
        foot_xy = np.array([0.0, 0.0])

    # 输出为相对 body 坐标系的 (x, z)
    return np.array([foot_xy[0], 0.0, foot_xy[1]])  # 添加 y = 0 作为中间项（用于 3D）

def forward_kinematics(abduction_angle, hip_angle, knee_angle, L1=0.08505, L2=0.2, L3=0.2):
    r_y = L1
    # print(hip_angle, knee_angle)
    r_x = -L2 * sin(hip_angle) - L3 * sin(hip_angle + knee_angle)
    r_z = -L2 * cos(hip_angle) - L3 * cos(hip_angle + knee_angle)
    return np.array([r_x, r_y, r_z])

def inverse_kinematics(x, y, z, theta_init, l1=0.08505, l2=0.2, l3=0.2):
    max_iter = 10000
    tolerance = 1e-4
    theta = theta_init
    for i in range(max_iter):
        x_curr, y_curr, z_curr = forward_kinematics(theta[0], theta[1], theta[2])
        error = np.array([x - x_curr, y - y_curr, z - z_curr])
        error_num = np.sqrt((x - x_curr)**2 + (y - y_curr)**2 + (z - z_curr)**2)
        J = compute_J(theta, l1, l2, l3)
        delta_theta = np.linalg.pinv(J) @ error
        theta += delta_theta
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        if error_num < tolerance:
            break
        if i == max_iter - 1:
            print("Inverse kinematics failed to converge.")
            return None
    
    return theta
        

def compute_J(q, L1=0.08505, L2=0.2, L3=0.2):
    theta1, theta2, theta3 = q
    J = np.array([[0, - L3*cos(theta2 + theta3) - L2*cos(theta2), -L3*cos(theta2 + theta3)], 
                  [0, 0, 0], 
                  [0,  L3*sin(theta2 + theta3) + L2*sin(theta2),  L3*sin(theta2 + theta3)]])
    return J


leg_phase = {
    0: 0.5,   # FR
    1: 0.0,  # FL
    2: 0.0,   # RR
    3: 0.5   # RL
}

# 初始化控制器
pid_params = {
    'abduction': PIDController(50, 0, 2, model.opt.timestep),
    'hip': PIDController(50, 0, 2, model.opt.timestep),
    'knee_front': PIDController(30, 0, 1, model.opt.timestep),  # 前腿膝盖
    'knee_rear': PIDController(100, 0, 2, model.opt.timestep),   # 后腿膝盖（支撑力更大）
}

gait = GaitParams()
mujoco.mj_resetDataKeyframe(model, data, 0)

x_list = []
z1 = []
z2 = []
z3 = []
t = 0

# fp = forward_kinematics(0, -0.9, 1.8)
# joint_angles = inverse_kinematics(fp[0], fp[1], fp[2], [0, 0.1, 0.1])
# print('joint_angles', joint_angles)
# fp = forward_kinematics(joint_angles[0], joint_angles[1], joint_angles[2])
# print(fp)


# foot_relevent_xpos = [0, 0.085, -0.25]
#             # foot_relevent_xpos[1] = foot_relevent_xpos[1] * side_sign
#             foot_relevent_xpos = foot_relevent_xpos + foot_target_local
#             x, y, z = foot_relevent_xpos
init_angles = np.array([0, 0.9, -1.8])
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        sim_time = data.time
        
        # 获取 roll（左右倾角）作为姿态反馈
        quat = [1, 0, 0, 0]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy 使用 [x, y, z, w]
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        # data.qpos[2] = 1
        # 生成目标关节角度
        target_pos = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]  # 初始化为直立姿态
        body_pos = data.qpos[0:3]  # 身体位置
        body_euler = np.array([roll, pitch, yaw])
        duty_ratio = 0.75  # 支撑相比例
        swing_time = (1.0 - duty_ratio) / gait.frequency
        stance_time = duty_ratio / gait.frequency

        leg_origins = {
            0: np.array([0.25, -0.1, -0.27]),  # FR
            1: np.array([0.25, 0.1, -0.27]),  # FL
            2: np.array([-0.25, -0.1, -0.27]),  # RR
            3: np.array([-0.25, 0.1, -0.27]),  # RL
        }

        for leg in range(4):
            phase = (sim_time * gait.frequency + leg_phase[leg]) % 1.0
            side_sign = 1 if leg in [0, 2] else -1

            # 生成足端轨迹（相对身体）
            foot_target_local = foot_trajectory(
                phase * (swing_time + stance_time),
                swing_time=swing_time,
                stance_time=stance_time,
                step_height=0.1,
                step_length=0.15
            )
            foot_relevent_xpos = [0, 0.085, -0.25]
            # foot_relevent_xpos[1] = foot_relevent_xpos[1] * side_sign
            foot_relevent_xpos = foot_relevent_xpos + foot_target_local
            x, y, z = foot_relevent_xpos
            

            ik_ans = inverse_kinematics(x, y, z, init_angles)
            if ik_ans is None:
                continue
            joint_angles = ik_ans
            fp = forward_kinematics(joint_angles[0], joint_angles[1], joint_angles[2])
            if leg == 0 and foot_target_local[2] != 0 and foot_target_local[0] != 0:
                x_list.append(fp[0])
                z1.append(fp[2])
                # z2.append(joint_angles[1])
                # z3.append(joint_angles[2])
            elif leg == 0:
                x_list.append(np.nan)
                z1.append(np.nan)
                # z2.append(np.nan)
                # z3.append(np.nan)

            target_pos[leg * 3 + 0] = joint_angles[0]
            target_pos[leg * 3 + 1] = joint_angles[1]
            target_pos[leg * 3 + 2] = joint_angles[2]

        
        # 计算控制量
        ctrl = np.zeros(model.nu)  
        for i, joint_idx in enumerate(joint_indices):
            current_pos = data.qpos[model.jnt_qposadr[joint_idx]]
            current_vel = data.qvel[model.jnt_dofadr[joint_idx]]
            error = target_pos[i] - current_pos

            if i % 3 == 0:
                joint_type = "abduction"
            elif i % 3 == 1:
                joint_type = "hip"
            else:  # i % 3 == 2，膝盖
                if i // 3 in [0, 1]:  # 0:FR, 1:FL -> 前腿
                    joint_type = "knee_front"
                else:  # 2:RR, 3:RL -> 后腿
                    joint_type = "knee_rear"

            ctrl[actuator_indices[i]] = pid_params[joint_type].compute(error, current_vel)

        # 限制范围并赋值
        data.ctrl[:] = np.clip(ctrl, -33.5, 33.5)
        
        # 步进仿真
        mujoco.mj_step(model, data)
        viewer.sync()

# print("x_list:", x_list)
# print("z_list:", z_list)
# plt.figure(figsize=(8, 4))
# plt.plot(x_list, z1, z2, z3, label='Foot trajectory')
# plt.title('Foot trajectory in body frame over one gait cycle')
# plt.xlabel('X (forward) [m]')
# plt.ylabel('Z (up) [m]')
# plt.grid(True)
# plt.axis('equal')
# plt.legend()
# plt.show()