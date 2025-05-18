import mujoco
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import mink
import matplotlib.pyplot as plt

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
        self.frequency = 0.1
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


def leg_inverse_kinematics(foot_pos_world, body_pos, body_rot, leg_origin_body, side_sign):
    """
    计算给定足端世界坐标位置下的关节角度，使机器人保持稳定姿态。

    参数：
        foot_pos_world: 世界坐标下的足端目标位置 (3,)
        body_pos: 期望身体中心位置 (3,)
        body_rot: 期望身体姿态旋转矩阵 (3, 3)
        leg_origin_body: 该腿在身体坐标系下的位置
        side_sign: 左右腿标志，右腿为 1, 左腿为 -1

    返回：
        abduction, hip, knee 关节角度
    """

    # 世界 -> 身体坐标系
    foot_pos_body = np.dot(body_rot.T, foot_pos_world - body_pos)

    # 身体坐标系 -> 腿坐标系（假设腿原点就是 hip 的位置）
    foot_pos_leg = foot_pos_body - leg_origin_body
    x, y, z = foot_pos_leg

    # 三连杆参数
    L1 = 0.0838  # hip to thigh
    L2 = 0.2     # thigh to knee
    L3 = 0.2     # knee to foot

    abduction = np.arctan2(y, -z)

    hip_to_foot = np.sqrt(x**2 + z**2)
    hip_angle = np.arctan2(-x, -z)
    D = (hip_to_foot**2 - L2**2 - L3**2) / (2 * L2 * L3)
    knee = np.arccos(np.clip(D, -1.0, 1.0))

    alpha = np.arctan2(z, x)
    # beta = np.arccos(np.clip((L2**2 + hip_to_foot**2 - L3**2) / (2 * L2 * hip_to_foot), -1.0, 1.0))
    eps = 1e-6  # 防止除以零的小常数
    denom = 2 * L2 * hip_to_foot

    if denom < eps:
        beta = 0.0  # 或者设置为合理的缺省值
    else:
        cos_beta = np.clip((L2**2 + hip_to_foot**2 - L3**2) / denom, -1.0, 1.0)
        beta = np.arccos(cos_beta)
    hip = -(alpha + beta)

    return np.array([abduction * side_sign, hip, -knee])

def euler_to_rot(euler):
    roll, pitch, yaw = euler
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

leg_phase = {
    0: 0.0,   # FR
    1: 0.25,  # FL
    2: 0.5,   # RR
    3: 0.75   # RL
}

# 初始化控制器
pid_params = {
    'abduction': PIDController(50, 0, 2, model.opt.timestep),
    'hip': PIDController(50, 0, 2, model.opt.timestep),
    'knee_front': PIDController(30, 0, 1, model.opt.timestep),  # 前腿膝盖
    'knee_rear': PIDController(50, 0, 2, model.opt.timestep),   # 后腿膝盖（支撑力更大）
}

gait = GaitParams()
mujoco.mj_resetDataKeyframe(model, data, 0)

x_list = []
z_list = []

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
        body_rot = euler_to_rot(body_euler)
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
                step_height=0.15,
                step_length=0.3
            )
            # 转为世界坐标系下的目标足端位置
            if leg == 1 and foot_target_local[2] != 0 and foot_target_local[0] != 0:
                x_list.append(foot_target_local[0] + leg_origins[leg][0])
                z_list.append(foot_target_local[2] + leg_origins[leg][2])
            elif leg == 1:
                x_list.append(np.nan)
                z_list.append(np.nan)
            foot_target_world = body_pos + np.dot(body_rot, leg_origins[leg] + foot_target_local)

            joint_angles = leg_inverse_kinematics(
                foot_pos_world=foot_target_world,
                body_pos=body_pos,
                body_rot=body_rot,
                leg_origin_body=leg_origins[leg],
                side_sign=side_sign
            )

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
plt.figure(figsize=(8, 4))
plt.plot(x_list, z_list, label='Foot trajectory')
plt.title('Foot trajectory in body frame over one gait cycle')
plt.xlabel('X (forward) [m]')
plt.ylabel('Z (up) [m]')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()