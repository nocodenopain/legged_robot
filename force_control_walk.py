import mujoco
import numpy as np
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

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
        self.amplitude = 0.15
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

def foot_trajectory(phase, swing_time, stance_time, step_height, step_length):
    phase = phase % (swing_time + stance_time)
    if phase < stance_time:
        # 支撑相：足端不动，保持接触
        return np.array([0.0, 0.0, -0.25])
    else:
        # 摆动相：椭圆轨迹
        swing_phase = (phase - stance_time) / swing_time
        x = -0.5 * step_length * np.cos(np.pi * swing_phase)  # 从后向前
        z = -0.25 + step_height * np.sin(np.pi * swing_phase)  # 抬腿
        return np.array([x, 0.0, z])

def leg_inverse_kinematics(foot_pos, leg_origin, side_sign):
    """
    foot_pos: 足端在腿坐标系下的目标位置
    leg_origin: 该腿相对身体的位置（用于变换坐标）
    side_sign: 左右腿区分符（左右反转）

    返回：[abduction, hip, knee]
    """
    x, y, z = foot_pos
    # 三连杆简化模型
    L1 = 0.0838  # hip to thigh
    L2 = 0.2     # thigh to knee
    L3 = 0.2     # knee to foot

    abduction = np.arctan2(y, -z)

    hip_to_foot = np.sqrt(x**2 + z**2)
    hip_angle = np.arctan2(-x, -z)
    D = (hip_to_foot**2 - L2**2 - L3**2) / (2 * L2 * L3)
    knee = np.arccos(np.clip(D, -1.0, 1.0))

    # Law of cosines
    alpha = np.arctan2(z, x)
    beta = np.arccos(np.clip((L2**2 + hip_to_foot**2 - L3**2) / (2 * L2 * hip_to_foot), -1.0, 1.0))
    hip = -(alpha + beta)

    return np.array([abduction * side_sign, hip, -knee])


# leg_phase = {
#     0: 0,          # FR
#     1: np.pi,      # FL
#     2: np.pi,      # RR
#     3: 0           # RL
# }

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
    'knee_rear': PIDController(70, 0, 2, model.opt.timestep),   # 后腿膝盖（支撑力更大）
}

gait = GaitParams()
mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        sim_time = data.time
        
        # 获取 roll（左右倾角）作为姿态反馈
        quat = data.qpos[3:7]  # 四元数 w, x, y, z
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy 使用 [x, y, z, w]
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)

        # 生成目标关节角度
        # target_pos = np.zeros(12)
        target_pos = [0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8]  # 初始化为直立姿态
        # base 期望高度（站立高度）
        duty_ratio = 0.75  # 支撑相比例
        swing_time = (1.0 - duty_ratio) / gait.frequency
        stance_time = duty_ratio / gait.frequency

        leg_origins = {
            0: np.array([0.0, 0.9, -1.8]),  # FR
            1: np.array([0.0, 0.9, -1.8]),  # FR
            2: np.array([0.0, 0.9, -1.8]),  # FR
            3: np.array([0.0, 0.9, -1.8]),  # FR
        }

        for leg in range(4):
            phase = (sim_time * gait.frequency + leg_phase[leg]) % 1.0
            side_sign = 1 if leg in [0, 2] else -1

            # 生成足端轨迹
            foot_target = foot_trajectory(
                phase * (swing_time + stance_time),
                swing_time=swing_time,
                stance_time=stance_time,
                step_height=0.08,
                step_length=0.1
            )

            # roll 补偿（仅在支撑相）
            roll_comp = 0.04 * roll
            if phase < duty_ratio:
                if leg in [0, 2]:  # 右侧腿
                    foot_target[0] += roll_comp
                    foot_target[2] -= 0.03 * roll_comp
                else:              # 左侧腿
                    foot_target[0] -= roll_comp
                    foot_target[2] += 0.03 * roll_comp

            # 使用IK求解角度
            joint_angles = leg_inverse_kinematics(
                foot_pos=foot_target,
                leg_origin=leg_origins[leg],
                side_sign=side_sign
            )

            # 写入目标角度
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

# gpt卡了周三问
# 不对，你的修改肯定有问题。你想，实际上你修改之前算出来的末端是对的啊，你起码得沿用那个末端的结果。而你是把foot_target_leg = np.linalg.inv(R_base) @ (foot_world - base_pos - leg_origin)，也就是改变了这个值。我认为正确的修改方式是，足端坐标不变，而是给出一个base，一个足端，然后参考这两个点进行逆运动学求解，这样既能维持原本的足端轨迹，又能限制高度