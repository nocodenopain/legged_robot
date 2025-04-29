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
        self.amplitude = 0.3
        self.frequency = 0.2
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

leg_phase = {
    0: 0,          # FR
    1: np.pi,      # FL
    2: np.pi,      # RR
    3: 0           # RL
}

# 初始化控制器
pid_params = {
    'abduction': PIDController(50, 0, 2, model.opt.timestep),
    'hip': PIDController(50, 0, 2, model.opt.timestep),
    'knee_front': PIDController(30, 0, 1, model.opt.timestep),  # 前腿膝盖
    'knee_rear': PIDController(80, 0, 10, model.opt.timestep),   # 后腿膝盖（支撑力更大）
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
        target_pos = np.zeros(12)
        # for leg in range(4):
        #     phase = sim_time * 2 * np.pi * gait.frequency + leg_phase[leg]
        #     side_sign = 1 if leg in [0, 2] else -1   # FR/RR是右腿，FL/RL是左腿
        #     target_pos[leg*3 + 0] = 0.2 * side_sign * np.sin(phase)    # abduction镜像
        #     target_pos[leg*3 + 1] = 0.7 + gait.amplitude * np.sin(phase)  # hip
        #     target_pos[leg*3 + 2] = -1.5 + 0.3 * np.sin(phase)  # knee
        for leg in range(4):
            phase = sim_time * 2 * np.pi * gait.frequency + leg_phase[leg]
            side_sign = 1 if leg in [0, 2] else -1  # FR/RR是右腿，FL/RL是左腿

            # 基础步态
            abduction = 0.2 * side_sign * np.sin(phase)
            hip = 0.7 + gait.amplitude * np.sin(phase)
            knee = -1.5 + 0.3 * np.sin(phase)

            # 追加 roll 姿态补偿项（单位为 rad，小于 0.3），尝试系数 0.4 可调
            roll_comp = 0.4 * roll
            if leg in [0, 2]:  # FR / RR，右侧，roll > 0 会压右腿
                hip -= roll_comp
                knee -= 0.7 * roll_comp
            else:              # FL / RL，左侧
                hip += roll_comp
                knee += 0.7 * roll_comp

            # 赋值目标角度
            target_pos[leg * 3 + 0] = abduction
            target_pos[leg * 3 + 1] = hip
            target_pos[leg * 3 + 2] = knee

        
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

        ctrl = ctrl / 2

        # 限制范围并赋值
        data.ctrl[:] = np.clip(ctrl, -33.5, 33.5)
        
        # Debug用
        print(data.actuator_force[:])

        # 步进仿真
        mujoco.mj_step(model, data)
        viewer.sync()
