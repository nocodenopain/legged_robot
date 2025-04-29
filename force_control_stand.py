import mujoco
import numpy as np
from mujoco import viewer

sence_xml = """
<mujoco model="a1 scene">
  <include file="a1.xml"/>

  <statistic center="0 0 0.1" extent="0.8"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <body name="world_axes" pos="0 0 0.005">
    <!-- x axis -->
        <geom type="cylinder" pos="0.25 0 0" euler="0 1.578 0" size="0.005 0.25" rgba="1 0 0 0.1" contype="0" conaffinity="0"/>
    <!-- y axis -->
        <geom type="cylinder" pos="0 0.25 0" euler="1.578 0 0" size="0.005 0.25" rgba="0 1 0 0.1" contype="0" conaffinity="0"/>
    <!-- z axis -->
        <geom type="cylinder" pos="0 0 0.25" euler="0 0 0" size="0.005 0.25" rgba="0 0 1 0.1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
# 加载模型和仿真数据
model = mujoco.MjModel.from_xml_string(sence_xml)  # 替换为你的 XML 文件
data = mujoco.MjData(model)

#  应用站立姿态关键帧
mujoco.mj_resetDataKeyframe(model, data, 0)

#  设置零速度、零加速度
data.qvel[:] = 0
data.qacc[:] = 0

# 计算正向动力学
mujoco.mj_forward(model, data)

# 准备RNE计算
rn_flags = 1  # 计算重力补偿扭矩
rn_result = np.zeros(model.nv)  

# 调用mj_rne计算逆向动力学
mujoco.mj_rne(model, data, rn_flags, rn_result)

print("\n=== RNE计算的重力补偿扭矩 ===")
for i in range(0, model.nu):  
    joint_id = model.actuator_trnid[i, 0]
    if joint_id >= 0 and joint_id < model.njnt:  
        jnt_type = model.jnt_type[joint_id]
        jnt_addr = model.jnt_qposadr[joint_id]
        
        if jnt_addr < len(rn_result) + 1:  
            torque = rn_result[jnt_addr - 1]
            print(f"执行器 {i} ({model.actuator(i).name}) -> 关节 {joint_id} ({model.joint(joint_id).name}): {torque:.4f} N·m")
            
            data.ctrl[i] = -torque  
        else:
            print(f"警告: 关节 {joint_id} 的地址 {jnt_addr} 超出范围")
    else:
        print(f"警告: 执行器 {i} 映射到无效关节 {joint_id}")

qpos_ref = model.key_ctrl[0]

# PD 控制参数
Kp = 100.0
Kd = 2.0

with viewer.launch_passive(model, data) as v:

    # 仿真主循环
    while v.is_running():
        # 计算 PD 控制信号
        q_error = qpos_ref[:] - data.qpos[7:19]
        qvel_error = -data.qvel[1:13]
        torque = Kp * q_error + Kd * qvel_error

        data.ctrl[:] = torque

        mujoco.mj_step(model, data)
        v.sync()