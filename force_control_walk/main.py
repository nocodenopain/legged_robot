import mujoco
import mujoco.viewer
import numpy as np
from controller.mpc_controller import MpcController

# 加载 Mujoco 模型
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 打开一个可视化窗口
viewer = mujoco.viewer.launch_passive(model, data)

# 控制器初始化
controller = MpcController(horizon=5)  # 例如设置预测步长

# 时间步长
dt = model.opt.timestep  # 例：0.002

# 关节名称对应表
hip_joints = ['FR_hip_joint', 'FL_hip_joint', 'RR_hip_joint', 'RL_hip_joint']
leg_joints = ['FR_thigh_joint', 'FL_thigh_joint', 'RR_thigh_joint', 'RL_thigh_joint']
foot_joints = ['FR_calf_joint', 'FL_calf_joint', 'RR_calf_joint', 'RL_calf_joint']

# 获取关节id
hip_ids = [model.joint(name).id for name in hip_joints]
leg_ids = [model.joint(name).id for name in leg_joints]
foot_ids = [model.joint(name).id for name in foot_joints]

# 控制循环
while viewer.is_running():
    # 从Mujoco获取状态量
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()
    xpos = data.xpos[model.body('torso').id]  # 身体中心位置
    xquat = data.xquat[model.body('torso').id]  # 身体朝向四元数
    xmat = data.xmat[model.body('torso').id].reshape(3, 3)  # 旋转矩阵

    # 姿态角速度
    ang_vel = data.qvel[3:6]  

    # 关节位置和速度
    q = qpos[7:7+12]  # 跳过自由度（一般7维：位置+四元数）
    q_dot = qvel[6:6+12]

    # 读取触地传感器（可以用接触检测）
    foot_contacts = np.zeros(4)
    for contact in data.contact[:data.ncon]:
        geom1 = model.geom(contact.geom1).name
        geom2 = model.geom(contact.geom2).name
        if 'calf' in geom1 or 'calf' in geom2:
            for i, foot in enumerate(['FR_calf', 'FL_calf', 'RR_calf', 'RL_calf']):
                if foot in geom1 or foot in geom2:
                    foot_contacts[i] = 1

    # 设置参数（根据你的mpcController需要）
    vx, vy, vz = 0.1, 0.0, 0.0  # 目标速度
    d_roll, d_pitch, d_yaw = 0, 0, 0  # 目标角度
    offsets = np.zeros((3,4))
    durations = np.ones(4) * 0.25
    rbs = np.zeros((3,4))
    mass = model.body_mass[model.body('torso').id]
    I = np.diag(model.body_inertia[model.body('torso').id])
    offset = np.zeros((3,4))
    Kpcom, Kdcom = 50, 10
    Kpbase, Kdbase = 50, 10
    IterationsBetweenMpc = 30
    stancetime = 0.2
    swingtime = 0.25
    height = 0.02
    horizon = 5
    Kp_cartesian = 100
    Kd_cartesian = 5
    nIterations = 1000
    
    # 调用你的 MPC 控制器
    t = data.time
    tao1, tao2, tao3, tao4 = controller.compute_control(
        xmat, ang_vel, xpos[0], xpos[1], xpos[2], 
        data.qvel[0:3], q[0], q[1], q[2], q[3], 
        0, 0, 0, 0, t, foot_contacts, 
        vx, vy, offsets, durations, 
        bodyheight=0.27, d_roll=d_roll, d_pitch=d_pitch, d_yaw=d_yaw,
        rbs=rbs, mass=mass, I=I, offset=offset,
        Kpcom=Kpcom, Kdcom=Kdcom, 
        Kpbase=Kpbase, Kdbase=Kdbase,
        dt=dt, IterationsBetweenMpc=IterationsBetweenMpc,
        stancetime=stancetime, swingtime=swingtime,
        height=height, horizon=horizon, 
        Kp_cartesian=Kp_cartesian, Kd_cartesian=Kd_cartesian,
        vz=vz, nIterations=nIterations
    )

    # 将力矩写入Mujoco的数据结构
    tau = np.zeros(model.nv)
    tau[hip_ids[0]] = tao1[0]
    tau[leg_ids[0]] = tao1[1]
    tau[foot_ids[0]] = tao1[2]

    tau[hip_ids[1]] = tao2[0]
    tau[leg_ids[1]] = tao2[1]
    tau[foot_ids[1]] = tao2[2]

    tau[hip_ids[2]] = tao3[0]
    tau[leg_ids[2]] = tao3[1]
    tau[foot_ids[2]] = tao3[2]

    tau[hip_ids[3]] = tao4[0]
    tau[leg_ids[3]] = tao4[1]
    tau[foot_ids[3]] = tao4[2]

    data.ctrl[:] = tau

    # 前进一步
    mujoco.mj_step(model, data)

    viewer.sync()
