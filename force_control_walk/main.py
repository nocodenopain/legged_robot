import mujoco
import mujoco.viewer
import numpy as np
from controller.mpc_controller import MpcController
# 加载 Mujoco 模型
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

TIME_STEP = 2  # 时间步长
t = 0
body_height = 0.28

vx = 0.0
vy = 0
vz = 0.0

d_roll = 0
d_pitch = 0
d_yaw = 0

rbs = np.array([[0.183, 0.183, -0.183, -0.183],
                [-0.13205, 0.13205, -0.13205, 0.13205],
                [-0.4, -0.4, -0.4, -0.4]])

# 参数定义
def inertia_vector_to_tensor(I_vec):
    return np.diag(I_vec)

def parallel_axis(I_body, m, r_body, r_total):
    r = r_body - r_total
    return I_body + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))

total_mass = np.sum(model.body_mass)
com_total = sum(model.body_mass[i] * data.xipos[i] for i in range(model.nbody)) / total_mass

I_total = np.zeros((3, 3))
for i in range(model.nbody):
    m = model.body_mass[i]
    r_body = data.xipos[i]
    I_diag = model.body_inertia[i]
    I_tensor = inertia_vector_to_tensor(I_diag)
    I_total += parallel_axis(I_tensor, m, r_body, com_total)
offset = np.array([[0.204, 0.204, -0.204, -0.204],
                   [-0.146, 0.146, -0.146, 0.146],
                   [0, 0, 0, 0]])

Kpcom = np.diag([400, 400, 400])
Kdcom = np.diag([160, 160, 160])
Kpbase = np.diag([1000, 1000, 1000])
Kdbase = np.diag([40, 40, 40])

# 其他参数
dt = 0.002  # 时间间隔
IterationsBetweenMpc = 15  # MPC之间的迭代次数
stancetime = 0.15
swingtime = 0.15
height = 0.07
horizon = 10
Kp_cartesian = np.diag([100, 100, 100])
Kd_cartesian = np.diag([10, 20, 10])

# 迭代次数和其他参数
nIterations = 10
offsets = np.array([0, 5, 5, 0])
durations = np.array([5, 5, 5, 5])

# 初始化状态变量
q1 = np.zeros(4)
q2 = np.zeros(4)   
q3 = np.zeros(4)
q4 = np.zeros(4)
XYZ = np.zeros(3)
RPY = np.zeros(3)
tao1 = np.zeros(3)
tao2 = np.zeros(3)
tao3 = np.zeros(3)
tao4 = np.zeros(3)
base_id = model.body("trunk").id
controller = MpcController(horizon=horizon)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        frc_FR = data.sensordata[0:3]
        frc_FL = data.sensordata[3:6]
        frc_RR = data.sensordata[6:9]
        frc_RL = data.sensordata[9:12]

        fz_FR = frc_FR[2]
        fz_FL = frc_FL[2]
        fz_RR = frc_RR[2]
        fz_RL = frc_RL[2]
        forces = np.array([fz_FR, fz_FL, fz_RR, fz_RL])
        foot_senser = (forces > 0.5).astype(int)
        sim_time = data.time
        ctrl = controller.compute_control(
            R=data.xmat[base_id].reshape(3, 3),
            w=data.qvel[3:6],
            x=data.qpos[0],
            y=data.qpos[1],
            z=data.qpos[2],
            v=data.qvel[0:3],
            q1=data.qpos[7:10],
            q2=data.qpos[10:13],
            q3=data.qpos[13:16],
            q4=data.qpos[16:19],
            w1=data.qvel[6:9],
            w2=data.qvel[9:12],
            w3=data.qvel[12:15],
            w4=data.qvel[15:18],
            t=sim_time,
            foot_senser=foot_senser,
            vx=vx,
            vy=vy,
            offsets=offsets,
            durations=durations,
            bodyheight=body_height,
            d_roll=d_roll,
            d_pitch=d_pitch,
            d_yaw=d_yaw,
            rbs=rbs,
            mass=total_mass,
            I=I_total,
            offset=offset,
            Kpcom=Kpcom,
            Kdcom=Kdcom,
            Kpbase=Kpbase,
            Kdbase=Kdbase,
            dt=dt,  # 调整时间步长
            IterationsBetweenMpc=IterationsBetweenMpc,  # 调整迭代次数
            stancetime=stancetime,  # 调整站立时间
            swingtime=swingtime,  # 调整摆动时间
            height=height,  # 调整高度
            horizon=horizon,  # MPC预测步数
            Kp_cartesian=Kp_cartesian,  # 笛卡尔空间位置控制增益
            Kd_cartesian=Kd_cartesian,  # 笛卡尔空间速度控制增益
            vz=vz,
            nIterations=nIterations
        )
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()