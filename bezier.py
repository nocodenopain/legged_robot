import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(P0, P1, P2, P3, num_points=100):
    t = np.linspace(0, 1, num_points).reshape(-1, 1)  # 变成 (100, 1)
    curve = (1 - t)**3 * P0 + \
            3 * (1 - t)**2 * t * P1 + \
            3 * (1 - t) * t**2 * P2 + \
            t**3 * P3
    return curve

# 控制点 (x, z)，假设 z 是垂直方向
P0 = np.array([0.0, 0.0])
P1 = np.array([0.1, 0.15])
P2 = np.array([0.2, 0.15])
P3 = np.array([0.3, 0.0])

# 计算贝塞尔曲线
# curve = bezier_curve(P0, P1, P2, P3)

# # 绘图
# plt.figure(figsize=(8, 4))
# plt.plot(curve[:, 0], curve[:, 1], label='Foot trajectory', linewidth=2)
# plt.plot([P0[0], P1[0], P2[0], P3[0]], [P0[1], P1[1], P2[1], P3[1]], 'ro--', label='Control Points')
# plt.title('Quadruped Foot Trajectory using Bézier Curve')
# plt.xlabel('X (forward direction)')
# plt.ylabel('Z (vertical direction)')
# plt.grid(True)
# plt.axis('equal')
# plt.legend()
# plt.show()



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

def main():
    swing_time = 0.25      # 300 ms 腾空
    stance_time = 0.75     # 700 ms 支撑
    step_length = 0.1     # 10 cm 跨步
    step_height = 0.05    # 5 cm 抬脚

    total_time = swing_time + stance_time
    dt = 0.01  # 时间分辨率

    x_list = []
    z_list = []

    t_list = np.arange(0, total_time, dt)
    for t in t_list:
        pos = foot_trajectory(
            phase_time=t,
            swing_time=swing_time,
            stance_time=stance_time,
            step_height=step_height,
            step_length=step_length
        )
        x_list.append(pos[0])
        z_list.append(pos[2])  # z 是竖直方向
        print(f"t: {t:.2f}, foot_pos: {pos}")

    # 画图
    plt.figure(figsize=(8, 4))
    plt.plot(x_list, z_list, label='Foot trajectory')
    plt.title('Foot trajectory in body frame over one gait cycle')
    plt.xlabel('X (forward) [m]')
    plt.ylabel('Z (up) [m]')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
