import sys
sys.path.append("D:/code/sire/install/Release/python") # 编译将输出pyd文件以及相关的lib文件

import sire_python

import math

PI = math.pi

# # 定义关节的位置，以及轴线，SCARA为RRPR机构，包含3个转动副和1个移动副，轴线都是Z轴
joint1_position = [0.0, 0.0, 0.0]
joint1_axis = [0.0, 0.0, 1.0]
joint2_position = [1.0, 0.0, 0.0]
joint2_axis = [0.0, 0.0, 1.0]
joint3_position = [1.0, 1.0, 0.0]
joint3_axis = [0.0, 0.0, 1.0]
joint4_position = [1.0, 1.0, 0.0]
joint4_axis = [0.0, 0.0, 1.0]

# # 定义3个杆件的位置与321欧拉角，以及10维的惯量向量
# # inertia_vector的定义为：[m, m*x, m*y, m*z, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]，其中x,y,z为质心位置
link1_position_and_euler321 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
link1_inertia_vector = [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0]
link2_position_and_euler321 = [1.0, 0.0, 0.0, PI/2, 0.0, 0.0]
link2_inertia_vector = [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0]
link3_position_and_euler321 = [1.0, 1.0, 0.0, PI/2, 0.0, 0.0]
link3_inertia_vector = [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0]
link4_position_and_euler321 = [1.0, 1.0, 0.0, PI/2, 0.0, 0.0]
link4_inertia_vector = [2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 0.0, 0.0, 0.0]

# # 定义末端位置与321欧拉角，这个位置为机构起始时的位置
end_effector_position_and_euler321 = [1.0, 1.0, 0.0, PI/2, 0.0, 0.0]

# # 机器人建模
m = sire_python.Model()

link1 = m.addPartByPe(link1_position_and_euler321, "321", link1_inertia_vector)
link2 = m.addPartByPe(link2_position_and_euler321, "321", link2_inertia_vector)
link3 = m.addPartByPe(link3_position_and_euler321, "321", link3_inertia_vector)
link4 = m.addPartByPe(link4_position_and_euler321, "321", link4_inertia_vector)
joint1 = m.addRevoluteJoint(link1, m.ground(), joint1_position, joint1_axis)
joint2 = m.addRevoluteJoint(link2, link1, joint2_position, joint2_axis)
joint3 = m.addPrismaticJoint(link3, link2, joint3_position, joint3_axis)
joint4 = m.addRevoluteJoint(link4, link3, joint4_position, joint4_axis)
motion1 = m.addMotion(joint1)
motion2 = m.addMotion(joint2)
motion3 = m.addMotion(joint3)
motion4 = m.addMotion(joint4)
end_effector = m.addGeneralMotionByPe(link4, m.ground(), end_effector_position_and_euler321, "321")

# # 添加求解器
iv_solver = m.solverPool().add_ik()
inverse_dynamics_solver = m.solverPool().add_id()
# # adams = m.simulatorPool().add_adams()

# m.addSolvers()

m.init()

# 位置反解
end_effector_pos_and_eul = [1.3, 1, -0.3, 0.3, 0, 0]
end_effector.setMpe(end_effector_pos_and_eul, "321")

# print(sire_python.toXmlString(m))

try:
    if iv_solver.kinPos():
        raise RuntimeError("Kinematic position failed")
except Exception as e:
    print(f"Caught exception: {e}")
    raise

print("input position :", motion1.mp(), motion2.mp(), motion3.mp(), motion4.mp())

# # 速度反解
end_effector_point_and_angular_velocity = [0.3, -0.2, 0.2, 0.0, 0.0, 0.3]
end_effector.setMva(end_effector_point_and_angular_velocity)
m.solverPool()[0].kinVel()
print("input velocity :", motion1.mv(), motion2.mv(), motion3.mv(), motion4.mv())
# 动力学反解
motion_acceleration = [9.0, 8.0, 7.0, 6.0]
motion1.setMa(motion_acceleration[0])
motion2.setMa(motion_acceleration[1])  
motion3.setMa(motion_acceleration[2])
motion4.setMa(motion_acceleration[3])

inverse_dynamics_solver.dynAccAndFce()
print("input force :", motion1.mf(), motion2.mf(), motion3.mf(), motion4.mf())

ee_result = end_effector.getMaa()
print("end effector acceleration :", ee_result[0], ee_result[1], ee_result[2], ee_result[3], ee_result[4], ee_result[5])

