import sys
sys.path.append("D:/code/sire/install/Release/python") # 编译将输出pyd文件以及相关的lib文件

import sire_python

import math

PI = math.pi

param = sire_python.DeltaParam()

param.a = 0.5
param.b = 0.2
param.c = 0.1
param.d = 0.7
param.e = 0.1

model = sire_python.createModelDelta(param)

pos = [-0.1, 0.1, -0.45, 0.2]
model.setOutputPos(pos)
if model.inverseKinematics():
    print("Inverse Kinematics error!")

input = model.getInputPos()
print("input1: ", input)

input2 = [-0.1, 0.1, -0.4, 0.2]
model.setInputPos(input2)
model.forwardKinematics()

output2 = model.getOutputPos()
print("output2: ", output2)

model.setOutputPos(output2)
model.inverseKinematics()

input = model.getInputPos()
print("input2: ", input)

input_below = [0, 0, 0, 0, 0]
input_upper = [2 * PI, 2 * PI, 2 * PI, 2 * PI]
print(sire_python.test_model_kinematics_pos(model, 5, input_below, input_upper))