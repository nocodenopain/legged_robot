import sys
sys.path.append("D:/code/sire/install/Release/python") # 编译将输出pyd文件以及相关的lib文件

import sire_python

cs = sire_python.ControlServer.instance()
m = cs.model()
sire_python.fromXmlFile(m, "seven.xml")
try:
    cs.init()
    cs.open()
    cs.runCmdLine()
except Exception as e:
    print("Error:", e)
                    