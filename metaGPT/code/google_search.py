import os
from datetime import datetime
# 获取当前路径
path = os.getcwd() / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


print(path)