# 导入 PyTorch 库
import torch
# 导入 PyTorch 神经网络模块，用于构建和训练神经网络
import torch.nn as nn
# 导入 NumPy 库，用于处理数组和矩阵
import numpy as np

# 准备数据
def prepare_data(data, window_size):
    X, Y = [], [] # 初始化两个空列表用于存储输入和输出的数据
    for i in range(len(data) - window_size): # 遍历数据的长度减去窗口大小的范围
        X.append(data[i:i+window_size]) # 将当前数据作为输入添加到X列表中
        Y.append(data[i+window_size]) # 将当前数据作为输出添加到Y列表中
    return np.array(X), np.array(Y) # 将X和Y转换为NumPy数组并返回

# 定义模型
class BP_Model(nn.Module):
    def __init__(self):
        super(BP_Model, self).__init__() # 调用父类的初始化方法
        self.fc1 = nn.Linear(3, 5)  # 定义一个全连接层，输入维度为3,输出维度为5
        self.fc2 = nn.Linear(5, 1)  # 定义一个全连接层，输入维度为5,输出维度为1

    def forward(self, x):
        x = torch.tanh(self.fc1(x))  # 将输入数据传入第一个全连接层，并使用tansig激活函数处理后得到隐藏层输出
        x = torch.sigmoid(self.fc2(x))  # 将隐藏层输出传入第二个全连接层，并使用logsis激活函数处理后得到最终输出
        return x

# 提供的销量数据
sales_data = np.array([2056, 2395, 2600, 2298, 1634, 1600, 1873, 1478, 1900, 1500, 2046, 1556])
# 计算销量数据的平均值
mean = np.mean(sales_data)
# 计算销量数据的标准差
std = np.std(sales_data)
# 对销量数据进行标准化处理，即将每个数据减去平均值并除以标准差
normalized_sales_data = (sales_data - mean) / std

# 准备数据
window_size = 3  # 定义窗口大小为3
X, Y = prepare_data(normalized_sales_data, window_size)  # 调用prepare_data函数，对数据进行处理，得到特征矩阵X和标签向量Y
X_train = torch.FloatTensor(X)  # 将特征矩阵X转换成PyTorch的浮点型张量
Y_train = torch.FloatTensor(Y).view(-1, 1)  # 将标签向量Y转换成PyTorch的浮点型张量，并将其reshape成列向量

# 初始化模型和优化器
model = BP_Model()  # 创建BP_Model类的实例对象，即定义了一个神经网络模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam算法对模型进行优化，学习率为0.001
loss_fn = nn.MSELoss()  # 使用均方误差作为损失函数

# 训练数据
num_epochs = 1200 # 训练周期为1200个周期
for epoch in range(num_epochs):
    model.train() # 将模型设置为训练模式
    optimizer.zero_grad() # 将优化器的梯度清零
    output = model(X_train) # 使用模型对训练数据进行预测，得到预测结果output
    loss = loss_fn(output, Y_train) # 计算预测结果与真实值之间的均方误差损失函数loss
    loss.backward() # 对损失函数进行反向传播，计算梯度
    optimizer.step() # 根据梯度更新模型参数
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}') # 每100个周期输出一次当前训练的epoch和损失函数值

# 获取最后三个月的规范化销量数据作为模型输入
last_three_months = torch.FloatTensor(normalized_sales_data[-3:]).view(1, -1)  # 将最近三个月的销量数据转换为PyTorch张量，并将其reshape成一列向量

# 模型预测
model.eval()  # 设置模型为评估模式，即不更新权重
with torch.no_grad():  # 不计算梯度，即只进行前向传播，不计算反向传播
    normalized_prediction = model(last_three_months)  # 对输入数据进行预测，得到预测结果normalized_prediction

# 反向规范化预测值
predicted_sales = normalized_prediction.numpy() * std + mean  # 将预测结果从[-1,1]映射回原始数据域，即将其反向规范化到[0,1]之间
predicted_sales = predicted_sales.flatten()  # 将输出转换为1D数组，便于显示和处理
print('预测第二年一月结果：',predicted_sales[0])  # 显示预测的销量值