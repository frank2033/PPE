

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 创建一个简单的神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建随机数据
X = torch.randn(1000, 10)  # 1000个样本，每个样本10个特征
y = torch.randint(0, 2, (1000,))  # 二分类标签

# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型、损失函数和优化器
model = SimpleModel().to(device)  # 将模型移动到GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for batch_X, batch_y in dataloader:
        # 将数据移动到GPU
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("训练完成!")

# 验证GPU是否真正参与计算
if torch.cuda.is_available():
    # 检查模型参数是否在GPU上
    first_param_device = next(model.parameters()).device
    print(f"模型参数所在设备: {first_param_device}")
    
    # 检查计算是否在GPU上进行
    test_input = torch.randn(1, 10).to(device)
    with torch.no_grad():
        output = model(test_input)
    print(f"输出张量所在设备: {output.device}")
    
    if first_param_device.type == 'cuda' and output.device.type == 'cuda':
        print("GPU已成功启用并参与计算!")
    else:
        print("GPU未正常参与计算!")
else:
    print("未检测到GPU，使用CPU进行计算")
