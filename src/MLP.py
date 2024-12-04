import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=3*32*32, hidden_size1=1024, hidden_size2=512, num_classes=10):
        super(MLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size1),  # 输入层到第一个隐藏层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 防止过拟合
            
            nn.Linear(hidden_size1, hidden_size2),  # 第一个隐藏层到第二个隐藏层
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 防止过拟合
            
            nn.Linear(hidden_size2, num_classes)  # 第二个隐藏层到输出层
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平为一维向量
        x = self.mlp(x)
        return x

# 测试网络
if __name__ == "__main__":
    model = MLP(input_size=3*32*32, hidden_size1=1024, hidden_size2=512, num_classes=10)
    x = torch.randn(2, 3, 32, 32)  # CIFAR-10 图像大小为 32x32x3
    y = model(x)
    print(y.size())  # 输出大小为 [2, 10]
