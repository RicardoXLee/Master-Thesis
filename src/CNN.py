import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 输入通道 3，输出通道 32
            nn.BatchNorm2d(32),  # 批归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化，输出大小 16x16x32
        )
        
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输入 32 通道，输出 64 通道
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出大小 8x8x64
        )
        
        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输入 64 通道，输出 128 通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出大小 4x4x128
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # 展平后的输入大小为 4x4x128
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Dropout 防止过拟合
            nn.Linear(256, num_classes)  # 最终输出类别数
        )
    
    def forward(self, x):
        x = self.conv1(x)  # 第一层卷积
        x = self.conv2(x)  # 第二层卷积
        x = self.conv3(x)  # 第三层卷积
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # 全连接层
        return x

# 测试网络
if __name__ == "__main__":
    model = CNN(num_classes=10)  # CIFAR-10 有 10 个类别
    x = torch.randn(2, 3, 32, 32)  # CIFAR-10 图像大小为 32x32x3
    y = model(x)
    print(y.size())  # 输出大小为 [2, 10]
