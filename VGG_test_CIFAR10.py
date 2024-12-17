import torch
import torch.nn as nn
from src.VGG import VGG16
from tqdm import tqdm
from src.dataloader import test_loader

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(dataloader, desc="Testing", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix(
                    loss=f"{running_loss / len(dataloader):.4f}",
                    accuracy=f"{100. * correct / total:.2f}%"
                )
    return running_loss / len(dataloader), 100. * correct / total

# 加载最佳模型权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = VGG16().to(device)
vgg_model.dense[-1] = nn.Linear(4096, 10).to(device)
vgg_model.load_state_dict(torch.load("./models/CE_VGG_Best.pth", map_location=device))

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 测试模型
test_loss, test_acc = test(vgg_model, test_loader, criterion, device)

# 打印测试结果
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
