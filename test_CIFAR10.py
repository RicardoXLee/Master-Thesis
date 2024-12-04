import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.VGG import VGG16
from tqdm import tqdm

def load_cifar10(file_paths):
    data = []
    labels = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            data.append(batch['data'])
            labels += batch['labels']
    data = np.vstack(data).reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    return data, labels

def preprocess(data):
    data = data / 255.0
    data = (data - 0.5) / 0.5
    return data

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

# 测试数据路径
test_file = "./cifar-10-batches-py/test_batch"

# 加载测试数据
test_data, test_labels = load_cifar10([test_file])
test_data = torch.tensor(preprocess(test_data), dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载最佳模型权重
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = VGG16().to(device)
vgg_model.dense[-1] = nn.Linear(4096, 10).to(device)
vgg_model.load_state_dict(torch.load("CE_VGG_Best.pth", map_location=device))

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 测试模型
test_loss, test_acc = test(vgg_model, test_loader, criterion, device)

# 打印测试结果
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
