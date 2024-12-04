import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.VGG import VGG16
from tqdm import tqdm
from PIL import Image
from src.CNN import CNN
from src.MLP import MLP

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 调整颜色
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


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

def preprocess_with_transforms(data, transform):
    """
    将 numpy.ndarray 数据转换为 PIL.Image 后应用数据增强。
    """
    data = np.transpose(data, (0, 2, 3, 1))  # 转换形状为 (N, H, W, C)
    data = [transform(Image.fromarray(image.astype('uint8'))) for image in data]  # 转换为 PIL.Image 并应用数据增强
    data = torch.stack(data)  # 合并为单个 Tensor
    return data

train_files = [
    "./cifar-10-batches-py/data_batch_1",
    "./cifar-10-batches-py/data_batch_2",
    "./cifar-10-batches-py/data_batch_3",
    "./cifar-10-batches-py/data_batch_4",
    "./cifar-10-batches-py/data_batch_5"
]

train_data, train_labels = load_cifar10(train_files)

train_data = torch.tensor(preprocess_with_transforms(train_data, train_transforms), dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)

dataset = TensorDataset(train_data, train_labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc="Training", leave=False) as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix(
                loss=f"{running_loss / len(dataloader):.4f}",
                accuracy=f"{100. * correct / total:.2f}%"
            )
    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(dataloader, desc="Validating", leave=False) as pbar:
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = VGG16().to(device)
#cnn_model = CNN().to(device)

vgg_model.dense[-1] = nn.Linear(4096, 10).to(device) 

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=1e-4)
#optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 100
best_val_loss = float('inf')
best_model_path = "CE_VGG_Best.pth"

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss, train_acc = train(vgg_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(vgg_model, val_loader, criterion, device)
    scheduler.step()

    # Save the model if validation loss is the best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(vgg_model.state_dict(), best_model_path)
        print(f"New best model saved with validation loss {val_loss:.4f}")

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
