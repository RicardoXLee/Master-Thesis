import pickle
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from PIL import Image

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # 随机旋转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 调整颜色
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
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

# def preprocess(data):
#     data = data / 255.0
#     data = (data - 0.5) / 0.5
#     return data

train_files = [
    "./cifar-10-batches-py/data_batch_1",
    "./cifar-10-batches-py/data_batch_2",
    "./cifar-10-batches-py/data_batch_3",
    "./cifar-10-batches-py/data_batch_4",
    "./cifar-10-batches-py/data_batch_5"
]
test_file = "./cifar-10-batches-py/test_batch"

train_data, train_labels = load_cifar10(train_files)

train_data = preprocess_with_transforms(train_data, train_transforms).clone().detach().float()
train_labels = torch.tensor(train_labels, dtype=torch.long)

dataset = TensorDataset(train_data, train_labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

test_file = "./cifar-10-batches-py/test_batch"

# 加载测试数据
test_data, test_labels = load_cifar10([test_file])
test_data = preprocess_with_transforms(test_data, test_transforms).clone().detach().float()
test_labels = torch.tensor(test_labels, dtype=torch.long)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)