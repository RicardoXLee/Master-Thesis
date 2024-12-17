import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
from PIL import Image

# Define the custom CIFAR-100 Dataset
class CIFAR100Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        self.data = data_dict[b'data']
        self.labels = data_dict[b'fine_labels']
        self.data = self.data.reshape((-1, 3, 32, 32)).astype(np.uint8)  # Convert to NCHW format
        self.labels = np.array(self.labels)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        image = Image.fromarray(image.transpose(1, 2, 0))  # Convert to HWC format for PIL.Image

        if self.transform:
            image = self.transform(image)

        # Convert label to LongTensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label
    
# Define data transformations
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
])

# Load datasets
train_dataset = CIFAR100Dataset(data_path='./cifar-100-python/train', transform=train_transforms)
test_dataset = CIFAR100Dataset(data_path='./cifar-100-python/test', transform=test_transforms)

# Split train dataset into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Assign test_transforms to validation dataset
val_dataset.dataset.transform = test_transforms

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
