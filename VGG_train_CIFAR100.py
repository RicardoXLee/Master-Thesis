import torch
import torch.nn as nn
import torch.optim as optim
from src.VGG import VGG16
from tqdm import tqdm
from src.CIFAR100_dataloader import train_loader, val_loader

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

vgg_model.dense[-1] = nn.Linear(4096, 100).to(device) 

criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adam(vgg_model.parameters(), lr=1e-3)
optimizer = optim.SGD(vgg_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 100
best_val_loss = float('inf')
best_model_path = "./models/100_CE_VGG_Best.pth"

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
