import torch
import torch.nn as nn
import torch.optim as optim
from src.VGG import VGG16
from tqdm import tqdm
from src.dataloader import train_loader, val_loader
from src.losses import uce_loss

def train(model, train_loader, optimizer, device, regularization_weight=1e-5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Convert labels to one-hot encoding
        one_hot_labels = torch.zeros_like(outputs)
        one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

        # Compute UCE Loss
        loss = uce_loss(outputs, one_hot_labels, regularization_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    return running_loss / len(train_loader), accuracy


def validate(model, val_loader, device, regularization_weight=1e-5):
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            one_hot_labels = torch.zeros_like(outputs)
            one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

            # Compute UCE Loss
            loss = uce_loss(outputs, one_hot_labels, regularization_weight)
            running_val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    return running_val_loss / len(val_loader), accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg_model = VGG16().to(device)

vgg_model.dense[-1] = nn.Sequential(
    nn.Linear(4096, 10),
    nn.Softplus(beta=1, threshold=20)  # Ensure outputs are positive for Dirichlet
).to(device)

#criterion = lambda outputs, targets: uce_loss(outputs, targets, regularization_weight=1e-5)
optimizer = optim.Adam(vgg_model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

num_epochs = 100
best_val_loss = float('inf')
best_model_path = "./models/Dirichlet_VGG_Best.pth"

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    # Train and validate
    train_loss, train_acc = train(vgg_model, train_loader, optimizer, device)
    val_loss, val_acc = validate(vgg_model, val_loader, device)

    scheduler.step()

    # Save the model if validation loss is the best
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(vgg_model.state_dict(), best_model_path)
        print(f"New best model saved with validation loss {val_loss:.4f}")

    # Print progress
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

