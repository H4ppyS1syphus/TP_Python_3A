from tqdm import tqdm
from helper_tools.dataset import LicensePlateDataset, label_to_char
from helper_tools.model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from torchinfo import summary
from torch.utils.data import random_split

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory for TensorBoard logs
log_dir = "runs/license_plate_experiment"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# Define transformations with normalization
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
    transforms.Lambda(lambda img: transforms.functional.hflip(img)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # EMNIST mean and std
])

# Create the dataset
dataset = LicensePlateDataset(
    emnist_dataset=datasets.EMNIST(root='data', split='balanced', train=True, download=True),
    num_chars=7,
    transform=transform,
    max_samples=10000
)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# Initialize the model with correct input_size
model = CNN(num_chars=7, num_classes=62, input_size=(1, 28, 196))
model.to(device)  # Move model to device

# Initialize the weights
model.apply(model.init_weights)

# Define loss and optimizer with weight decay for regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Define a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
num_epochs = 20
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # Initialize tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)  # Outputs shape: (batch_size, num_chars, num_classes)
        labels = labels.long()

        # Reshape outputs and labels for loss computation
        outputs_flat = outputs.view(-1, model.num_classes)  # Shape: (batch_size * num_chars, num_classes)
        labels_flat = labels.view(-1)  # Shape: (batch_size * num_chars)

        # Compute loss
        loss = criterion(outputs_flat, labels_flat)

        # Backward pass and optimization
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0) * model.num_chars  # Multiply by total characters

        # Predictions and accuracy calculation
        _, preds = torch.max(outputs_flat, 1)
        running_corrects += torch.sum(preds == labels_flat.data)
        total_samples += images.size(0) * model.num_chars  # Total number of characters

        # Update tqdm progress bar with loss and accuracy
        current_loss = running_loss / total_samples
        current_acc = running_corrects.double() / total_samples
        pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.4f}'})

    # Adjust learning rate
    scheduler.step()

    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    # Append metrics for plotting
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    # Validation loop
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            outputs = model(images)
            outputs_flat = outputs.view(-1, model.num_classes)
            labels_flat = labels.view(-1)
            loss = criterion(outputs_flat, labels_flat)
            val_running_loss += loss.item() * images.size(0) * model.num_chars
            _, preds = torch.max(outputs_flat, 1)
            val_running_corrects += torch.sum(preds == labels_flat.data)
            val_total_samples += images.size(0) * model.num_chars

    epoch_val_loss = val_running_loss / val_total_samples
    epoch_val_acc = val_running_corrects.double() / val_total_samples

    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc.item())

    # Log to TensorBoard
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
    writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', epoch_val_acc, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
          f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

# Plotting Loss and Accuracy Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, 'b', label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, 'r', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, 'b', label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, 'r', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save the trained model
torch.save(model.state_dict(), 'cnn_model.pth')

# Close the SummaryWriter
writer.close()
