from tqdm import tqdm
from helper_tools.dataset import LicensePlateDataset, label_to_char
from helper_tools.model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
transform = transforms.Compose([
    lambda img: transforms.functional.rotate(img, -90),
    lambda img: transforms.functional.hflip(img),
    transforms.ToTensor()
])

train_loader = torch.utils.data.DataLoader(
    LicensePlateDataset(emnist_dataset=datasets.EMNIST(root='data', split='balanced', train=True, download=True),
                        num_chars=7, transform=transform, max_samples=10000),
    batch_size=32, shuffle=True)

# Training the model with tqdm and accuracy tracking
num_epochs = 20
train_losses = []
train_accuracies = []
model = CNN(num_chars=7, num_classes=62)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Initialize tqdm progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    
    for images, labels in pbar:
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
    
    # Calculate epoch loss and accuracy
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    # Append metrics for plotting
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# Plotting Loss and Accuracy Curves
epochs = range(1, num_epochs + 1)

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b', label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'r', label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

torch.save(model.state_dict(), 'cnn_model.pth')