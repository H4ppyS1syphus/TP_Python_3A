import torch
from matplotlib import pyplot as plt
from helper_tools.dataset import FixedLicensePlateDataset, generate_fixed_validation_data, OnTheFlyLicensePlateDataset
from helper_tools.dataset import label_to_char
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from helper_tools.model import CNN

# Define character to label mapping for validation
char_to_label = {str(i): i for i in range(10)}  # 0-9
char_to_label.update({chr(i + ord('A')): i + 10 for i in range(26)})  # A-Z

# Number of samples
num_train_samples = 10000  # Number of training samples
num_val_samples = 2000    # Number of validation samples

# Font path for the license plate generation
font_path = 'fe_font/FE-FONT.TTF'

# Training dataset (on-the-fly generation)
train_dataset = OnTheFlyLicensePlateDataset(
    num_samples=num_train_samples,
    num_chars=7,
    font_path=font_path
)

# Generate validation dataset
val_dataset = generate_fixed_validation_data(num_val_samples, num_chars=7, font_path=font_path)

# DataLoaders
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

# Initialize the model
model = CNN(num_chars=7, num_classes=62)  # Adjust num_classes if necessary
model.load_state_dict(torch.load('cnn_model.pth'))

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 30

# Lists to store loss and accuracy values
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    # Initialize tqdm progress bar for training
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training', unit='batch')

    for images, labels in train_loader_tqdm:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        labels = labels.long()

        outputs_flat = outputs.view(-1, model.num_classes)
        labels_flat = labels.view(-1)

        loss = criterion(outputs_flat, labels_flat)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0) * model.num_chars
        _, preds = torch.max(outputs_flat, 1)
        running_corrects += torch.sum(preds == labels_flat.data)
        total_samples += images.size(0) * model.num_chars

        # Update progress bar
        train_loader_tqdm.set_postfix({'Loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc.item())

    print(f'Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    val_total_samples = 0

    # Initialize tqdm progress bar for validation
    val_loader_tqdm = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation', unit='batch')

    with torch.no_grad():
        for images, labels in val_loader_tqdm:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            labels = labels.long()

            outputs_flat = outputs.view(-1, model.num_classes)
            labels_flat = labels.view(-1)

            loss = criterion(outputs_flat, labels_flat)

            val_running_loss += loss.item() * images.size(0) * model.num_chars
            _, preds = torch.max(outputs_flat, 1)
            val_running_corrects += torch.sum(preds == labels_flat.data)
            val_total_samples += images.size(0) * model.num_chars

            # Update progress bar
            val_loader_tqdm.set_postfix({'Loss': f'{loss.item():.4f}'})

    val_epoch_loss = val_running_loss / val_total_samples
    val_epoch_acc = val_running_corrects.double() / val_total_samples

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_epoch_acc.item())

    print(f'Epoch [{epoch+1}/{num_epochs}] Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')


torch.save(model.state_dict(), 'new_train.pth')

# Plotting Loss and Accuracy Curves
epochs = range(1, num_epochs + 1)

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b', label='Training Loss')
plt.plot(epochs, val_losses, 'r', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Save the plot as an image file
plt.savefig('training_validation_curves.png')
plt.show()
