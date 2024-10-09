import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from helper_tools.model import CNN
from helper_tools.dataset import OnTheFlyLicensePlateDataset
import urllib.request
import cv2
import numpy as np

# Load the dataset

test_loader = torch.utils.data.DataLoader(
    OnTheFlyLicensePlateDataset(num_samples=100, num_chars=7, font_path='fe_font/FE-FONT.TTF'),
    batch_size=1, shuffle=True
)
 
test_dataset = OnTheFlyLicensePlateDataset(num_samples=100, num_chars=7, font_path='fe_font/FE-FONT.TTF')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_path = 'new_train.pth'
model = CNN(num_chars=7, num_classes=62)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
with torch.no_grad():
    for i in range(10):
        image, label = test_dataset[i]
        image = image.unsqueeze(0)  # Add batch dimension

        outputs = model(image)
        _, predicted = torch.max(outputs, 2)
        predicted = predicted.squeeze(0).cpu().numpy()

        # Map predictions to characters
        label_to_char = {v: k for k, v in test_dataset.char_to_label.items()}
        predicted_chars = ''.join([label_to_char[p] for p in predicted])
        true_chars = ''.join([label_to_char[l.item()] for l in label])

        # Display the image and predictions
        plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f'True: {true_chars} | Predicted: {predicted_chars}')
        plt.show()



# URL of the license plate image
url = 'https://evs-strapi-images-prod.imgix.net/changer_plaque_immatriculation_d235e7ed91.jpg?w=3840&q=75'

# Read the image from the URL
resp = urllib.request.urlopen(url)
image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

# Manually define cropping coordinates
x_start = 600
x_end = image.shape[1] - 600
y_start = 800
y_end = image.shape[0] - 800

# Crop the image
cropped_image = image[y_start:y_end, x_start:x_end]

# Resize the image
# num_chars = 7
# target_height = 28
# target_width = 28 * num_chars
# resized_image = cv2.resize(cropped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# Convert to grayscale and invert colors
gray_resized = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
inverted_image = cv2.bitwise_not(gray_resized)


# Display the preprocessed image
plt.figure(figsize=(10, 2))
plt.imshow(inverted_image, cmap='gray')
plt.axis('off')
plt.title('Inverted Image')
plt.show()

# # Apply thresholding
# thresh_image = cv2.adaptiveThreshold(
#     inverted_image, 
#     255, 
#     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#     cv2.THRESH_BINARY, 
#     11,  # Block size
#     2    # Constant subtracted from the mean
# )

thresh_image = cv2.threshold(inverted_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# Normalize
normalized_image = thresh_image.astype(np.float32) / 255.0

# Display the preprocessed image
plt.figure(figsize=(10, 2))
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')
plt.title('Preprocessed Image')
plt.show()

# Resize the image
num_chars = 7
target_height = 28
target_width = 28 * num_chars
resized_image = cv2.resize(normalized_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# Convert to tensor and add dimensions
tensor_image = torch.from_numpy(resized_image)
input_tensor = tensor_image.unsqueeze(0).unsqueeze(0)

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)

# Load the model

model.eval()

# Make prediction
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 2)
    predicted = predicted.squeeze(0).cpu().numpy()

# Map predictions to characters
char_to_label = {str(i): i for i in range(10)}  # 0-9
char_to_label.update({chr(i + ord('A')): i + 10 for i in range(26)})  # A-Z
label_to_char = {v: k for k, v in char_to_label.items()}

predicted_chars = ''.join([label_to_char.get(p, '?') for p in predicted])

# Display the prediction
plt.figure(figsize=(10, 2))
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')
plt.title(f'Predicted License Plate: {predicted_chars}')
plt.show()

