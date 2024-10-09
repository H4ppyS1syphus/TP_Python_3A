import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm  # For progress bars

import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2
import torch.optim as optim
import torch.nn as nn

# Path to the font file
font_path = 'fe_font/FE-FONT.TTF'

def char_to_label(char):
    if '0' <= char <= '9':
        return ord(char) - ord('0')
    elif 'A' <= char <= 'Z':
        return ord(char) - ord('A') + 10
    else:
        return 36  # '?' label

def generate_license_plate(text, font_path=None, augmentations=None):
    """
    Generate a license plate image with optional augmentations.
    """
    # Create a blank image with white background
    plate_width = np.random.randint(300, 700)
    plate_height = np.random.randint(100, 300)
    plate_image = Image.new('RGB', (plate_width, plate_height), 'white')

    # Initialize the drawing context
    draw = ImageDraw.Draw(plate_image)

    # Load the font
    if font_path is None:
        font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        font_size = 100  # Adjust font size as needed
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font_size = 70  # Adjust font size as needed
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            print(f"Font file '{font_path}' not found. Using default font.")
            font = ImageFont.load_default()

    # Calculate text width and height to center it
    left, top, right, bottom = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top
    position = ((plate_width - text_width) // 2, (plate_height - text_height) // 2)

    # Draw the text onto the image
    draw.text(position, text, fill='black', font=font)

    # Add a black border around the plate
    border_width = 5
    draw.rectangle(
        [border_width // 2, border_width // 2, plate_width - border_width // 2, plate_height - border_width // 2],
        outline='black',
        width=border_width
    )

    # Apply augmentations if any
    if augmentations:
        for aug in augmentations:
            if aug == 'rotation':
                angle = random.uniform(-15, 15)  # Increase rotation angle
                plate_image = plate_image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor='white')
            elif aug == 'gaussian_noise':
                plate_image = add_gaussian_noise(plate_image, std_range=(20, 50))  # Increase noise level
            elif aug == 'thermal_noise':
                plate_image = add_thermal_noise(plate_image, num_hot_pixels_range=(500, 2000))  # More hot pixels
            elif aug == 'occlusion':
                plate_image = add_occlusion(plate_image, num_patches_range=(2, 5), size_range=(30, 100))  # Larger occlusions
            elif aug == 'brightness':
                enhancer = ImageEnhance.Brightness(plate_image)
                factor = random.uniform(0.3, 1.7)  # Wider brightness range
                plate_image = enhancer.enhance(factor)
            elif aug == 'contrast':
                enhancer = ImageEnhance.Contrast(plate_image)
                factor = random.uniform(0.3, 1.7)  # Wider contrast range
                plate_image = enhancer.enhance(factor)
            elif aug == 'blur':
                plate_image = plate_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.0)))  # Increase blur radius
            elif aug == 'perspective_transform':
                plate_image = apply_perspective_transform(plate_image, shift_range=(-plate_width//4, plate_width//4))  # More extreme perspective

    # Resize the image back to original size if needed
    plate_image = plate_image.resize((plate_width, plate_height), Image.LANCZOS)

    return plate_image

def add_gaussian_noise(image, std_range=(20, 50)):
    """
    Add Gaussian noise to an image.
    """
    np_image = np.array(image)
    mean = 0
    std = random.uniform(*std_range)
    gauss = np.random.normal(mean, std, np_image.shape).astype(np.uint8)
    np_image = cv2.add(np_image, gauss)
    return Image.fromarray(np_image)

def add_thermal_noise(image, num_hot_pixels_range=(500, 2000)):
    """
    Add thermal noise (hot pixels) to an image.
    """
    np_image = np.array(image)
    num_hot_pixels = random.randint(*num_hot_pixels_range)
    for _ in range(num_hot_pixels):
        x = random.randint(0, np_image.shape[1] - 1)
        y = random.randint(0, np_image.shape[0] - 1)
        np_image[y, x] = [255, 255, 255]  # White pixel
    return Image.fromarray(np_image)

def add_occlusion(image, num_patches_range=(2, 5), size_range=(30, 100)):
    """
    Add occlusion patches to an image.
    """
    draw = ImageDraw.Draw(image)
    num_patches = random.randint(*num_patches_range)
    for _ in range(num_patches):
        x1 = random.randint(0, image.width - size_range[1])
        y1 = random.randint(0, image.height - size_range[1])
        x2 = x1 + random.randint(*size_range)
        y2 = y1 + random.randint(*size_range)
        draw.rectangle([x1, y1, x2, y2], fill='white')
    return image

def apply_perspective_transform(image, shift_range=(-130, 130)):
    """
    Apply a perspective transform to an image.
    """
    width, height = image.size

    # Ensure shift_range is valid
    shift_min, shift_max = shift_range
    shift_min = int(shift_min)
    shift_max = int(shift_max)

    # Random shifts for perspective
    x_shift_top = random.randint(shift_min, shift_max)
    x_shift_bottom = random.randint(shift_min, shift_max)
    y_shift_top = random.randint(-height // 4, height // 4)
    y_shift_bottom = random.randint(-height // 4, height // 4)

    # Define source points (corners of the plate)
    src_pts = [
        (0, 0),                # Top-left
        (width, 0),            # Top-right
        (width, height),       # Bottom-right
        (0, height)            # Bottom-left
    ]

    # Define destination points with perspective shifts
    dst_pts = [
        (0 + x_shift_top, 0 + y_shift_top),                      # Shift top-left corner
        (width + x_shift_top, 0 + y_shift_top),                  # Shift top-right corner
        (width + x_shift_bottom, height + y_shift_bottom),       # Shift bottom-right corner
        (0 + x_shift_bottom, height + y_shift_bottom)            # Shift bottom-left corner
    ]

    coeffs = find_coeffs(dst_pts, src_pts)

    image = image.transform((width, height), Image.PERSPECTIVE, coeffs, resample=Image.BICUBIC, fillcolor='white')

    return image

def find_coeffs(pa, pb):
    """
    Find coefficients for perspective transform.
    """
    matrix = []
    for p1, p2 in zip(pb, pa):
        # p1 from source, p2 from destination
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.array(matrix)
    B = np.array(pa).reshape(8)

    res = np.linalg.lstsq(A, B, rcond=None)[0]
    return res

def label_to_char(label):
    """
    Convert a numerical label to a character.
    """
    label = int(label)
    if 0 <= label <= 9:
        return str(label)
    elif 10 <= label <= 35:
        return chr(label - 10 + ord('A'))
    elif 36 <= label <= 61:
        return chr(label - 36 + ord('a'))
    else:
        return '?'

# Define the transformations
transform = transforms.Compose([
    transforms.Lambda(lambda img: transforms.functional.rotate(img, -90)),
    transforms.Lambda(lambda img: transforms.functional.hflip(img)),
    transforms.ToTensor()
])

# Custom dataset for license plates
class LicensePlateDataset(Dataset):
    def __init__(self, emnist_dataset, num_chars=7, transform=None, max_samples=None):
        self.emnist_dataset = emnist_dataset
        self.num_chars = num_chars
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        total_samples = len(self.emnist_dataset) // self.num_chars
        if self.max_samples is not None:
            return min(total_samples, self.max_samples)
        else:
            return total_samples

    def __getitem__(self, idx):
        chars_images = []
        chars_labels = []

        for _ in range(self.num_chars):
            rand_idx = random.randint(0, len(self.emnist_dataset) - 1)
            img, label = self.emnist_dataset[rand_idx]

            if self.transform:
                img = self.transform(img)

            chars_images.append(img)
            chars_labels.append(label)

        # Concatenate images horizontally (along the width)
        concatenated_image = torch.cat(chars_images, dim=2)

        # Combine labels into a tensor
        labels = torch.tensor(chars_labels, dtype=torch.long)

        return concatenated_image, labels

class OnTheFlyLicensePlateDataset(Dataset):
    def __init__(self, num_samples, num_chars=7, augmentations=None, font_path=None):
        self.num_samples = num_samples  # Total number of samples
        self.num_chars = num_chars
        self.augmentations = augmentations
        self.font_path = font_path

        # Character to label mapping
        self.char_to_label_dict = {str(i): i for i in range(10)}  # 0-9
        self.char_to_label_dict.update({chr(i + ord('A')): i + 10 for i in range(26)})  # A-Z

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random license plate text in French format
        letters1 = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        numbers = ''.join(random.choices('0123456789', k=3))
        letters2 = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        text = f"{letters1}-{numbers}-{letters2}"

        # Randomly choose augmentations if not specified
        if self.augmentations is None:
            augmentation_options = [
                ['rotation'],
                ['gaussian_noise'],
                ['thermal_noise'],
                ['occlusion'],
                ['brightness'],
                ['contrast'],
                ['blur'],
                ['perspective_transform'],
                ['rotation', 'gaussian_noise'],
                ['rotation', 'occlusion'],
                ['blur', 'brightness'],
                ['rotation', 'perspective_transform'],
                ['occlusion', 'perspective_transform'],
            ]
            augmentations = random.choice(augmentation_options)
        else:
            augmentations = self.augmentations

        # Generate the license plate image
        image = generate_license_plate(text, font_path=self.font_path, augmentations=augmentations)

        # Convert PIL image to numpy array
        image_np = np.array(image.convert('L'))  # Convert to grayscale numpy array

        # Resize the image to (28 * num_chars, 28)
        image_resized = cv2.resize(image_np, (28 * self.num_chars, 28), interpolation=cv2.INTER_AREA)

        # Invert colors if necessary
        inverted_image = cv2.bitwise_not(image_resized)

        # Normalize pixel values to [0, 1]
        normalized_image = inverted_image.astype(np.float32) / 255.0

        # Convert to tensor
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0)  # Add channel dimension

        # Convert text label to numerical labels
        label_str = text.replace('-', '')  # Remove dashes for labels
        label = torch.tensor([self.char_to_label_dict.get(char, 36) for char in label_str], dtype=torch.long)  # Default to 36 ('?')

        return image_tensor, label
    
class FixedLicensePlateDataset(Dataset):
    def __init__(self, image_tensors, labels):
        self.image_tensors = image_tensors
        self.labels = labels

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        return self.image_tensors[idx], self.labels[idx]
    
def generate_fixed_validation_data(num_val_samples, num_chars=7, font_path=None):
    val_license_plate_texts = []
    val_image_tensors = []

    for _ in tqdm(range(num_val_samples), desc='Generating Validation Data'):
        # Generate random license plate text in French format
        letters1 = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        numbers = ''.join(random.choices('0123456789', k=3))
        letters2 = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=2))
        text = f"{letters1}-{numbers}-{letters2}"

        # Choose augmentations for validation (less augmentation)
        augmentations = random.choice([
            None,
            ['brightness'],
            ['contrast'],
            ['blur'],
        ])

        # Generate the license plate image
        image = generate_license_plate(text, font_path=font_path, augmentations=augmentations)

        # Process the image
        image_np = np.array(image.convert('L'))
        image_resized = cv2.resize(image_np, (28 * num_chars, 28), interpolation=cv2.INTER_AREA)
        inverted_image = cv2.bitwise_not(image_resized)
        normalized_image = inverted_image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(normalized_image).unsqueeze(0)  # Add channel dimension

        # Convert text label to numerical labels
        label_str = text.replace('-', '')
        label = torch.tensor([char_to_label(char) for char in label_str], dtype=torch.long)  # Default to 36 ('?')

        val_image_tensors.append(image_tensor)
        val_license_plate_texts.append(label)

    return FixedLicensePlateDataset(val_image_tensors, val_license_plate_texts)




if __name__ == '__main__':
    # Load the EMNIST dataset
    test_dataset = datasets.EMNIST(
        root='data',
        split='letters',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create a DataLoader for the dataset
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # Display a few samples
    for i, (img, labels) in enumerate(test_loader):
        if i >= 5:
            break

        img_np = img.squeeze(0).squeeze(0).numpy()
        plt.figure(figsize=(10, 2))
        plt.imshow(img_np, cmap='gray')
        plt.axis('off')
        plt.title(''.join([label_to_char(l) for l in labels[0]]))
        plt.show()

