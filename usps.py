import h5py
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Define a custom Dataset class
class USPSDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].reshape(16, 16)  # Reshape to 16x16
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)  # Convert label to long

# Load the USPS data from the h5 file
def load_usps_data(file_path):
    with h5py.File(file_path, 'r') as hf:
        train_data = np.array(hf.get('train/data')[:])
        train_labels = np.array(hf.get('train/target')[:])
        test_data = np.array(hf.get('test/data')[:])
        test_labels = np.array(hf.get('test/target')[:])
    return train_data, train_labels, test_data, test_labels

# Define transformations: resize to 32x32 and convert to 3 channels
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# Create data loaders
def get_usps_data(file_path, batch_size=64):
    train_data, train_labels, test_data, test_labels = load_usps_data(file_path)

    train_dataset = USPSDataset(train_data, train_labels, transform=transform)
    test_dataset = USPSDataset(test_data, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Example usage
file_path = r'data\usps\usps.h5'  # Use a raw string (prefix with 'r') to handle backslashes
# Alternatively, you can use forward slashes
# file_path = 'data/usps/usps.h5'
batch_size = 64
train_loader, test_loader = get_usps_data(file_path, batch_size)
