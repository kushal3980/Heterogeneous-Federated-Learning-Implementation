import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms.functional as TF
import h5py
import h5py
import h5py
import h5py
# class USPSDataset(Dataset):
#     def __init__(self, root_file, transform=None):
#         self.root_file = root_file
#         self.transform = transform
#         self.images, self.labels = self.load_data(root_file)

#     def load_data(self, root_file):
#         with h5py.File(root_file, 'r') as hf:
#             train = hf.get('train')
#             X_tr = train.get('data')[:]
#             y_tr = train.get('target')[:]
#             test = hf.get('test')
#             X_te = test.get('data')[:]
#             y_te = test.get('target')[:]

#         # Combine train and test data
#         images = np.concatenate((X_tr, X_te), axis=0)
#         labels = np.concatenate((y_tr, y_te), axis=0)

#         # Normalize images to [0, 1] and adjust shape if necessary
#         images = images.astype(np.float32) / 255.0  # Assuming images are in [0, 255] range

#         # Ensure images are in RGB format (num_samples, channels, height, width)
#         images = np.transpose(images, (0, 3, 1, 2))  # Assuming images are (num_samples, height, width, channels)

#         return images, labels

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]

#         if self.transform:
#             image = self.transform(image)

#         return image, label
class SYNDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images, self.labels = self.load_data(root_dir)

    def load_data(self, root_dir):
        data = []
        labels = []
        for i in range(10):
            folder = os.path.join(root_dir, str(i))
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder {folder} does not exist.")
            for filename in os.listdir(folder):
                image_path = os.path.join(folder, filename)
                image = Image.open(image_path).convert('RGB')  # Convert to RGB
                if self.transform:
                    image = self.transform(image)
                data.append(image)
                labels.append(i)
        return data, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def get_mnist_data(batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    train_loader = DataLoader(
        datasets.MNIST(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.MNIST(root='./data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# def get_usps_data(batch_size):
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     try:
#         train_dataset = USPSDataset(root_file='./data/usps/usps.h5', transform=transform)
#         test_dataset = USPSDataset(root_file='./data/usps/usps.h5', transform=transform)  # Assuming same HDF5 file for test
        
#     except FileNotFoundError:
#         print("Error: USPS dataset not found. Please make sure the dataset is downloaded and the path to usps.h5 is correct.")
#         return None, None
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#     return train_loader, test_loader


def get_svhn_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    try:
        train_loader = DataLoader(
            datasets.SVHN(root='./data', split='train', download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            datasets.SVHN(root='./data', split='test', download=True, transform=transform),
            batch_size=batch_size, shuffle=False)
    except OSError:
        print("Error: SVHN dataset download failed. Please check your internet connection and try again.")
        return None, None
    
    return train_loader, test_loader
import os
def get_syn_data(batch_size):
    # Assuming you've manually downloaded and extracted the SYN dataset into './data/archive'
    root_dir = './data/archive/synthetic_digits'
    train_dir = os.path.join(root_dir, 'imgs_train')
    valid_dir = os.path.join(root_dir, 'imgs_valid')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        train_dataset = SYNDataset(root_dir=train_dir, transform=transform)
        test_dataset = SYNDataset(root_dir=valid_dir, transform=transform)
    except FileNotFoundError:
        print("Error: SYN dataset not found. Please make sure the dataset is downloaded and the paths are correct.")
        return None, None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
# Example usage:
# mnist_train_loader, mnist_test_loader = get_mnist_data(batch_size=128)
# usps_train_loader, usps_test_loader = get_usps_data(batch_size=128)
# svhn_train_loader, svhn_test_loader = get_svhn_data(batch_size=128)
# syn_train_loader, syn_test_loader = get_syn_data(batch_size=128)
