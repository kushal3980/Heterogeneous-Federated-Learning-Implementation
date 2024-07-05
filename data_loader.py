import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def filter_classes(dataset, class_indices, num_samples=None):
    targets = torch.tensor(dataset.targets)
    mask = torch.tensor([target in class_indices for target in targets])
    dataset.data = dataset.data[mask.numpy()]
    dataset.targets = targets[mask].tolist()
    dataset.targets = [class_indices.index(target) for target in dataset.targets]

    if num_samples is not None:
        # Randomly select num_samples indices
        indices = torch.randperm(len(dataset.data))[:num_samples]
        dataset.data = dataset.data[indices]
        dataset.targets = [dataset.targets[i] for i in indices]

    return dataset

def get_cifar100_data(batch_size=128, num_classes=10, num_samples=5000):
    # Define the classes to be used (first 10 classes from CIFAR-100 by default)
    class_indices = list(range(num_classes))

    # Define the data augmentation transformations
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    # Load CIFAR-100 training and test datasets
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Filter datasets to only include the selected classes
    trainset = filter_classes(trainset, class_indices, num_samples)
    testset = filter_classes(testset, class_indices, num_samples)

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# Example usage:
# trainloader, testloader = get_cifar100_data(batch_size=128, num_classes=10, num_samples=5000)
