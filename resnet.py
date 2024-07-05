# import pickle
# import torch
# import torch.nn as nn
# from torchvision.models import resnet18
# from data_loader import get_cifar100_data
# import numpy as np
# import socket
# import struct
# import warnings
# import torch.optim as optim

# warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated.*")

# class Client:
#     def __init__(self, host='127.0.0.1', port=65432):
#         self.host = host
#         self.port = port
#         self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

#     def connect_to_server(self):
#         self.s.connect((self.host, self.port))
    
#     def send_data(self, data):
#         shape_bytes = struct.pack('ii', *data.shape)
#         self.s.sendall(shape_bytes)
#         self.s.sendall(data.tobytes())
#         self.s.sendall(struct.pack('f', 0.0))  # Termination signal
#         print("Data sent to server.")

#     def receive_average_logits(self, shape):
#         length_data = self.s.recv(4)
#         if not length_data:
#             raise ValueError("Did not receive data length from server.")

#         data_length = struct.unpack('!I', length_data)[0]

#         # Receive the actual data based on the received length
#         received_data = b""
#         while len(received_data) < data_length:
#             chunk = self.s.recv(min(4096, data_length - len(received_data)))
#             if not chunk:
#                 raise ValueError("Connection closed before all data received.")
#             received_data += chunk

#         # Deserialize the received data
#         received_array = pickle.loads(received_data)
#         return received_array

#     def close_connection(self):
#         self.s.close()
#         print("Connection closed.")

# def check_for_nans(data, name):
#     if torch.isnan(data).any():
#         print(f"NaN values found in {name}")
#         return True
#     return False

# def normalize_data(data):
#     mean = data.mean(0)
#     std = data.std(0) + 1e-8  # Adding epsilon to avoid division by zero
#     return (data - mean) / std

# def off_diagonal(x):
#     n, m = x.shape
#     assert n == m
#     return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# def cross_co(z1, z2):
#     if check_for_nans(z1, "z1") or check_for_nans(z2, "z2"):
#         return torch.tensor(float('nan'), requires_grad=True)

#     z1n = normalize_data(z1)
#     z2n = normalize_data(z2)

#     c = z1n.T @ z2n
#     c.div_(z1.shape[0])  # Normalize by the number of samples

#     if check_for_nans(c, "cross-correlation matrix"):
#         return torch.tensor(float('nan'), requires_grad=True)

#     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
#     off_diag = off_diagonal(c).add_(1).pow_(2).sum()
#     col_loss = on_diag + 0.0051 * off_diag
    
#     return col_loss

# def extract_linear_outputs_resnet(model, images, device):
#     model.train()
#     images = images.to(device)
#     with torch.no_grad():
#         linear_output = model(images).detach()
#     return linear_output

# def update_model(resnet, images, avg_logits_resnet, optimizer, device, batch_idx):
#     # Extract linear output
#     linear_output = extract_linear_outputs_resnet(resnet, images, device)
#     linear_output.requires_grad_(True)

#     avg_logits_resnet = torch.tensor(avg_logits_resnet, dtype=torch.float32, device=device)
#     if check_for_nans(avg_logits_resnet, "avg_logits_resnet"):
#         print(f"Batch {batch_idx + 1} - NaN detected in avg_logits_resnet. Replacing with zeros.")
#         avg_logits_resnet = torch.zeros_like(avg_logits_resnet)

#     # Compute contrastive loss
#     col_loss = cross_co(linear_output, avg_logits_resnet)
#     print(f"Batch {batch_idx + 1} - Col Loss: {col_loss.item()}")

#     # Update the model
#     optimizer.zero_grad()
#     col_loss.backward()
#     optimizer.step()

#     return col_loss.item()

# if __name__ == "__main__":
#     # Load CIFAR-100 data
#     train_loader, _ = get_cifar100_data(batch_size=512, num_classes=10)

#     # Limit training samples to 5000
#     train_loader = torch.utils.data.DataLoader(
#         train_loader.dataset,
#         batch_size=512,
#         sampler=torch.utils.data.SubsetRandomSampler(range(4608))
#     )

#     # Load the pre-trained ResNet-18 model
#     resnet = resnet18(pretrained=False)
#     resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # Modify the final fully connected layer
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     resnet = nn.DataParallel(resnet).to(device)

#     optimizer = optim.Adam(resnet.parameters(), lr=0.001)

#     # Example usage
#     client = Client()
#     client.connect_to_server()

#     col_loss_list = []

#     for batch_idx, (images, _) in enumerate(train_loader):
#         # Extract linear output
#         linear_output = extract_linear_outputs_resnet(resnet, images, device)
#         linear_output.requires_grad_(True)

#         # Send to server
#         client.send_data(linear_output.detach().numpy())
#         avg_logits_resnet = client.receive_average_logits(linear_output.shape)
        
#         # Update model and collect loss
#         col_loss = update_model(resnet, images, avg_logits_resnet, optimizer, device, batch_idx)
#         col_loss_list.append(col_loss)

#         print(f"Batch {batch_idx + 1} - Model updated using col_loss.")

#     client.close_connection()
#     print("Training completed and connection closed.")
#     print(f"Contrastive Losses for all batches: {col_loss_list}")
import pickle
import torch
import torch.nn as nn
from torchvision.models import resnet18
from data_loader import get_cifar100_data
import numpy as np
import socket
import struct
import warnings
import torch.optim as optim
from pvt_data_loader import get_mnist_data
from scipy.special import softmax
from copy import deepcopy
import torch.nn.functional as F




warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated.*")

class Client:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect_to_server(self):
        self.s.connect((self.host, self.port))
    
    def send_data(self, data):
        shape_bytes = struct.pack('ii', *data.shape)
        self.s.sendall(shape_bytes)
        self.s.sendall(data.tobytes())
        self.s.sendall(struct.pack('f', 0.0))  # Termination signal
        print("Data sent to server.")

    def receive_average_logits(self, shape):
        length_data = self.s.recv(4)
        if not length_data:
            raise ValueError("Did not receive data length from server.")

        data_length = struct.unpack('!I', length_data)[0]

        # Receive the actual data based on the received length
        received_data = b""
        while len(received_data) < data_length:
            chunk = self.s.recv(min(4096, data_length - len(received_data)))
            if not chunk:
                raise ValueError("Connection closed before all data received.")
            received_data += chunk

        # Deserialize the received data
        received_array = pickle.loads(received_data)
        return received_array

    def close_connection(self):
        self.s.close()
        print("Connection closed.")
    
    def send_signal(self, signal):
        self.s.sendall(signal.encode())


def check_for_nans(data, name):
    if torch.isnan(data).any():
        print(f"NaN values found in {name}")
        return True
    return False

def normalize_data(data):
    mean = data.mean(0)
    std = data.std(0) + 1e-8  # Adding epsilon to avoid division by zero
    return (data - mean) / std

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cross_co(z1, z2):
    if check_for_nans(z1, "z1") or check_for_nans(z2, "z2"):
        return torch.tensor(float('nan'), requires_grad=True)

    z1n = normalize_data(z1)
    z2n = normalize_data(z2)

    c = z1n.T @ z2n
    c.div_(z1.shape[0])  # Normalize by the number of samples

    if check_for_nans(c, "cross-correlation matrix"):
        return torch.tensor(float('nan'), requires_grad=True)

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).add_(1).pow_(2).sum()
    col_loss = on_diag + 0.0051 * off_diag
    
    return col_loss

def extract_linear_outputs_resnet(model, images, device):
    model.train()
    images = images.to(device)
    with torch.no_grad():
        linear_output = model(images).detach()
    return linear_output

def update_model(resnet, images, avg_logits_resnet, optimizer, device, batch_idx):
    # Extract linear output
    linear_output = extract_linear_outputs_resnet(resnet, images, device)
    linear_output.requires_grad_(True)

    avg_logits_resnet = torch.tensor(avg_logits_resnet, dtype=torch.float32, device=device)
    if check_for_nans(avg_logits_resnet, "avg_logits_resnet"):
        print(f"Batch {batch_idx + 1} - NaN detected in avg_logits_resnet. Replacing with zeros.")
        avg_logits_resnet = torch.zeros_like(avg_logits_resnet)

    # Compute contrastive loss
    col_loss = cross_co(linear_output, avg_logits_resnet)
    print(f"Batch {batch_idx + 1} - Col Loss: {col_loss.item()}")

    # Update the model
    optimizer.zero_grad()
    col_loss.backward()
    optimizer.step()

    return col_loss.item()
def send_signal(self, signal):
    self.s.sendall(signal.encode())


def generate_logits(model, data_loader, device):
    model.eval()
    logits_list = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            logits = model(images)
            logits_list.append(logits.cpu().numpy())
    
    return np.vstack(logits_list)


def train_on_mnist(resnet, optimizer, device, mnist_train_loader):
    for mnist_batch_idx, (mnist_images, mnist_labels) in enumerate(mnist_train_loader):
        mnist_images = mnist_images.to(device)
        mnist_labels = mnist_labels.to(device)

        # Update the model on MNIST data
        optimizer.zero_grad()
        outputs = resnet(mnist_images)
        loss = nn.CrossEntropyLoss()(outputs, mnist_labels)
        loss.backward()
        optimizer.step()

        print(f"MNIST Batch {mnist_batch_idx + 1} - Loss: {loss.item()}")
        # break
    return loss

def calculate_inter_domain_loss(previous_logits, current_logits):
    current_logits_n = normalize_data(current_logits)
    previous_logits_n = normalize_data(previous_logits)
    
    current_logits_t = torch.tensor(current_logits_n, dtype=torch.float32)
   
    previous_logits_t = torch.tensor(previous_logits_n, dtype=torch.float32)
    current_softmax = F.softmax(current_logits_t, dim=1)
    previous_softmax = F.softmax(previous_logits_t, dim=1)

    # Convert current_softmax to log-probabilities
    log_previous_softmax = torch.log(previous_softmax)

    # Calculate KL divergence
    inter_domain_loss = F.kl_div(log_previous_softmax, current_softmax, reduction='batchmean')

    return inter_domain_loss

def intra_domain_loss(current_logits, initial_logits):
    current_logits_n = normalize_data(current_logits)
    initial_logits_n = normalize_data(initial_logits)

   
    current_logits_t = torch.tensor(current_logits_n, dtype=torch.float32)
    initial_logits_t = torch.tensor(initial_logits_n, dtype=torch.float32)
    
    current_softmax = F.softmax(current_logits_t, dim=1)
    initial_softmax = F.softmax(initial_logits_t, dim=1)

    # Convert current_softmax to log-probabilities
    log_current_softmax = torch.log(current_softmax)

    # Calculate KL divergence
    kl_divergence = F.kl_div(log_current_softmax, initial_softmax, reduction='batchmean')

    # print("KL Divergence:", kl_divergence.item())

    
    return kl_divergence


def freeze_model_and_generate_logits(model, data_loader, device, save_path):
    model.eval()
    logits = generate_logits(model, data_loader, device)
    np.save(save_path, logits)
    print(f"Generated logits saved to {save_path}")
    return logits

if __name__ == "__main__":
    # Load CIFAR-100 data
    CommunicationEpoch = 20
    train_loader, _ = get_cifar100_data(batch_size=512, num_classes=10)

    # Limit training samples to 5000
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=512,
        sampler=torch.utils.data.SubsetRandomSampler(range(4608))
    )

    mnist_train_loader, _ = get_mnist_data(batch_size=512)
    mnist_subset = torch.utils.data.Subset(mnist_train_loader.dataset, range(150))
    mnist_subset_loader = torch.utils.data.DataLoader(mnist_subset, batch_size=150, shuffle=True)


    # Load the pre-trained ResNet-18 model
    resnet = resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 10)  # Modify the final fully connected layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = nn.DataParallel(resnet).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    initial_resnet = deepcopy(resnet)
    # initial_resnet.module.load_state_dict(resnet18(pretrained=True).state_dict())



    # Save the model
    torch.save(resnet.state_dict(), 'resnet_model.pth')
   
    # Example usage
    client = Client()
    client.connect_to_server()
    col_loss_list = []
    local_loss_c=[]
    for epoch_index in range(CommunicationEpoch):

        if epoch_index > 0:
            # Load the model from the previous epoch
            resnet.load_state_dict(torch.load(f'resnet_model_epoch_{epoch_index - 1}.pth'))
        else:
            resnet.load_state_dict(torch.load('resnet_model.pth'))


        

        for batch_idx, (images, _) in enumerate(train_loader):
            # Load the model from saved state

            # Extract linear output
            linear_output = extract_linear_outputs_resnet(resnet, images, device)
            linear_output.requires_grad_(True)

            # Send to server
            client.send_data(linear_output.detach().numpy())
            avg_logits_resnet = client.receive_average_logits(linear_output.shape)
            
            # Update model and collect loss
            col_loss = update_model(resnet, images, avg_logits_resnet, optimizer, device, batch_idx)
            col_loss_list.append(col_loss)

            print(f"Batch {batch_idx + 1} - Model updated using col_loss.")
        torch.save(resnet.state_dict(), f'resnet_IM_model_.pth')

        resnet.load_state_dict(torch.load(f'resnet_IM_model_.pth'))
        for i in range(10):
            mnist_logits = generate_logits(resnet, mnist_subset_loader, device)
            # np.save('mnist_logits.npy', mnist_logits)
            print("shape of mnist logits:-",mnist_logits.shape)

            cross_entropy=train_on_mnist(resnet, optimizer, device,mnist_subset_loader)
            print("cross_entropy loss=",cross_entropy)

            # Load the previous round's model if this is not the first epoch
            if epoch_index > 0:
                previous_model_path = f'resnet_model_epoch_{epoch_index - 1}.pth'
                resnet.load_state_dict(torch.load(previous_model_path))

                # Generate logits with the previous round's model
                mnist_logits_previous = generate_logits(resnet, mnist_subset_loader, device)
                # np.save(f'mnist_logits_previous_epoch_{epoch_index}.npy', mnist_logits_previous)
                print("Shape of mnist logits with previous round model:", mnist_logits_previous.shape)

                # Calculate inter-domain loss
                inter_domain_loss = calculate_inter_domain_loss(mnist_logits_previous, mnist_logits)
                # print(f"Inter-domain loss: {inter_domain_loss}")

                initial_logits = freeze_model_and_generate_logits(initial_resnet, mnist_subset_loader, device, 'initial_mnist_logits.npy')
                # print("Initial logits shape:", initial_logits.shape)


                intra_loss = intra_domain_loss(mnist_logits, initial_logits)
                # print(f"Intra-domain loss: {intra_loss}")
                
                dual_loss=inter_domain_loss+intra_loss
                # print("dual loss:-",dual_loss)

                local_loss=cross_entropy+dual_loss
                print("local loss is:-",local_loss)
                # Update the model using the local loss
                optimizer.zero_grad()
                local_loss_tensor = torch.tensor(local_loss, requires_grad=True, device=device).clone().detach().requires_grad_(True)
                local_loss_tensor.backward()  
                optimizer.step()
                print("model update from continual learning")
                # optimizer.zero_grad()
                # local_loss.backward(retain_graph=True)  # retain_graph=True if you plan to do multiple backward passes
                # optimizer.step()
                local_loss_c.append(local_loss)
        torch.save(resnet.state_dict(), f'resnet_model_epoch_{epoch_index}.pth')
        client.send_signal("ready")


    print("local loss list:-",local_loss_c)
    client.close_connection()
        # print("Training completed and connection closed.")
        # print(f"Contrastive Losses for all batches: {col_loss_list}")
