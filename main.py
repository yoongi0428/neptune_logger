# Core codes for training MNIST classifier are from https://nextjournal.com/gkoehler/pytorch-mnist

import os
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from utilities import set_random_seed
from net import Net
from Neptune import NeptuneLogger

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=2020)
args = parser.parse_args()

set_random_seed(args.seed)

# Load MNIST data from torchvision
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
                                batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                    ])),
                                batch_size=args.batch_size, shuffle=False)

# Define classifier and optimizer
network = Net().to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=args.learning_rate)

# Build NeptuneLogger
api = os.getenv('NEPTUNE_API_TOKEN')
logger = NeptuneLogger(
    api_key=api,
    project_name='yoonkij/neptunetest',
    experiment_name='NeptuneMNIST',
    description='Sample MNIST code for Neptune Test',
    tags=['MNIST', 'YourTag'],
    hparams=vars(args),
    upload_source_files=['main.py', 'net.py', 'utilities.py'],
    hostname='MY-SERVER',
    offline=False
)

# Log hyper-parameters
logger.log_hparams(vars(args))

# Train classifier
train_loss_history = []
test_acc_history = []

best_acc = -1
for epoch in range(1, args.num_epochs + 1):
    train_losses = []

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            # Log on Console
            print('Train Epoch: {} [{}/{}] \tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                loss.item()))

        # Accumulate
        train_losses.append(loss.item())
    
    train_loss = sum(train_losses) / len(train_losses)

    network.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = network(data)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()
            total += len(data)
    test_acc = correct / total

    print(f'Test Epoch: {epoch} Acc: {test_acc}')

    if best_acc < test_acc:
        best_acc = test_acc
        torch.save(network.state_dict(), 'model.p')
    
    # Log train loss and test accuracy every epoch to Neptune
    logger.log_metric('Train Loss', train_loss, epoch=epoch)
    logger.log_metric('Test Acc.', test_acc, epoch=epoch)

    train_loss_history.append(train_loss)
    test_acc_history.append(test_acc)

logger.log_artifact('model.p')

# Plot train loss and log to Neptune
fig = plt.figure()
plt.title('Train Loss by Epoch')
plt.xlabel('Epoch')
plt.ylabel('NLL Loss')
plt.plot(list(range(len(train_loss_history))), train_loss_history, color='blue')
logger.log_image('Train Loss Plot', fig)

# Plot test accuracy and log to Neptune
fig = plt.figure()
plt.title('Test Accuracy by Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(list(range(len(test_acc_history))), test_acc_history, color='blue')
# plt.show()
logger.log_image('Test Accuracy Plot', fig)

# Plot Sample predictions and log to Neptune
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    output = network(example_data.to(device))
example_data = example_data.detach().cpu().numpy()

fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])

logger.log_image('Sample Predictions', fig)