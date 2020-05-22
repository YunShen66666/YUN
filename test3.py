import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import net

BATCH_SIZE = 32
EPOCH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])

test_load = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False,transform=transforms_train),
    batch_size = BATCH_SIZE,
    shuffle = True
)

net = net.Net()
