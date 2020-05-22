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

train_load = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=True,transform=transforms_train),
    batch_size = BATCH_SIZE,
    shuffle = True
)
net = net.Net(BATCH_SIZE)
optimizer = optim.SGD(net.parameters(),lr=0.01)
citizerion = nn.L1Loss()
def train():
    for i in range(EPOCH):
        for j, (img, y) in enumerate(train_load):
            img = img.to(DEVICE)
            y = y.to(DEVICE)
            output = net(img)

            y = y.view(32, -1).float()

            loss = citizerion(output, y)
            loss.backward()
            optimizer.step()

            if j % 32 == 0:
                print(j)
                print("loss = %.4f" % loss)
        torch.save("epoch{}".format(i + 1))

train()
