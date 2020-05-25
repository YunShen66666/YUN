import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import net

BATCH_SIZE = 512
EPOCH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

transforms_train = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])

train_load = torch.utils.data.DataLoader(
    datasets.MNIST('data',download=True,train=True,transform=transforms_train),
    batch_size = BATCH_SIZE,
    shuffle = True
)
net = net.Net().to(DEVICE)
optimizer = optim.SGD(net.parameters(),lr=0.001)
citizerion = nn.CrossEntropyLoss()
def train():
    for i in range(EPOCH):
        for j, (img, y) in enumerate(train_load):
            img = img.to(DEVICE)
            y = y.to(DEVICE)
            output = net(img)


            loss = citizerion(output, y)
            loss.backward()
            optimizer.step()

            if j % 30 == 0:
                print("epoch:{},batch:{},loss={:.4f}".format(i,j,loss.item()))
    state = {'net':net.state_dict()}
    torch.save(state,"epoch10")

train()
