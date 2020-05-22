import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms

BATCH_SIZE = 32
EPOCH = 10

transforms_train = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])
train_load = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=True,transform=transforms_train),
    batch_size = BATCH_SIZE,
    shuffle = True
)
test_load = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False,transform=transforms),
    batch_size = BATCH_SIZE,
    shuffle = True
)




class Net(nn.Module):#细节 这class是小写
    def __init__(self):
        super().__init__()#细节 这没有冒号
        self.conv1 = nn.Conv2d(1,6,3,padding=1) #输入单通道图片(1,32,32) 1batch_size  (1,1,32,32)
        self.conv2 = nn.Conv2d(6,10,3,padding=1)
        self.conv3 = nn.Conv2d(10,6,3,padding=1)
        self.conv4 = nn.Conv2d(6,3,3,padding=1)
        self.fc1 = nn.Linear(3*28*28,500) #细节 这参数输入元素数，不用加batch batch输入的时候加
        self.fc2 = nn.Linear(500,100)
        self.fc3 = nn.Linear(100,10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        model1 = nn.Sequential(self.conv1,self.relu,self.conv2,self.relu,
                      self.conv3,self.relu,self.conv4,self.relu,)
        model2 = nn.Sequential(self.fc1,self.relu,self.fc2,self.relu,self.fc3)
        x = model1(x)
        x = x.view(BATCH_SIZE,-1)
        x = model2(x)
        y = self.softmax(x)
        return y
net = Net()
optimizer = optim.SGD(net.parameters(),lr=0.01)
citizerion = nn.CrossEntropyLoss()

# for i,(img,target) in enumerate(train_load):
#     print(i,img.shape,target.shape)

for i in range(EPOCH):
    for j,(img,y) in enumerate(train_load):

        output = net(img)

        loss = citizerion(output,y)
        loss.backward()
        optimizer.step()

        if j%32 ==0:
            print(j)
            print("loss = %.4f"%loss)
    torch.save("epoch{}".format(i+1))
