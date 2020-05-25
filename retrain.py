import torch
import torch.nn as nn
import net
from torchvision import datasets,transforms

transform_test = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))])

test_load = torch.utils.data.DataLoader(
    datasets.MNIST('data',train=False,transform=transform_test),
    batch_size = 32,
    shuffle = True
)

net = net.Net(32)

checkpoint = torch.load("epoch8",map_location=torch.device('cpu'))

net.load_state_dict(checkpoint['net'])

correct = 0
for i,(img,target) in enumerate(test_load):

    output = net(img)
    _,predict = torch.max(output,1)
    correct += (predict==target).sum().item()
print(correct/len(test_load))
