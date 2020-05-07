import torch.nn as nn
import torch.nn.functional as F


class Mymodel(nn.Module):

    def __init__(self):
        super(Mymodel,self).__init__()
        self.conv1=nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1)
        self.conv2=nn.Conv2d(96, 256, kernel_size=5, stride=1,padding=1)
        self.conv3=nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1)
        self.fc1=nn.Linear(384*6*6,4096)#384* image dim
        self.fc2=nn.Linear(4096,1024)
        self.fc3=nn.Linear(1024,512)
        self.fc4=nn.Linear(512,8)

    def forward(self, x):
        out = self.conv1(x)
        out = F.dropout(out, p=0.8)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.conv2(out)
        out = F.dropout(out, p=0.8)
        out = F.relu(out)
        out = F.max_pool2d(out,2)
        out = self.conv3(out)
        out = F.dropout2d(out, p=0.8)
        out = F.max_pool2d(out,2)
        #print("h",out.size())
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out=self.fc4(out)
        return out









