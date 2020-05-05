import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch
import timeit
num_epochs=100
num_classes=9
learning_rate=0.001
momentum=0.95
image_size=256
crop_size=256
path_to_train_data="/home/vakilsearch/Workspace/Resnext-classifier/dataset/train"
path_to_test_data="/home/vakilsearch/Workspace/Resnext-classifier/dataset/test"
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_transform=transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.ImageFolder(path_to_train_data, transform=train_transform)
test_dataset=datasets.ImageFolder(path_to_test_data, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=45,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=45, shuffle=True)
#helper.imshow(images[0], normalize=False)
#check availailbty of gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

classes=("aadhar","bankStatement","drivingLicense","electricityBill","pan","passport","phoneBill","rentalAgreement","voterID")

model = torch.hub.load('pytorch/vision:v0.5.0', 'resnext50_32x4d', pretrained=False)

#change the last layer output feature based on our requirement
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(2048, num_classes)

net=model
net=net.to(device)
print(net)

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()


# training the resnext
print("Number of train samples: ", len(train_dataset))
print("Number of test samples: ", len(test_dataset))
print("Detected Classes are: ", train_dataset.class_to_idx) # classes are detected by folder structure
#move model to GPU


#move the dataset to the GPU
#optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum,nesterov=True)
criterion = nn.CrossEntropyLoss()



#traing the dataset
for epoch in range(num_epochs):# loop over the dataset multiple times
    print("training_started at ")
    start = timeit.default_timer()

    running_loss = 0.0
    for i, data in enumerate(train_loader , 0):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


        # print statistics
        running_loss += loss.item()
        print("loss:", running_loss, " and epoch number is ", f"Training loss: {running_loss / len(train_loader)}",(epoch + 1, i + 1))
    stop = timeit.default_timer()
print('Finished Training')
print('Time: ', stop - start)

#save the model
path_save="./resnext_9__0.001.pth"
torch.save(net.state_dict(),path_save)
#load the  saved model
net.load_state_dict(torch.load(path_save))
# we need to make another dataloader for testing data
dataiter = iter(test_loader)
images, labels = dataiter.next()
images=images.to(device)
outputs = net(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(25)))

# performance of the cnn
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 3808 test images: %d %%' % (
    100 * correct / total))
# accuracy of each classes

class_correct = list(0. for i in range(num_classes))#number of classes
class_total = list(0. for i in range(num_classes))#number of classes
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(num_classes):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(num_classes):#number of classes
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))