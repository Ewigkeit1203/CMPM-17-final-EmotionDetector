import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 
import numpy as np

#-------------------------------------DATA Augumentation-----------------------------------------------------

train_transform = v2.Compose([
    v2.ToTensor(),
    v2.RandomHorizontalFlip(0.3), 
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True) 
])

test_transform = v2.Compose([
    v2.ToTensor(),
    v2.Grayscale(num_output_channels=1),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

'''
Cannot use v2.ToPILImage():

UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release.
Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.
Output is equivalent up to float precision.  
warnings.warn(
'''
#----------------------------------Creating DataLoader-------------------------------------------

base_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(base_path, 'data', 'data_organized')

train_dataset = datasets.ImageFolder(os.path.join(root, 'train'), transform=train_transform)
test_dataset = datasets.ImageFolder(os.path.join(root, 'test'), transform=test_transform)
val_dataset = datasets.ImageFolder(os.path.join(root, 'val'), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True) #Batchsize 100 to display 100 images later
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#---------------------------Display 100 images----------------------------------------------
images, labels = next(iter(train_loader)) 
plt.figure(figsize=(15, 15))
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(images[i].squeeze(), cmap='gray') 
    plt.title(train_dataset.classes[labels[i]], fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.show()

#----------------------loop dataloader--------------------------------

#-----------------train-------------------
for inputs, outputs in train_loader:
    print("Train:\n")
    print(f"Input Data:\n{inputs}")     
    print(f"Output Labels:\n{outputs}") 
    break #Just to make sure we output once

# ----------------------Test----------------------------
for inputs, outputs in test_loader:
    print("Test:\n")
    print(f"Input Data:\n{inputs}")
    print(f"Output Labels:\n{outputs}")
    break 

#------------------Val-------------------
for inputs, outputs in val_loader:
    print("Val:\n")
    print(f"Input Data:\n{inputs}")
    print(f"Output Labels:\n{outputs}")
    break

#-------------init-----------------
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,1,1)
        self.conv2 = nn.Conv2d(16,32,3,1,1)
        self.conv3 = nn.Conv2d(32,64,3,1,1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 6)
        self.relu = nn.ReLU()


#--------------forward-----------------
    def forward(self, X):
        X = self.relu(self.conv1(X))
        X = self.pool(X)
        X = self.relu(self.conv2(X))
        X = self.pool(X)
        X = self.relu(self.conv3(X))
        X = self.pool(X)
        X = X.flatten(start_dim=1)
        X = self.relu(self.fc1(X))
        output = self.fc2(X)
        return output
    


model = ConvNet()
model.train()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.00001)
EPOCHS = 3

#-------------training loop-------------
for i in range(EPOCHS):  

    for train_x, train_y in train_loader:  
        model.train()
        loss = 0
        pred = model(images)  
        pred = pred.to(torch.float)
        train_y = train_y.to(torch.float)
        loss = loss_function(pred, train_y) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss += loss.item()
 
    avg_loss = loss / len(train_loader) 
    print(f"training loss: {avg_loss} | EPOCHS: {i}")


#-------------testing loop------------------

total_loss = 0
total_preds = 0
total_correct = 0

for x, y in test_loader:
    test_preds = model(x)
    total_loss = loss_function(pred, train_y) 
    total_loss += loss.item()
total_preds += test_preds.item()
avg_loss = total_loss / len(test_loader)
total_correct = loss_function(pred, y)
print(f"avg loss: {avg_loss} | total loss: {total_loss} | total correct: {total_correct} | total preds: {total_preds}")
