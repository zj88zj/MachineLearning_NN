import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, utils
from skimage import io, transform
import io
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.optim as optim
import math
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from skimage.io import imread
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##Parameters
EPOCH = 5
BATCH_SIZE = 1
LR = 0.01
USE_GPU = False
LMBDA = 0.001
##Label data load
labels_df = pd.read_csv('../label.csv')


##Load images, labels and do transform
class FurnitureDataset(Dataset):
   def __init__(self, img_dir, labels_csv_file=None, transform=None):
        self.img_dir = img_dir
        
        if labels_csv_file:
            self.labels_df = pd.read_csv(labels_csv_file)
        else:
            self.images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]

        self.transform = transform

   def __getitem__(self, idx):
        try:
            img_path = os.path.join(
                self.img_dir,
                "{}.jpg".format(self.labels_df.iloc[idx, 1])
            )
        except AttributeError:
            img_path = self.images[idx]
      
        img = imread(img_path) 
        img = gray2rgb(img)

        if self.transform:
            img = self.transform(img) 

        sample = {
            "image": img,
        }
        try:
            sample["label"] = self.labels_df.loc[idx, "label"]
            sample["id"] = self.labels_df.loc[idx, "id"]
        except AttributeError:
            sample["id"] = os.path.basename(self.images[idx]).replace(".jpg", "")
        
        return sample

   def __len__(self):
        try:
            return self.labels_df.shape[0]
        except AttributeError:
            return len(self.images)

transform_pipe = transforms.Compose([
    transforms.ToPILImage(),  # Convert np array to PILImage
    transforms.Resize(
        size=(224, 224)
    ), 
    transforms.ToTensor(), # Convert PIL image to tensor with image values in [0, 1]
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_data = FurnitureDataset(
    img_dir="../trainImage/",
    labels_csv_file="../label.csv",
    transform=transform_pipe
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

#CNN
class CNN(nn.Module):
   def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 224, 224)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=8,             # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (8, 224, 224)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (8, 112, 112)
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 5, 1, 2),      
            nn.ReLU(),                      
            nn.MaxPool2d(2),                # output shape (16, 56, 56)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),    
            nn.ReLU(),                      
            nn.MaxPool2d(2),                # output shape (32, 28, 28)
        )
        self.out = nn.Sequential(
            nn.Linear(32 * 28 * 28, 5),   # fully connected layer, output 5 classes
            nn.Softmax(dim=1)
        )
        
   def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)           # flatten
        output = self.out(x)
        return output    # for visualization can also return x 

model = CNN()
#print(model)

if USE_GPU:
    model = model.cuda()  # Should be called before instantiating optimizer

#optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay= LMBDA) 
optimizer = torch.optim.SGD(model.parameters(), lr =LR, momentum= 0.9, weight_decay= LMBDA)

criterion = nn.CrossEntropyLoss()                 

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

##Training
for i in range(EPOCH):
        model.train()
        samples = 0
        loss_sum = 0
        correct_sum = 0
        for j, batch in enumerate(train_loader):
            X = batch["image"]
            labels = batch["label"]
            if USE_GPU:
                X = X.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y = model(X)
                loss = criterion(y, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()*X.shape[0]
                samples += X.shape[0]
                _, corrects = torch.max(y.data, 1)
                correct_sum += (corrects == labels).sum().item()
               ## Print batch statistics every # batches
        epoch_acc = float(correct_sum) / float(samples)
        epoch_loss = float(loss_sum) / float(samples)
        print("epoch: {} - loss: {} acc: {}".format(i, epoch_loss, epoch_acc))
        
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "resnet50.pth")

##Reconstruction of model      
model1 = CNN()
model1.load_state_dict(torch.load("resnet50.pth"))

##Prediction
test_data = FurnitureDataset(
    img_dir="../testImage/",
    transform=transform_pipe
)
test_loader= DataLoader(
    test_data,
    batch_size=BATCH_SIZE
)
model1.eval()
if USE_GPU:
    model1 = model1.cuda()

ids_all = []
predictions = []

for j, batch in enumerate(test_loader):
    X = batch["image"]
    ids = batch["id"]
    if USE_GPU:
        X = X.cuda()
    
    for _id in ids:
        ids_all.append(_id)

    with torch.set_grad_enabled(False):
        y_pred = model1(X)
        _, indicates = torch.max(y_pred.data, 1)
        predictions.append(indicates)
        

print("Done making predictions!")

##prediction data output
submissions = pd.DataFrame({
    "id": ids_all,
    "label": np.concatenate(predictions).reshape(-1,).astype("int")
}).set_index("id")
submissions.to_csv("./cnnpredictions.csv")