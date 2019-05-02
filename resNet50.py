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
EPOCH = 10
BATCH_SIZE = 20
LR = 0.01
#LAMBDA = 20 regularization parameter
USE_GPU = False

##Label data load
labels_df = pd.read_csv('../label.csv')
# labels_df.head()
# labels_df["label"].value_counts().plot(kind="pie")
# plt.show()
train_indices, test_indices = train_test_split(labels_df.index, test_size=0.25) #0.75 for training 0.25 for test
#train_indices.shape, test_indices.shape

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

        #print("img_path:", img_path)      
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
    ),   # Resize image to 224 x 224 as required by most vision models
    transforms.ToTensor(), # Convert PIL image to tensor with image values in [0, 1]
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_data = FurnitureDataset(
    img_dir="../trainImage/",
    labels_csv_file="../label.csv",
    transform=transform_pipe
)

## plot one example
# print(train_data.train_data.size())                 
# print(train_data.labels_df.size())              
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.labels_df[0])
# plt.show()

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    sampler=torch.utils.data.SubsetRandomSampler(
        train_indices
    )
#     num_workers = 5
)

test_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    sampler=torch.utils.data.SubsetRandomSampler(
        test_indices
    )
#     num_workers = 5
)

dataloaders = {
    "train": train_loader,
    "test": test_loader
}

##Rresnet50
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(
        in_features=2048,
        out_features=5
    ),
    nn.Softmax(dim=1)
)
# print(model)
# out = model(train_data[0]["image"].view(1, 3, 224, 224))
# print(out.shape)

if USE_GPU:
    model = model.cuda()  # Should be called before instantiating optimizer

##Optimizer
#optimizer = torch.optim.Adam(model.parameters(), lr=LR) 
optimizer = torch.optim.SGD(model.parameters(), lr =LR, momentum= 0.9)

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()     
#criterion = nn.BCELoss()             

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

##Training
for i in range(EPOCH):
    for phase in ["train","test"]:
        if phase == "train":
           model.train()
        else:
           model.eval()

        samples = 0
        loss_sum = 0
        correct_sum = 0
        for j, batch in enumerate(dataloaders[phase]):
            X = batch["image"]
            labels = batch["label"]
            if USE_GPU:
                X = X.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                #for name, param in model.named_parameters():
	            #    print(name, '      ', param.size(), param)
                y = model(X)
                loss = criterion(y, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item()*X.shape[0]
                samples += X.shape[0]
                _, corrects = torch.max(y.data, 1)
                correct_sum += (corrects == labels).sum().item()
                
                ## Print batch statistics every # batches
                if j % 1 == 0 and phase == "train":
                    print("{}:{} - loss: {}, acc: {}".format(
                        i + 1, 
                        j + 1, 
                        float(loss_sum) / float(samples), 
                        float(correct_sum) / float(samples)
                    ))
        # Print epoch statistics
        epoch_acc = float(correct_sum) / float(samples)
        epoch_loss = float(loss_sum) / float(samples)
        print("epoch: {} - {} loss: {}, {} acc: {}".format(i + 1, phase, epoch_loss, phase, epoch_acc))
        
        # Deep copy the model
        if phase == "test" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, "resnet50.pth")

##Reconstruction of model      
model1 = torchvision.models.resnet50(pretrained=True)
model1.fc = torch.nn.Sequential(
    torch.nn.Linear(
        in_features=2048,
        out_features=5
    ),
    nn.Softmax(dim=1)
)
model1.load_state_dict(torch.load("resnet50.pth"))

##Prediction
test_data = FurnitureDataset(
    img_dir="../testImage/",
    transform=transform_pipe
)
test_loader1 = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=8
)

model1.eval()
if USE_GPU:
    model1 = model1.cuda()

ids_all = []
predictions = []

for j, batch in enumerate(test_loader1):
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
submissions.to_csv("./submissions.csv")
