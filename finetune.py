import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from sklearn import svm
from sklearn import metrics
import tqdm
from normalize import Normalize
import torchextractor as tx
import time
import copy
import torch.optim as optim
from torch.optim import lr_scheduler

path = os.path.join(os.path.curdir,"Data_set2")        #"Data/train"

# Calculating Mean and Std Diviation for the images. Needed for Data Normalization.
normalizer = Normalize(path=path, batch_size=10)
loaded_data = normalizer.data_load()
mean, std= normalizer.batch_mean_and_sd(loaded_data)
print("mean and std: \n", mean, std)


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# transforms on training and validation datasets
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = os.path.join(os.path.curdir,"Data")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda:0") # if torch.cuda.is_available() else "cpu")
#print(class_names)
inputs, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])
#print(image_datasets['val'].class_to_idx)

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print(phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #itr = 0
            for inputs, labels in dataloaders[phase]:

                out = torchvision.utils.make_grid(inputs)
                #imshow(out, title=[x for x in labels.cpu().detach().numpy()])

            #     print(type(inputs))
            #for inputs, _ in dataloaders[phase]:
                #print(inputs.size()[0])
                # tensor1 = torch.ones(1, inputs.size()[0])
                # tensor2 = torch.zeros(1, inputs.size()[0])
                # labels = torch.cat(tensors=[tensor1, tensor2])
                #
                # labels = torch.transpose(labels, 0, 1)
                #print(labels)
                #labels = labels.type(torch.float32)
                #labels = torch.unsqueeze(labels,1)
                #print(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)
                soft_max = nn.Softmax(dim=0)
                #sigmoid = nn.Sigmoid()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs)
                    outputs = soft_max(outputs)
                    #print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    #print(preds)

                    loss = criterion(outputs, labels)
                    #print('loss is: ',loss)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                #print(torch.sum(preds == labels.data))
                running_loss += loss.item() * inputs.size(0)
                #print(running_loss)
                running_corrects += torch.sum(preds == labels.data)
                #running_corrects += torch.sum(torch.tensor(preds, dtype=torch.float32) == labels.data[:, 0].cpu())
                #print(dataset_sizes[phase])

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #     else:
    #         print("requires_grad the layer is set to false")

    #print(best_model_wts)
    #print(model)
    return model

model = torchvision.models.vgg16(pretrained=True)
#print(model)

for name, param in model.named_parameters():
     if param.requires_grad and 'features' in name:
        param.requires_grad = False

model.classifier[6]= nn.Linear(4096, 2)
#print(model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
    else:
        print("requires_grad on this layer is set to false")
#print(model.state_dict())
model_conv = model.to(device)

#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer = optim.SGD(model_conv.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=2)


