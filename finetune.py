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


# path for mean and standard deviation calculation
path = os.path.join(os.path.curdir,"Data_set2")        #Path for data normalization (actual road data is in Data_set2)

# Calculating Mean and Std Deviation for the images. Needed for Data Normalization.
normalizer = Normalize(path=path, batch_size=64)
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
        transforms.RandomHorizontalFlip(),
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

# data directory path
data_dir = os.path.join(os.path.curdir,"Data")      # actual road data is in Data
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print("class names /:",class_names)
device = torch.device("cuda:0") # if torch.cuda.is_available() else "cpu")
inputs, classes = next(iter(dataloaders['train']))
#out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])
#print(image_datasets['val'].class_to_idx)
def AddNoise(Inputs):
    noise_shape = np.shape(Inputs)

    noise = np.random.normal(0, 0.12, noise_shape)       #np.random.normal(mean, variance, shape)
    noise = torch.from_numpy(noise)
    noise = noise.type(torch.float)
    inputs = Inputs + noise

    return inputs

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    losses_v = []       #validation loss for each epoch
    losses_t = []       #training loss for each epoch
    acc_t = []          #training accuracy for each epoch
    acc_v = []          #validation accuracy for each epoch

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

                #out = torchvision.utils.make_grid(inputs)
                #imshow(out, title=[x for x in labels.cpu().detach().numpy()])

                # if phase == 'train':
                #     inputs = AddNoise(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # to be used only with Binary Cross Entropy loss, Cross Entropy loss implements soft_max
                # activation internally
                #soft_max = nn.Softmax(dim=0)
                #sigmoid = nn.Sigmoid()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs)
                    #outputs = sigmoid(outputs)
                    #print(outputs.shape)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        # print("before backprop :/", model.features[0].weight.grad)    #check .grad attribute of frozen layers
                        loss.backward()
                        # print("after backprop :/", model.features[0].weight.grad)     #check .grad attribute of frozen layers
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
                epoch_loss_t = running_loss / dataset_sizes[phase]      #train loss
                epoch_acc_t = running_corrects.double() / dataset_sizes[phase]

            epoch_loss = running_loss / dataset_sizes[phase]            #loss for both phases but appended loss is validation loss

            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        losses_v.append(epoch_loss)
        losses_t.append(epoch_loss_t)
        acc_v.append(epoch_acc.detach().cpu().tolist())             #t.detach().cpu().numpy()
        acc_t.append(epoch_acc_t.detach().cpu().tolist())
        print()

    plt.figure(1)
    plt.title("Loss Curve 1.0", fontsize=16)
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.plot(losses_v, label="Val loss")
    plt.plot(losses_t, label ="Train Loss")
    plt.legend(loc="upper right")

    plt.figure(2)
    plt.title("Accuracy 1.0", fontsize=16)
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("accuracy", fontsize=12)
    plt.plot(acc_v, label="Val accuaracy")
    plt.plot(acc_t, label="Train accuaracy")
    plt.legend(loc="lower right")
    plt.show()
    #print("training accuracy /:", acc_t)
    #print("Validation accuracy /:", acc_v)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = torchvision.models.vgg16(pretrained=True)
print(model)

for name, param in model.named_parameters():
     if param.requires_grad and "features" in name:
         if name == "features.26.weight" or name == "features.26.bias" or name == "features.28.weight" or name == "features.28.bias":
            param.requires_grad = True
         else:
            param.requires_grad = False

#print(model.features[28])

model.classifier[6]= nn.Linear(4096, 2)
#nn.init.xavier_uniform_(model.classifier[6].weight)
model.classifier[6].weight.data.fill_(0.0001)
model.classifier[6].bias.data.fill_(0.0001)
#print(model)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
    else:
        print(f"requires_grad on {name} is false")
#print(model.state_dict())
#print(model.features[26].parameters())
model_conv = model.to(device)

#criterion = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model_conv.parameters(), lr=0.00001, momentum=0.9)

optimizer = optim.SGD(
    [
        {"params": model.features[26].parameters()},
        {"params": model.features[28].parameters()},
        {"params": model.classifier[0].parameters()},
        {"params": model.classifier[3].parameters()},
        {"params": model.classifier[6].parameters(), "lr": 0.001},
    ], lr = 0.00001, momentum=0.9
)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer,
                         exp_lr_scheduler, num_epochs=130)

if not os.path.exists("Weights"):
    os.makedirs("Weights")
# Saving the model
save_path = "Weights/model_2.pth"
torch.save(model.state_dict(), save_path)


