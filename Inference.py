import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os, random
from PIL import Image
import tqdm
import torch.nn.functional as F
import pandas as pd
from normalize import Normalization

normalizer = Normalization(path=os.path.join(os.path.curdir, "Data_set2"), batch_size=32)
loaded_data = normalizer.data_load()
mean, std = normalizer.batch_mean_and_sd(loaded_data)


def Imshow(inp):

    """Imshow for Tensor."""

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

def blueprint():

    '''function to make blueprint model based on VGG16 and load trained weights'''

    print("///:Building model blueprint and loading trained weights")
    model = torchvision.models.vgg16()  # initializing the model with random weights (this is our model blueprint)
    model.classifier[6] = nn.Linear(4096, 2)  # changing blueprint final layer according to oure requirement
    # print(model)
    model.load_state_dict(
        torch.load("Weights/model_best.pth"))  # loading our trained weights and biases into the blueprint model
    return model


def infer(img, model):

    '''function to perform inference on an input image'''

    torch.manual_seed(17)           #for reproduceability
    # Preprocessing
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean = mean,
            std = std
            )
        ])
    device = torch.device("cuda:0")         #setting device to cuda
    model.to(device)  # setting model to GPU

    img = transform(img)
    img = img.unsqueeze(0)
    # print(img.shape)
    img = img.to(device)
    model.eval()  # setting the model to evaluation mode
    outputs = model(img)  # inference
    preds = F.softmax(outputs, dim=1)[:, 1].tolist()
    return preds


#____________________________Predictions_____________________________________________________________________
#inference starts here
model = blueprint()
id_list = []        #list to store image id to be used at later stage
pred_list = []      #list to store the pridictions of model on test data to be used for actual classification and visualization
path = os.path.join(os.path.curdir,"Data/Test")     #path to test set
#infering on each image
for filename in tqdm.tqdm(os.listdir(path)):
        img = Image.open(os.path.join(path,filename))   #.convert('RGB')
        _id = int(filename.split('.')[0])
        #print(_id)
        preds = infer(img=img, model=model)
        #print(preds)
        id_list.append(_id)
        pred_list.append(preds[0])
# making a data frame containing image id and prediction result
res = pd.DataFrame({
    'id': id_list,
    'label': pred_list
})

res.sort_values(by='id', inplace=True)
res.reset_index(drop=True, inplace=True)
#res.to_csv("predictions.csv")
#print(res)

#________________Visualize Prediction_____________________________________________
id_list = []
class_ = {0: 'Valid Image', 1: 'Invalid Image'}     #our actual classes

fig, axes = plt.subplots(3, 6, figsize=(10, 10), facecolor='w')

for ax in axes.ravel():

    i = random.choice(res['id'].values)

    label = res.loc[res['id'] == i, 'label'].values[0]
    if label > 0.5:             # label 1 is for invalid class as we are storing coulmn 1 which corresponds to invalid class
        label = 1
    else:
        label = 0

    img_path = os.path.join(path, '{}.jpg'.format(i))
    img = Image.open(img_path)

    ax.set_title(class_[label])
    ax.imshow(img)
    plt.pause(1)
plt.show()