import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import time
import copy
from sklearn import svm
import cv2
import tqdm
from PIL import Image
import pandas as pd
import json
import re
import io
import base64
import pickle


mean=np.array([0.47582975, 0.4897879, 0.4882829])
std=np.array([0.2630566, 0.2686199, 0.28436255])

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
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])
    device = torch.device("cuda:0")  # setting device to cuda
    return model,transform, device


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        #self.classifier = list(model.classifier[0:4])
        # Extract VGG-16 Average Pooling Layer
        self.avgpool = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        #self.classifier = nn.Sequential(*self.classifier)
        self.fc1 = model.classifier[0]
        # self.ReLU = model.classifier[1]
        # self.Dropout = model.classifier[2]
        # self.fc2 = model.classifier[3]



    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        #print(out.shape)
        out = self.avgpool(out)
        out = self.flatten(out)

        out = self.fc1(out)
        # out = self.ReLU(out)
        # out = self.Dropout(out)
        # out = self.fc2(out)
        return out

torch.manual_seed(17)
model,transform,device = blueprint()

# model = torchvision.models.vgg16(pretrained=True)
print(model)

# transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=mean,
#             std=std
#         )
#     ])
# device = torch.device("cuda:0")  # setting device to cuda

F_extractor = FeatureExtractor(model)
print(F_extractor)
for name, param in F_extractor.named_parameters():
     if param.requires_grad: #and 'features' in name:
         param.requires_grad = False

# for name, param in F_extractor.named_parameters():
#     if param.requires_grad:
#         print(name)
#     else:
#         print(f"requires_grad on {name} is false")

F_extractor.to(device)
path = os.path.join(os.path.curdir,"Data/svm_train")
features = []
for filename in tqdm.tqdm(os.listdir(path)):
        img = cv2.imread(os.path.join(path,filename)) # image fetching
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # .fromarray(img): converts np array to pillow image
        img = transform(img)
        img = torch.unsqueeze(img,0)
        img = img.to(device)
        # We only need extracted features, so we don't need to compute the gradient
        with torch.no_grad():
            feature = F_extractor(img)
            #print(f'feature shape{feature.shape}: \n',feature)


        # Convert to NumPy Array, Reshape it, and save it to features list
        features.append(feature.cpu().detach().numpy().reshape(-1))
# type casting features list to np array as SVM needs a np array
features = np.array(features)
print(features.shape)

ocsvm_classifier = svm.OneClassSVM(kernel='rbf', gamma='scale' , nu=0.08) # nu: describes classification line. Value is: b/w 0 and 1
ocsvm_classifier.fit(features)
print("OCSVM is trained")

#Prediction phase
test_score = []
# id = []
# path = os.path.join(os.path.curdir,"Data/Test")
# for filename in tqdm.tqdm(os.listdir(path)):
#     _id = int(filename.split('.')[0])
#     img = cv2.imread(os.path.join(path,filename))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     Pil_img = Image.fromarray(img)
#     Pil_img = transform(Pil_img)
#     #print(img.size())
#     Pil_img = torch.unsqueeze(Pil_img,0)
#     Pil_img = Pil_img.to(device)
#     # We only extract features, so we don't need gradient
#     with torch.no_grad():
#         # Extract the feature from the image
#         feature = F_extractor(Pil_img)
#     # Convert to NumPy Array, Reshape it, and save it to features variable
#     feature = feature.cpu().detach().numpy().reshape(-1)
#         #cv2.destroyAllWindows()
#     feature = np.array(feature)
#     # print(feature.shape)
#     # ----------- Feature Extraction Complete
#
#     # One Class Calssification
#     score = ocsvm_classifier.decision_function([feature])
#     #print(score)
#     test_score.append(score[0])
#     id.append(_id)
# #print(len(test_score))
# # making a data frame containing image id and prediction result
# res = pd.DataFrame({
#     'id': id,
#     'test_score': test_score
# })
#
# res.sort_values(by='id', inplace=True)
# res.reset_index(drop=True, inplace=True)
# res.to_csv("OCSVM_scoresfc1.csv")
# print(res)
with open("out_0.json", 'r') as f:
    data = json.loads(f.read())
    frames_list = pd.json_normalize(data, record_path=['frames'])
    loop = len(frames_list)
    for i in range(loop):
        image_string = frames_list['frame'][i]
        result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", image_string, re.DOTALL)
        if result:
            ext = result.groupdict().get("ext")
            data = result.groupdict().get("data")
        else:
            raise Exception("Do not parse!")
        imgdata = base64.b64decode(str(data))
        image = Image.open(io.BytesIO(imgdata))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        Pil_img = transform(frame)
        Pil_img = torch.unsqueeze(Pil_img, 0)
        Pil_img = Pil_img.to(device)
        with torch.no_grad():
            feature = F_extractor(Pil_img)
        feature = feature.cpu().detach().numpy().reshape(-1)
        feature = np.array(feature)
        score = ocsvm_classifier.decision_function([feature])
        print(score)
        plt.pause(3)