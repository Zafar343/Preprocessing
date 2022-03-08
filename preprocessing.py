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


class FeatureExtractor(nn.Module):
    """Builds Feature extractor model based on Finetuned model"""
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

class Model():
    """Class to implement One Class SVM by using features extracted from finetuned VGG16"""
    def blueprint(self):
        '''function to make blueprint model based on VGG16 and load trained weights'''

        print("///:Building model blueprint and loading trained weights")
        model = torchvision.models.vgg16()  # initializing the model with random weights (this is our model blueprint)
        model.classifier[6] = nn.Linear(4096, 2)  # changing blueprint final layer according to oure requirement
        print(model)
        model.load_state_dict(
            torch.load("Weights/model_best.pth"))  # loading our trained weights and biases into the blueprint model
        return model

    def extractor(self):
        model = self.blueprint()
        # print(model)
        F_extractor = FeatureExtractor(model)
        #print(F_extractor)
        for name, param in F_extractor.named_parameters():
            if param.requires_grad:  # and 'features' in name:
                param.requires_grad = False

        classifier = pickle.load(open("Weights/OCSVM_model.sav", 'rb'))
        device = torch.device("cuda:0")  # setting device to cuda
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=np.array([0.47582975, 0.4897879, 0.4882829]),
                std=np.array([0.2630566, 0.2686199, 0.28436255])
            )
        ])
        return classifier, F_extractor, transform, device

    def classification(self, img, F_extractor, device, transform, classifier):
        F_extractor.to(device)
        print(F_extractor)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        with torch.no_grad():
        # Extract the feature from the image
            feature = F_extractor(img)
        # Convert to NumPy Array, Reshape it, and save it to features variable
        feature = feature.cpu().detach().numpy().reshape(-1)
        feature = np.array(feature)
        # One Class Classification
        score = classifier.decision_function([feature])

        return score


# instance = Model()
# #extractor_model, transform, device = instance.extractor()
# #print(extractor_model)
# # print(transform)
# # print(device)
# score = instance.classifier(path= os.path.join(os.path.curdir,"Data/Test"))
# print(score)

