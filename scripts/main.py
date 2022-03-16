import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from sklearn import svm
from sklearn import metrics
import tqdm
from normalize import Normalization
import torchextractor as tx

path = os.path.join(os.path.curdir,"Data/train")

# Calculating Mean and Std Diviation for the images. Needed for Data Normalization.
normalizer = Normalization(path=path, batch_size=10)
loaded_data = normalizer.data_load()
mean, std= normalizer.batch_mean_and_sd(loaded_data)
print("mean and std: \n", mean, std)


# preprocessor: for image resizing,
torch.manual_seed(17)
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean= mean,
std= std
)])

def imshow(inp, title=None):
    """Image: Tensor to Numpy and Shows using matplotlib"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)  # pause a bit so that plots are updated


model = models.vgg16(pretrained=True)
print("VGG16 Imported Model \n",model)

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        # Extract VGG-16 Feature Layers
        self.features = list(model.features)
        self.features = nn.Sequential(*self.features)
        # Extract VGG-16 Average Pooling Layer
        self.pooling = model.avgpool
        # Convert the image into one-dimensional vector
        self.flatten = nn.Flatten()
        # Extract the first part of fully-connected layer from VGG16
        self.fc1 = model.classifier[0]
        self.ReLU = model.classifier[1]
        self.Dropout = model.classifier[2]
        self.fc2 = model.classifier[3]



    def forward(self, x):
        # It will take the input 'x' until it returns the feature vector called 'out'
        out = self.features(x)
        out = self.pooling(out)
        #out = self.classifier(out)
        out = self.flatten(out)
        #out = self.classifier(out)
        out = self.fc1(out)
        out = self.ReLU(out)
        out = self.Dropout(out)
        out = self.fc2(out)
        return out



# Initialize the model
# model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model) # creating new model from newly created feature extractor above
print("New Model based on VGG16: \n",new_model)


# Change the device to GPU
# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device = torch.device("cuda:0")


new_model = new_model.to(device)
# print(new_model)


# Freezing weights of feature extracting layers of VGG16 in the feature extractor above
# to avoid backpropagation
#----------------------------------------------------------------------------------------
for name, param in new_model.named_parameters():
     if param.requires_grad: #and 'features' in name:
         param.requires_grad = False
#     elif param.requires_grad and 'fc1' in name:
#         param.requires_grad = False
#     elif param.requires_grad and 'fc2' in name:
#         param.requires_grad = False
#________________________________________________________________________________________




# printing parameters (weights and biases) of each layer - gradient flag appears if grad is set True for a layer
# for para in new_model.parameters():
#     print(para)

# Checking the layers for which gradient calculation is set to ON/True
for name, param in new_model.named_parameters():
    if param.requires_grad:
        print(name)
    else:
        print('gradients computations are off for this layer')



path = os.path.join(os.path.curdir,"Data/train/1")
# Now performing feature extraction on our data
features = [] # List of feature vectors of all images
# window_name = 'image'
for filename in tqdm.tqdm(os.listdir(path)):
        img = cv2.imread(os.path.join(path,filename)) # image fetching
        # Preprocessor needs PIL image as input. Pil img is in RGB format whereas
        # cv2 image is in BGR. Converting cv2 BGR image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)  # .fromarray(img): converts np array to pillow image
        img = preprocess(img)
        # imshow(img)

        # reshaping the image
        # (Batch_Size, Channels, Height, Width)
        img = img.reshape(1, 3, 224, 224)
        img = img.to(device)

        # Extracting features from the image: feature = new_model(img)
        # We only need extracted features, so we don't need to compute the gradient
        with torch.no_grad():
            feature = new_model(img)
            #print(f'feature shape{feature.shape}: \n',feature)


        # Convert to NumPy Array, Reshape it, and save it to features list
        # feature.cpu().detach().numpy().reshape(-1): "feature" is in tensor form. To convert it into np array, it's detached and keep the shape as it is
        # Tensor.cpu() Returns a copy of this object (tensor) in CPU memory
        features.append(feature.cpu().detach().numpy().reshape(-1))

# type casting features list to np array as SVM needs a np array
features = np.array(features)
print(features.shape) # (num_of_images x 4096)

# Instantiating OCSVM and TRAINING
ocsvm_classifier = svm.OneClassSVM(kernel='sigmoid', gamma='scale' , nu=0.2) # nu: describes classification line. Value is: b/w 0 and 1
ocsvm_classifier.fit(features)
print("OCSVM is trained")


# Testing
test_score = []
vals = []
path = "./../../img/actual_imgs"
for filename in tqdm.tqdm(os.listdir(path)):
    img = cv2.imread(os.path.join(path,filename))
    #print(img.shape)
    #cv2.imshow('',img)     #cv2_imshow(img)     #cv2.imshow does'nt work in colab
    #img = preprocess(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = preprocess(img)
    imshow(img)

    #print(img.size())
    img = img.reshape(1, 3, 224, 224)
    img = img.to(device)
    # We only extract features, so we don't need gradient
    with torch.no_grad():
        # Extract the feature from the image
        feature = new_model(img)
    # Convert to NumPy Array, Reshape it, and save it to features variable
    feature = feature.cpu().detach().numpy().reshape(-1)
        #cv2.destroyAllWindows()
    feature = np.array(feature)
    # print(feature.shape)
    # ----------- Feature Extraction Complete

    # One Class Calssification
    score = ocsvm_classifier.decision_function([feature])
    #print(score)

    vals.append(ocsvm_classifier.predict([feature])[0])
    test_score.append(score[0])

print(vals)
print("total misclassifieds: ",vals.count(-1))
print("Misclassification percentage: ", (vals.count(-1)/len(os.listdir(path)))*100) #
print("Accuracy: ",100 - ((vals.count(-1)/len(os.listdir(path)))*100))

# fpr, tpr, thresholds = metrics.roc_curve(test_label, test_score)

# area_under_curve = metrics.auc(fpr, tpr)
# print(area_under_curve)