# script to visualize the labels allong with images for confirmation
import pandas as pd
#from PIL import Image
import cv2 as cv
import os
import PyQt5
from matplotlib import pyplot as plt


df = pd.read_csv("labels.csv")
df.reset_index(drop=True, inplace=True)
print(df)
path = os.path.join(os.path.curdir, "Data/Test")
for filename in os.listdir(path):
    id_ = int(filename.split('.')[0])
    #print(df.iat[id_-1,1])
    img = cv.imread(os.path.join(path, filename))
    #print("Current Image:/ ",id_)
    image_label = "Image_id="+str(id_)+" /  label="+str(df.iat[id_-1,1])
    cv.namedWindow(image_label, cv.WINDOW_NORMAL)
    cv.imshow(image_label, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


