# in this code the sectioning and sub-sectioning of the pavement branches is implemented
import json
import os
import re
import base64
import io
import cv2
import pandas as pd
import numpy as np
from math import radians, sin, cos, asin, sqrt
from PIL import Image
import statistics


def haversine_distance(first, second):
    # The first coordinate in a gps point is latitude (N), and second one is the longitude (E)
    # Convert the degree to radians
    lat1 = radians(first[0])
    lat2 = radians(second[0])
    lon1 = radians(first[1])
    lon2 = radians(second[1])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    c = 2 * asin(sqrt(a))
    # Radius of earth in kilometers (6371). Use 3956 for miles
    r = 6371
    # calculate the result
    return (c * r * 1000)

with open("./../new_1.json", 'r') as f:
    data = json.loads(f.read())
    branch_full_path = data['full_path']

# print(branch_full_path)
# print(type(branch_full_path))
branch_full_path = re.sub("\(", '', branch_full_path)
branch_full_path = re.sub("\)", '', branch_full_path)
branch_full_path = re.sub("\s", '', branch_full_path)
branch_full_path_list = branch_full_path.split(',')
branch_full_path_list = list(map(lambda x: float(x), branch_full_path_list))
branch_full_path_numpy = np.array(branch_full_path_list)
branch_full_path_numpy = branch_full_path_numpy.reshape(-1,2)
# print(branch_full_path_numpy)
# print(branch_full_path_numpy.shape)
# print(branch_full_path_numpy[0][1])


print ("Haversine Distance of GPS Points: ")
dist = []
for i in range(len(branch_full_path_numpy)):
    if i < len(branch_full_path_numpy)-1:
        #haversine = haversine_distance(first, second)
        haversine = haversine_distance(branch_full_path_numpy[i], branch_full_path_numpy[i+1])
        dist.append(haversine)
        print(f'Distance between {branch_full_path_numpy[i]} and {branch_full_path_numpy[i+1]} = ',haversine)
print("Mean //: ",statistics.mean(dist))
print("Variance //: ",statistics.variance(dist))
