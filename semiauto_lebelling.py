#script to label test images. These labels will be used in model evaluation on unseen data
import tqdm
import os
import pandas as pd
from PIL import Image


id = []
labels = []
path = os.path.join(os.path.curdir, "Data/Test")


for filename in tqdm.tqdm(os.listdir(path)):
    id_ = int(filename.split('.')[0])
    img = Image.open(os.path.join(path, filename))
    img.show(img)
    print("image_id is/: ", id_)
    label = input("Enter label (0 for valid or 1 for invalid) for the image you seen just before/: ")
    id.append(id_)
    labels.append(int(label))

df = pd.DataFrame({
    'id': id,
    'label': labels
})

df.sort_values(by='id', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)
df.to_csv("labels.csv")

