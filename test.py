import json
import re
import base64
import io
import cv2
import pandas as pd
import numpy as np
from PIL import Image

with open("new_1.json", 'r') as f:
    data = json.loads(f.read())
    frames_list = pd.json_normalize(data, record_path=['markers'])
    loop = len(frames_list)
    for i in range(loop):
        image_string = frames_list['image'][i]
        result = re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", image_string, re.DOTALL)
        if result:
            ext = result.groupdict().get("ext")
            data = result.groupdict().get("data")
        else:
            raise Exception("Do not parse!")
        imgdata = base64.b64decode(str(data))
        image = Image.open(io.BytesIO(imgdata))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        #print(width,height)
        #frame = torch.unsqueeze(frame,0)
        #Imshow(frame)
        #frame.show()
        print('working')
        cv2.imshow("Image",np.array(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()