# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
cam = cv2.VideoCapture("./../../img/1.mp4")

try:

    # creating a folder named data
    if not os.path.exists('./../../img/vid_1'):
        os.makedirs('./../../img/vid_1')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0
#ret, frame = cam.read()
#ret = True
i = 0
frame_skip = 10

while (True):

    # reading from frame
    #cam.set(cv2.CAP_PROP_POS_MSEC, (currentframe * 1000))       #read one frame every one second
    ret, frame = cam.read()

    if ret:
        if i > frame_skip - 1:

        # if video is still left continue creating images
            name = './../../img/vid_1/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

        # writing the extracted images
            cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
            currentframe += 1
            i = 0
            continue
        i += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()