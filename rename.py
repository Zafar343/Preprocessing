# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():
    #folder = "./../../img/manual_roaddata/vid_8"
    folder = os.path.join(os.path.curdir,"Data/train/class_1")
    #a = 150
    for count, filename in enumerate(os.listdir(folder)):

        dst1 = f"h_{str(count)}.jpg"         #use count for nameing file from zero onwards
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst1}"
        #a += 1
        # rename() function will
        # rename all the files
        if filename != dst1:
            os.rename(src, dst)
        else:
            continue

print("done")

# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()