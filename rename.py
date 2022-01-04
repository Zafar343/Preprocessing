# Pythono3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():
    # folder = "./../../../Downloads/xyz"
    folder = "./../../img/perfect"
    for count, filename in enumerate(os.listdir(folder)):
        dst1 = f"perfect_{str(count)}.png"
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst1}"

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