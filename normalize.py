from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms
import os
import tqdm


class Normalize:
    """
        This class calculates the Mean and Standard Deviation for all images in a dataset
    """

    def __init__(self, path, batch_size):
        """ constructor for class: Normalize"""
        self.path = path
        self.batch_size = batch_size

    def data_load(self):
        """
            This function loads data in batches from the given path location and returns the loaded data
        """
        # transforming the images: resizing, center-cropping and making Tensor
        transform_img = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        # dataset object initialized
        data_set = torchvision.datasets.ImageFolder(root=self.path, transform=transform_img)
        # data loading
        loaded_data = DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )
        return  loaded_data

    def batch_mean_and_sd(self, data):
        """
            This function computes Mean and Standard Deviation of image data passed to it and returns the Mean and Std values
        """
        cnt = 0  # count on pixels of images in a batch
        fst_moment = torch.empty(3)  # mean
        snd_moment = torch.empty(3)  # variance (std is calculated from variance)
        i = 1
        for images, _ in tqdm.tqdm(data):
            b, c, h, w = images.shape  # b is batch size, c is channels in an image, h is height of an image, w is width of an image
            nb_pixels = b * h * w  # total number of pixels in a batch
            sum_ = torch.sum(images, dim=[0, 2, 3])  # calculating sum for pixels in a batch
            sum_of_square = torch.sum(images ** 2,
                                      dim=[0, 2, 3])  # calculating square of sum
            # formulation to calculate a normalized (normalized over total number of pixels in cnt each time) first and second moment for all the batches in loader
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)  # calculating first moment
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)  # calculating second moment
            # print(cnt)
            # print(i)
            i += 1
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
        mean = mean.numpy()     #.tolist()
        std = std.numpy()       #.tolist()
        return mean, std


#
# path = os.path.join(os.path.curdir,"Data/train")
# print(path)
# normalizer = Normalize(path=path, batch_size=10)
# loaded_data = normalizer.data_load()
# mean, std = normalizer.batch_mean_and_sd(data=loaded_data)
# print(mean, std)
