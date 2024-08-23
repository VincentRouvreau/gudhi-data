import argparse
import os
import numpy as np
from PIL import Image
from skimage.morphology import convex_hull_image

#
# This program requires to load data from https://sourceforge.net/projects/flavia/files/Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download
# This dataset is based on the paper A Leaf Recognition Algorithm for Plant classification Using Probabilistic Neural Network, by
# Stephen Gang Wu, Forrest Sheng Bao, Eric You Xu, Yu-Xuan Wang, Yi-Fan Chang and Qiao-Liang Xiang, published at
# IEEE 7th International Symposium on Signal Processing and Information Technology, Dec. 2007.
#
# Unzip the data and copy data_selecta.py in a same directory data folder
# Launch 'python data_selecta.py -s /my/path/to/flavia'
# Generated with Python 3.8 / Pandas 1.1.0
#

parser = argparse.ArgumentParser(
    description="Reads all the images from a source_directory, converts them to (image_size x image_size) black &"
    " white images, computes their convexity. The final numpy ndarray (number of images x (image_size x image_size + 1)"
    " is saved in a destination_file."
)
parser.add_argument("-s", "--source_directory", type=str, default="flavia")
parser.add_argument("-d", "--destination_file", type=str, default="flavia_convexity.npy")
parser.add_argument("-i", "--image_size", type=int, default=30)

args = parser.parse_args()

data = []
for image in sorted(os.listdir(args.source_directory)):
    if image.endswith('.jpg'):
        file_name = args.source_directory + "/" + image
        # print(file_name)
        img = Image.open(args.source_directory + "/" + image)
        # Convert image in gray scale and resize
        img = img.convert('L').resize((args.image_size,args.image_size), Image.LANCZOS)
        # Black & white image at a specific threshold
        img_bw = img < 0.9 * np.max(img)
        img_bw_ch = convex_hull_image(img_bw, offset_coordinates = False)
        convexity = np.sum(img_bw) / np.sum(img_bw_ch)
        data_line = np.append(img_bw.astype(float).reshape(args.image_size * args.image_size), convexity)
        data.append(data_line)

data = np.asarray(data)
np.save(args.destination_file, data, allow_pickle=False)
