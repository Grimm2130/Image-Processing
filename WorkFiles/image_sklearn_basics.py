import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from skimage import data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pmg

elephantPath = "/Users/chimaochiagha/opencv/Intermediate_Image_processing/00_Introduction/image_data/elephant.png"
# read in the image
elephant_raw = tf.io.read_file(elephantPath)
# decode the file format
elephant = tf.io.decode_png(elephant_raw).numpy()
print(f"Info on elephant:\n"
      f"Shape: {elephant.shape}\n"
      f"Size: {elephant.size}")

print("Converting the image to a grey scale image")

# get the max values in the image
maxPixVal = np.max(elephant, axis=0 )
elephant[elephant >= (maxPixVal/2)] = 225
elephant[elephant < (maxPixVal/2)] = 0

elephant = cv.cvtColor(elephant, cv.COLOR_RGB2BGR)
cv.imshow("Elephant", elephant)
cv.waitKey(0)