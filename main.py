# This is a sample Python script.
import os.path
import time

import numpy as np
import cv2
from WorkFiles.open_cv_videos_and_images import readVideo, \
    writeToFrames, detect_green, getPos, get_frame_difference, \
    save_video_frames, find_Lanes
from WorkFiles.making_path_from_green import filter_around_green, read_this_Video

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "/Users/chimaochiagha/opencv/Intermediate_Image_processing/my_computer_vision/image_data/SeniorDesign.mp4"
    # detect_green(path)
    # getPos()
    path1 = "/Users/chimaochiagha/opencv/Intermediate_Image_processing/my_computer_vision/output_data/green_path_10.png"
    # path2 = "/Users/chimaochiagha/opencv/Intermediate_Image_processing/00_Introduction/output_data/90.png"
    # frame1, frame2 = cv2.imread(path1), cv2.imread(path2)
    #
    # diff = get_frame_difference(frame1, frame2)
    # cv2.imshow("Image", diff)
    # cv2.waitKey(0)

    read_this_Video(path)
    # save_video_frames("green_path_", videoPath=path)


    # image = cv2.imread(path1)
    # image = filter_around_green(image)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)