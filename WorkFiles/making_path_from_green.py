import cv2
import numpy as np

# Generate the path to video path
def read_this_Video(video_path):
    camera = cv2.VideoCapture(video_path)
    # validate Video Stream
    # ret, prevFrame = camera.read()
    while camera.isOpened():
        if cv2.waitKey(10) == 27:
            break
        ret, currframe = camera.read()
        if ret is True and int(camera.get(cv2.CAP_PROP_POS_FRAMES))%10 == 0:
            # diff = get_frame_difference(prevFrame, currframe)
            image = segment_for_path(filter_around_green(currframe))
            cv2.imshow("Lane", image)
            # update previous frame
            # prevFrame = currframe


def filter_around_green(currImage):
    '''
    Function which returns an image with the rio (area between green segments) filtered for
    :param currImage: The frame/image being considered
    :return: filtered image
    '''
    currImage = cv2.cvtColor(currImage, cv2.COLOR_BGR2HSV)

    # convert RGB to green
    greenRGB = np.array([[[0, 255, 0]]], dtype=np.uint8)
    greenHSV = cv2.cvtColor(greenRGB, cv2.COLOR_BGR2HSV)

    # define the range for green
    hue = greenHSV[0, 0, 0]
    lower_hue = int(hue - 30)
    upper_hue = int(hue + 30)

    # kernel for erosion and dilation
    kernel = np.ones((5, 5), dtype=np.uint8)

    # get as mask of the green region in the image
    mask_green = cv2.inRange(currImage, (lower_hue, 0, 0), (upper_hue, 255, 255))

    # Erode
    mask_green = cv2.erode(mask_green, kernel, iterations=5)

    # Dilate
    mask_green = cv2.dilate(mask_green, kernel, iterations=1)

    # threshold the image for inversion
    _, thres = cv2.threshold(mask_green, 20, 255, cv2.THRESH_BINARY_INV)

    # erode the image, to smoothen the balck areas
    eroded = cv2.erode(thres, kernel, iterations=1)

    return eroded

def segment_for_path(green_filtered_image):
    '''
    Function takes in the filtered image and defines boundaries around the region of interest
    :param green_filtered_image: Image that has been filted for a path around the green
    :return:
    '''

    # define a bounding region for the image
    h, w = green_filtered_image.shape[:2]          # height and width
    poly = np.array([(int(w * 0.35), int(h*0.75)), (int(w * 0.67), int(h*0.75)), (w, h), (0, h)])

    # define a mask (size) of the image
    mask = np.zeros_like(green_filtered_image)

    # Define region of interest
    rio = cv2.fillPoly(mask, [poly], 255)

    # Mask the image for the region of interest
    masked_image = cv2.bitwise_and(green_filtered_image, green_filtered_image, mask=rio)

    return masked_image
