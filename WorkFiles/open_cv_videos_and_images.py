import cv2
import numpy as np


## Function to read in an image
# Generate the path to video path
def readVideo(video_path):
    camera = cv2.VideoCapture(video_path)
    # validate Video Stream
    # ret, prevFrame = camera.read()
    while camera.isOpened():
        if cv2.waitKey(10) == 27:
            break
        ret, currframe = camera.read()
        if ret is True and int(camera.get(cv2.CAP_PROP_POS_FRAMES))%10 == 0:
            # diff = get_frame_difference(prevFrame, currframe)
            image = find_Lanes(currframe)
            cv2.imshow("Lane", currframe)
            # update previous frame
            # prevFrame = currframe


def draw_rec_with_click():
    path = "/Users/chimaochiagha/opencv/Intermediate_Image_processing/00_Introduction/image_data/forest.jpg"
    # readVideo(path)
    # create a window
    myImage = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    cv2.namedWindow("myImage")
    cv2.imshow("myImage", myImage)

    pts = []

    ### left_mouse event callback to draw a rectangle
    def event_callback(event, x, y, flags, params):
        if len(pts) == 2:
            cv2.rectangle(myImage, pts[0], pts[1], [0, 0, 255], 4)
            cv2.imshow("myImage", myImage)
            # clear the
            pts.clear()
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            print(f"Point {len(pts)}  @{x},{y}")
            # wait until button is pushed again
            # draw the rectangle
        return

    # Bind the function to window
    cv2.setMouseCallback('myImage', event_callback)
    cv2.waitKey(0)


def writeToFrames(path):
    # Instantiate the video capture
    camera = cv2.VideoCapture(path)
    # get the image frame dimensions
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"{sz[0]} by {sz[1]}")

    # Define the codec and create VideoWriter object. http://www.fourcc.org/codecs.php
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for avi - DIVX, mp4 - mp4v
    # out = cv2.VideoWriter('./Users/chimaochiagha/opencv/Intermediate_Image_processing/00_Introduction/output_data/horse_text.mp4', fourcc, int(camera.get(cv2.CAP_PROP_FPS)), sz, True)
    # Start the counter
    count = 0
    while camera.isOpened() is True:
        ret, frame = camera.read()
        if cv2.waitKey(10) == 27:
            break
        if ret is True :
            # write the current count to the frame
            cv2.putText(frame, str(count), ( int(sz[0]-100), int(sz[1]-100) ), cv2.FONT_HERSHEY_COMPLEX, 1, [0,0,0])
            cv2.imshow("Horses", frame)
            # out.write(frame)
            count += 1
        else:
            break

def detect_green(path):
    # read in the image
    img = cv2.imread(path)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define a green image
    green_RGB = np.uint8([[[0, 255, 0]]])
    # convert to HSV format
    green_HSV = cv2.cvtColor(green_RGB, cv2.COLOR_BGR2HSV)
    # get the HUE value of green
    hue = green_HSV[0]
    # Set the upper and lower green limits
    lower, upper = np.array([30, 0, 0]), np.array([90, 255, 255])
    # get mask for green colors
    mask = cv2.inRange(img_HSV, lower, upper)
    # imask = mask > 0
    # green = np.zeros_like(img, np.uint8)
    # green[imask] = img[imask]
    green = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("IMG", green)
    cv2.waitKey(0)


def getPos():
    path = "/Users/chimaochiagha/opencv/Intermediate_Image_processing/00_Introduction/image_data/elephant.png"
    elephant = cv2.imread(path)
    cv2.namedWindow("Elephant")
    # define the window name
    cv2.imshow("Elephant", elephant)
    # define a mousclick callback
    def mouse_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"{x} , {y}")
        return
    # set callback
    cv2.setMouseCallback("Elephant", mouse_event)
    cv2.waitKey(0)

def save_video_frames(name : str, videoPath):
    cars = cv2.VideoCapture(videoPath)
    # begin streaming
    while cars.isOpened() is True:
        # read in the frames
        ret, frame = cars.read()
        # check for kill case
        if cv2.waitKey(10) == 27:
            break
        if ret is True:
            # show the image
            # cv2.imshow("Frame", frame)
            #  save the frames in the given seq
            seq = [10, 40, 60, 80]
            frame_id = int(cars.get(cv2.CAP_PROP_POS_FRAMES))
            if frame_id in seq:
                framePath = "/Users/chimaochiagha/opencv/Intermediate_Image_processing/my_computer_vision/output_data/" \
                            + name + str(frame_id) + ".png"
                cv2.imwrite(framePath, frame)

def get_frame_difference(prev_frame, curr_frame):
    '''

    BASE OF MOVING OBJECT DETECTION

    Takes in the previous and current frame of a video and compares them for changes
    The changes are then highlighted,contoured and the contours drawn
    :param prev_frame: Previous frame from video
    :param curr_frame: Current frame in video
    :return: frame with drawn contours
    '''

    # convert the frames to grey scale
    prev_grey = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_grey = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Get the absolute difference between the frames
    diff = cv2.absdiff(curr_grey, prev_grey)

    # Threshold the difference
    _, thres = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Equalize the image
    thres = cv2.equalizeHist(thres)

    # define a kernel for erosion and dilation
    kernel = np.ones((4,4), dtype=np.uint8)

    # erode to remove the white dots
    thres = cv2.erode(thres, kernel = kernel, iterations = 1)

    # dilate to increase the outline of the dominant shapes
    thres = cv2.dilate(thres, kernel = kernel, iterations = 3)

    thres = cv2.erode(thres, kernel=kernel, iterations=1)

    # # get the contours in the image
    contours, hier = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # extract the valid contours from the image
    valids = []
    for contour in contours:
        if cv2.contourArea(contour) >= 5000:
            valids.append(contour)
    #
    # # draw the valid contour on the current frame
    curr_frame = cv2.drawContours(curr_frame, valids, -1, [255,0,0], 2)

    return curr_frame


def find_Lanes(currFrame):
    '''
    Function to finc the lanes in an image
    :param currFrame:
    :return: CurrFrame with lanes drawn on
    '''

    # convert to gray scale
    grey = cv2.cvtColor(currFrame, cv2.COLOR_BGR2GRAY)

    # Perform a gaussian blur to remove unwanted noise
    gaus = cv2.GaussianBlur(grey, (5,5), 1)

    # Find the egges in the image
    canny = cv2.Canny(gaus, 10, 255)

    # Define a rio for the image
    h,w = canny.shape[:2]
    # triangle containing the desired rio
    rio = np.array([ [(0,h), (w,h), (400, 300)] ])

    # define a mask for the whole image
    mask = np.zeros_like(canny)

    # fill the rio region in the mask with zeros
    mask = cv2.fillPoly(mask, rio, 255)

    # Select for the rio in th canny image
    seg_rio_canny = cv2.bitwise_and(canny, canny, mask = mask)

    # Perform a hough transfrom on the image
    houghs = cv2.HoughLinesP(seg_rio_canny, 2, np.pi / 180, 10, np.array([]), minLineLength=100, maxLineGap=50)

    if houghs is not None:
        for i in range(len(houghs)):
            l = houghs[i][0]
            # draw lines
            cv2.line(currFrame, l[:2], l[2:], [0,0,255], 2)

    # return the image
    return currFrame