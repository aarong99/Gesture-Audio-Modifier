import numpy as np
import cv2

# access camera resource
cap = cv2.VideoCapture(0)

# set up hand recognition parameters
parameters = {'blockSize': 11, 'C': 8, 'minArea': 1000}

# loop over frames from the video stream
while True:
    # read the next frame from the video stream
    ret, frame = cap.read()

    # convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define the range of skin color values in the HSV color space
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # create a binary mask of the skin color regions in the frame
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # apply a Gaussian blur to the binary mask to remove noise
    skin_mask_blur = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    # apply a threshold to the blurred mask to create a binary image
    _, thresh = cv2.threshold(skin_mask_blur, 0, 255, cv2.THRESH_BINARY)

    # find the contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for contour in contours:
        # if the contour area is less than the minimum area threshold, skip it
        if cv2.contourArea(contour) < parameters['minArea']:
            continue

        # find the convex hull of the contour
        hull = cv2.convexHull(contour)

        # draw the convex hull on the original frame
        cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow('Hand detection', frame)

    # exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
