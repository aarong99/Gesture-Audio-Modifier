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

        # find the defects in the convex hull
        defects = cv2.convexityDefects(contour, hull)

        # loop over the defects
        for i in range(defects.shape[0]):
            # get the start, end, and far points of the defect
            start_idx, end_idx, far_idx, _ = defects[i, 0]

            # get the coordinates of the start, end, and far points
            start = tuple(contour[start_idx][0])
            end = tuple(contour[end_idx][0])
            far = tuple(contour[far_idx][0])

            # calculate the angle between the start, far, and end points
            angle = np.degrees(np.arctan2(far[1] - start[1], far[0] - start[0]) -
                               np.arctan2(end[1] - start[1], end[0] - start[0]))

            # if the angle is less than 90 degrees, draw a circle around the far point
            if angle < 90:
                cv2.circle(frame, far, 5, (0, 0, 255), -1)

    # display the resulting frame
    cv2.imshow('Hand detection', frame)

    # wait for a key press (time in milliseconds)
    #key = cv2.waitKey(1)
    key = cv2.waitKey(1) & 0xFF

    # exit the loop if the 'q' key is pressed
    if key == ord('q'):
        break

# release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)