import numpy as np
import cv2

# access camera resource
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to smooth out the image
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the largest area
    max_area = 0
    ci = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            ci = i

    # Find the convex hull of the hand contour
    cnt = contours[ci]
    #hull = cv2.convexHull(cnt)

    # Find the convexity defects between the hand contour and its convex hull
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    # Loop over the convexity defects
    fingers = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
        
            # Find the length of all sides of the triangle formed by the start, end, and far points
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

            # Determine if the angle is less than 90 degrees
            if angle <= np.pi / 2:
                fingers.append(far)

    # Draw circles around fingers
    for finger in fingers:
        cv2.circle(frame, finger, 5, (0, 255, 0), -1)

    # Display the number of fingers on the screen
    cv2.putText(frame, str(len(fingers)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # display the resulting frame
    cv2.imshow('Hand detection', frame)
    
    # exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera resource
cap.release()
cv2.destroyAllWindows()