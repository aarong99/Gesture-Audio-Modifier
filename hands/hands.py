import numpy as np
import cv2

# access camera resource
cap = cv2.VideoCapture(0)

# load our serialized hand detector model from disk
print("[INFO] loading hand detector model...")
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "hand_detector.xml")

# load the input image and convert it to grayscale
image = cap.read()
gray = cv2.UMat(np.array(image), cv2.CV_8UC3, image.shape, image.strides)

# detect hands in the image
print("[INFO] detecting hands...")
rects = hand_cascade.detectMultiScale(gray, scaleFactor=1.1,
	minNeighbors=5, minSize=(30, 30))

# loop over the detected hands and draw boxes around them
for (x, y, w, h) in rects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# display frame
cv2.imshow('image', image)

# quit with 'q' ascii value input
cv2.waitKey(0)

# release camera resource
cap.release()
cv2.destroyAllWindows()
