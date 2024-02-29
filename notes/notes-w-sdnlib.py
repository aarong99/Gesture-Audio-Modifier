import numpy as np
import librosa
import sounddevice as sd
import cv2

# Define callback function to process audio data
def callback(indata, outdata, frames, time, status):
    global note
    if status:
        print(status)
    if note == 1:
        # Pitch shift down one octave
        outdata[:] = librosa.effects.pitch_shift(indata, sr, n_steps=-12)
    elif note == 2:
        # Pitch shift up one octave
        outdata[:] = librosa.effects.pitch_shift(indata, sr, n_steps=12)
    elif note == 3:
        # Create major harmony
        data_harm = librosa.effects.harmonic(indata, margin=8)
        outdata[:] = np.vstack([indata, data_harm])
    elif note == 4:
        # Create minor harmony
        data_harm = librosa.effects.harmonic(indata, margin=8, pitches=[0, 3, 7])
        outdata[:] = np.vstack([indata, data_harm])
    elif note == 5:
        # Add reverb effect
        outdata[:] = librosa.effects.reverb(indata, room_scale=1.2)
    else:
        outdata[:] = indata

# Set up stream for audio input from microphone
sr = 44100
stream = sd.Stream(callback=callback, blocksize=1024, samplerate=sr, channels=1)

# Start the stream
stream.start()

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
        
            # Find the length of all
