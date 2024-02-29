import pyaudio
import numpy as np
import librosa
import sounddevice as sd
import cv2

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define callback function to process audio data
def callback(in_data, frame_count, time_info, status):
    global note
    data = np.frombuffer(in_data, dtype=np.float32)
    if note == 1:
        # Pitch shift down one octave
        data = librosa.effects.pitch_shift(data, sr, n_steps=-12)
    elif note == 2:
        # Pitch shift up one octave
        data = librosa.effects.pitch_shift(data, sr, n_steps=12)
    elif note == 3:
        # Create major harmony
        data_harm = librosa.effects.harmonic(data, margin=8)
        data = np.vstack([data, data_harm])
    elif note == 4:
        # Create minor harmony
        data_harm = librosa.effects.harmonic(data, margin=8, pitches=[0, 3, 7])
        data = np.vstack([data, data_harm])
    elif note == 5:
        # Add reverb effect
        data = librosa.effects.reverb(data, room_scale=1.2)
    return (data, pyaudio.paContinue)

# Set up stream for audio input from microphone
sr = 44100
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=sr,
                input=True,
                frames_per_buffer=1024,
                stream_callback=callback)

# access camera resource
cap = cv2.VideoCapture(0)

# Start the stream
stream.start_stream()

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
        
            # Find the length of all sides of the triangle
