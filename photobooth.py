import cv2
import time
import numpy as np
# import RPi.GPIO as GPIO

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Set width
cap.set(4, 240)  # Set height

# Load mustache image
overlay_image = cv2.imread('mustache.png')

# Set up GPIO
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(24, GPIO.IN)

# Load Haar cascades for face and nose
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')

# Function to add mustache to detected faces
def mustachify(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        noses = nose_cascade.detectMultiScale(gray)
        for (nx, ny, nw, nh) in noses:
            filter = cv2.resize(overlay_image, (nw, nh))
            roi = frame[ny:ny+nh, nx:nx+nw]
            if (roi.shape == filter.shape):
                roi[np.where(filter)] = 0
                roi += filter
            break
        break
    return frame

# Main loop
while True:
    ret, frame = cap.read()
    # inputValue = GPIO.input(24)
    inputValue = False
    if inputValue == True:
        frame = mustachify(frame)
        cv2.imwrite("mustache-" + str(time.time()) + ".jpg", frame)
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(3000)
    else:
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', mustachify(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()