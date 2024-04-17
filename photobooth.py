import cv2
import time
import numpy as np
import os
import RPi.GPIO as GPIO

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Set width
cap.set(4, 240)  # Set height

# Load mustache image
overlay_image1 = cv2.imread('mustache.png')
overlay_image2 = cv2.imread('hairband.png')

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.IN)
GPIO.setup(24, GPIO.IN)

# Load Haar cascades for face and nose
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
# Check if folder 'image' exists, if not create it
if not os.path.exists('image'):
    os.makedirs('image')

# Function to add mustache to detected faces
def mustachify(frame, sticker):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        noses = nose_cascade.detectMultiScale(gray)
        for (nx, ny, nw, nh) in noses:
            if sticker == 1:
                overlay_image = overlay_image1
                filter = cv2.resize(overlay_image, (2*nw, 2*nh))
                roi = frame[ny-nh//3:ny+2*nh-nh//3, nx-nw//3:nx+2*nw-nw//3]
            elif sticker == 2:
                overlay_image = overlay_image2
                filter = cv2.resize(overlay_image, (w, h//2))
                roi = frame[y-h//2+h//8:y+h//8, x:x+w]
            
            if (roi.shape == filter.shape):
                roi[np.where(filter)] = 0
                roi += filter
            break
        break
    return frame

# Main loop
while True:
    ret, frame = cap.read()
    switchValue = 1
    if GPIO.input(24):
        switchValue += 1
        if switchValue > 2:
            switchValue = 1

    if GPIO.input(23):
        frame = cv2.flip(frame, 1)
        frame = mustachify(frame, switchValue)
        cv2.imwrite("image/image-" + str(time.time()) + ".jpg", frame)
        print("Image saved")
        cv2.imshow('frame', frame)
        cv2.waitKey(3000)
    else:
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', mustachify(frame, switchValue))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
