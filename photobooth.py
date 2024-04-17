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
switchValue = 0
typeValue = 0

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(23, GPIO.IN)
GPIO.setup(24, GPIO.IN)
GPIO.setup(25, GPIO.IN)


# Load Haar cascades for face and nose
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_cascade = cv2.CascadeClassifier('haarcascade_mcs_nose.xml')
# Check if folder 'image' exists, if not create it
if not os.path.exists('image'):
    os.makedirs('image')

# Function to add mustache to detected faces
def mustachify(frame, sticker):
    if sticker == 0:
        return frame
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
    
def singlepost():
    ret, frame = cap.read()
    if GPIO.input(23):
        frame = cv2.flip(frame, 1)
        frame = mustachify(frame, switchValue)
        cv2.imwrite("image/singlepost-" + str(time.time()) + ".jpg", frame)
        print("Image saved")
        cv2.imshow('frame', frame)
        cv2.waitKey(3000)
    else:
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', mustachify(frame, switchValue))

def post3x3(frame):
    counter = 0
    while counter < 3:
        ret, frame = cap.read()
        if GPIO.input(23):
            frame = cv2.flip(frame, 1)
            frame = mustachify(frame, switchValue)
            cv2.imwrite("image/post3x3-" + str(time.time()) + ".jpg", frame)
            print("Image saved")
            cv2.imshow('frame', frame)
            cv2.waitKey(3000)
        else:
            frame = cv2.flip(frame, 1)
            cv2.imshow('frame', mustachify(frame, switchValue))

# Main loop
while True:
    if GPIO.input(24):
        switchValue += 1
        if switchValue > 2:
            switchValue = 1
        cv2.waitKey(1000)

    if GPIO.input(25):
        typeValue += 1
        if typeValue > 2:
            typeValue = 1
        cv2.waitKey(1000)

    print(typeValue)
    if typeValue == 0:
        singlepost()
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
