import cv2
import time
import numpy as np
import os

import RPi.GPIO as GPIO

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.IN) # Button 1 capture
GPIO.setup(23, GPIO.IN) # Button 2 decrease filter
GPIO.setup(24, GPIO.IN) # Button 3 increase filter
GPIO.setup(25, GPIO.IN) # Button 4 change type


# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 320)  # Set width
cap.set(4, 240)  # Set height

# Load mustache image
overlay_image1 = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
overlay_image2 = cv2.imread("hairband.png", cv2.IMREAD_UNCHANGED)

switchValue = 0
typeValue = 0

# Load Haar cascades for face and nose
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
# Check if folder 'image' exists, if not create it
if not os.path.exists("image"):
    os.makedirs("image")


# Function to add mustache to detected faces
def filterImage(frame, sticker):
    try:
        if sticker == 0:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray)
        for x, y, w, h in faces:
            noses = nose_cascade.detectMultiScale(gray)
            for nx, ny, nw, nh in noses:
                if sticker == 1:
                    overlay_image = overlay_image1
                    filter = cv2.resize(overlay_image, (2 * nw, 2 * nh))
                    mask = filter[:, :, 3] == 255
                    frame[
                        ny - nh // 3 : ny + 2 * nh - nh // 3,
                        nx - nw // 3 : nx + 2 * nw - nw // 3,
                    ][mask] = filter[:, :, :3][mask]
                elif sticker == 2:
                    overlay_image = overlay_image2
                    filter = cv2.resize(overlay_image, (w, h // 2))
                    mask = filter[:, :, 3] == 255
                    frame[y - h // 2 + h // 8 : y + h // 8, x : x + w][mask] = filter[
                        :, :, :3
                    ][mask]
                break
            break
        return frame
    except:
        return frame


def singlepost():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = filterImage(frame, switchValue)
    cv2.imshow("photobooth", frame)
    if GPIO.input(18):
        cv2.imwrite("image/single-" + str(time.time()) + ".jpg", frame)
        print("Image saved")
        cv2.waitKey(2000)


def post3x1():
    counter = 0
    _, frame = cap.read()
    show_frame = np.zeros([frame.shape[0] * 3, frame.shape[1], 3], dtype=np.uint8)
    frame = cv2.flip(frame, 1)
    frame = filterImage(frame, switchValue)
    roi = show_frame[
        counter * frame.shape[0] : counter * frame.shape[0] + frame.shape[0],
        0 : 0 + frame.shape[1],
    ]
    roi -= roi
    roi += frame
    cv2.imshow("photobooth", show_frame)
    if GPIO.input(18):
        counter += 1
        cv2.waitKey(1000)
    while counter >= 1 and counter <= 3:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = filterImage(frame, switchValue)
        roi = show_frame[
            counter * frame.shape[0] : counter * frame.shape[0] + frame.shape[0],
            0 : 0 + frame.shape[1],
        ]
        roi -= roi
        roi += frame
        cv2.imshow("photobooth", show_frame)
        if GPIO.input(18):
            counter += 1
            if counter == 3:
                cv2.imwrite("image/post3x1-" + str(time.time()) + ".jpg", show_frame)
                print("Image saved")
                counter = 0
            cv2.waitKey(1000)


def post2x2():
    counter = 0
    _, frame = cap.read()
    show_frame = np.zeros([frame.shape[0] * 2, frame.shape[1] * 2, 3], dtype=np.uint8)
    frame = cv2.flip(frame, 1)
    frame = filterImage(frame, switchValue)
    roi = show_frame[
        counter // 2 * frame.shape[0] : counter // 2 * frame.shape[0] + frame.shape[0],
        counter % 2 * frame.shape[1] : counter % 2 * frame.shape[1] + frame.shape[1],
    ]
    roi -= roi
    roi += frame
    cv2.imshow("photobooth", show_frame)
    if GPIO.input(18):
        counter += 1
        cv2.waitKey(1000)
    while counter >= 1 and counter <= 4:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = filterImage(frame, switchValue)
        roi = show_frame[
            counter // 2 * frame.shape[0] : counter // 2 * frame.shape[0]
            + frame.shape[0],
            counter % 2 * frame.shape[1] : counter % 2 * frame.shape[1]
            + frame.shape[1],
        ]
        roi -= roi
        roi += frame
        cv2.imshow("photobooth", show_frame)
        if GPIO.input(18):
            counter += 1
            if counter == 4:
                cv2.imwrite("image/post2x2-" + str(time.time()) + ".jpg", show_frame)
                print("Image saved")
                cv2.waitKey(2000)
                counter = 0
            cv2.waitKey(1000)


# Main loop
while True:
    if GPIO.input(23):
        switchValue -= 1
        if switchValue < 0:
            switchValue = 2
        cv2.waitKey(1500)

    if GPIO.input(24):
        switchValue += 1
        if switchValue > 2:
            switchValue = 0
        cv2.waitKey(1500)

    if GPIO.input(25):
        typeValue += 1
        if typeValue > 2:
            typeValue = 0
        cv2.waitKey(1500)
    print(typeValue)
    if typeValue == 0:
        singlepost()

    if typeValue == 1:
        post3x1()

    if typeValue == 2:
        post2x2()

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
