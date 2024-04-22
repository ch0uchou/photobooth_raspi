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
overlay_image1 = cv2.imread("mustache.png")
overlay_image2 = cv2.imread("hairband.png")
switchValue = 0
typeValue = 0
cap_count = 0

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.IN)
GPIO.setup(23, GPIO.IN)
GPIO.setup(24, GPIO.IN)
GPIO.setup(25, GPIO.IN)


# Load Haar cascades for face and nose
overlay_image1 = cv2.imread("mustache.png", cv2.IMREAD_UNCHANGED)
overlay_image2 = cv2.imread("hairband.png", cv2.IMREAD_UNCHANGED)
overlay_image3 = cv2.imread("jiwon.png", cv2.IMREAD_UNCHANGED)
overlay_image4 = cv2.imread("soohuyn.png", cv2.IMREAD_UNCHANGED)
overlay_image5 = cv2.imread("heart.png", cv2.IMREAD_UNCHANGED)
overlay_image6 = cv2.imread("flower.png", cv2.IMREAD_UNCHANGED)


# Load Haar cascades for face and nose
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
# Check if folder 'image' exists, if not create it
if not os.path.exists("image"):
    os.makedirs("image")


# Function to filter image
def filterImage(frame, sticker):
    try:
        if sticker == 0:
            return frame
        elif sticker == 3:
            overlay_image = overlay_image3
            filter = cv2.resize(overlay_image, (480, 480))
            mask = filter[:, :, 3] == 255
            frame[0:480, 0:480][mask] = filter[:, :, :3][mask]
            return frame
        elif sticker == 4:
            overlay_image = overlay_image4
            filter = cv2.resize(overlay_image, (480, 480))
            mask = filter[:, :, 3] == 255
            frame[-480:, -480:][mask] = filter[:, :, :3][mask]
            return frame
        elif sticker == 6:
            overlay_image = overlay_image6
            filter = cv2.resize(overlay_image, (640, 480))
            mask = filter[:, :, 3] == 255
            frame[0:480:, 0:640][mask] = filter[:, :, :3][mask]
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
                elif sticker == 5:
                    overlay_image = overlay_image5
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
        cv2.imwrite("image/singlepost-" + str(time.time()) + ".jpg", frame)
        print("Image saved")
        cv2.waitKey(3000)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()


def post3x1():
    # counter = 0
    _, frame = cap.read()
    show_frame = np.zeros([frame.shape[0] * 3, frame.shape[1], 3], dtype=np.uint8)
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = filterImage(frame, switchValue)
    roi = show_frame[
        cap_count * frame.shape[0] : cap_count * frame.shape[0] + frame.shape[0],
        0 : 0 + frame.shape[1],
    ]
    roi -= roi
    roi += frame
    cv2.imshow("photobooth", show_frame)
    if cap_count != 3:
        cv2.imwrite("image/post3x1-" + str(time.time()) + ".jpg", show_frame)
        print("Image saved")
        cv2.waitKey(2000)
    cv2.waitKey(1000)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()

    # while counter != 3:
    #     _, frame = cap.read()
    #     frame = cv2.flip(frame, 1)
    #     frame = filterImage(frame, switchValue)
    #     roi = show_frame[
    #         counter * frame.shape[0] : counter * frame.shape[0] + frame.shape[0],
    #         0 : 0 + frame.shape[1],
    #     ]
    #     roi -= roi
    #     roi += frame
    #     cv2.imshow("photobooth", show_frame)
    #     if GPIO.input(18):
    #         counter += 1
    #         if counter == 3:
    #             cv2.imwrite("image/post3x1-" + str(time.time()) + ".jpg", show_frame)
    #             print("Image saved")
    #             cv2.waitKey(2000)
    #         cv2.waitKey(1000)

    #     if cv2.waitKey(1) & 0xFF == ord("q"):
    #         cap.release()
    #         cv2.destroyAllWindows()


def post2x2():
    counter = 0
    _, frame = cap.read()
    show_frame = np.zeros([frame.shape[0] * 2, frame.shape[1] * 2, 3], dtype=np.uint8)
    while counter != 4:
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
            cv2.waitKey(1000)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()




# Main loop
while True:
    if GPIO.input(23):
        switchValue -= 1
        if switchValue < 0:
            switchValue = 6
        cv2.waitKey(1000)

    if GPIO.input(24):
        switchValue += 1
        if switchValue > 6:
            switchValue = 0
        cv2.waitKey(1000)

    if GPIO.input(25):
        typeValue += 1
        if typeValue > 2:
            typeValue = 0
        cv2.waitKey(1000)

    if GPIO.input(18):
        cap_count += 1
        cv2.waitKey(1000)

    if typeValue == 0:
        singlepost()

    if typeValue == 1:
        post3x1()

    # if typeValue == 2:
    #     post2x2()
