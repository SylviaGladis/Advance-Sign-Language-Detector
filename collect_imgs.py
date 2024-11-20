import os

import cv2
import pyautogui as pyautogui

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

cam = cv2.VideoCapture(0)
screen_w, screen_h= pyautogui.size()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while True:
        ret, frame = cam.read()
        cv2.putText(frame, 'Start Collecting :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('s'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cam.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cam.release()
cv2.destroyAllWindows()
