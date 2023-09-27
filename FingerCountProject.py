import cv2
import time
import os
import handTrackingModule as htm


cam = cv2.VideoCapture(0)

cam.set(3, 640)
cam.set(4, 480)

pTime = 0

folderPath = 'fingers'
myList = os.listdir(folderPath)
# print(myList)

overplaylist = []

for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")

    image = cv2.resize(image, (200, 200))  # Resize the image to match the region of interest

    overplaylist.append(image)
    # print(f"{folderPath}/{imPath}")
    # print(len(overplaylist))


detector = htm.handDetector(detectionCon=0.75)

while True:
    success, img = cam.read()

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:

        if lmList[8][2] < lmList[6][2]:
            print('index finger open')
        else:
            print('close')

    h, w, c = overplaylist[4].shape
    # img[0:h, 0:w] = overplaylist[4]
    img[0:200, 0:200] = overplaylist[4]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS : {int(fps)}", (400, 70),  cv2.FONT_HERSHEY_COMPLEX, 1, (0,  255, 0), 2)

    cv2.imshow('image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
