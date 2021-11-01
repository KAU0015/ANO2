import cv2
import time

cap = cv2.cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

counter = 0
nFrame = 10
FPS = 0.0
start_time = time.time()

face_cascade = cv2.CascadeClassifier("lbp.xml")#haarcascade_frontalface_default.xml -> data

while(True):
    ret, opencv_frame = cap.read()


    faces_rects = face_cascade.detectMultiScale(opencv_frame, 1.1, 5)

    for one_rect in faces_rects:
        cv2.rectangle(opencv_frame, one_rect, (0, 0, 255), 12)
        cv2.rectangle(opencv_frame, one_rect, (255, 255, 255), 4)

    if counter == nFrame:
        end_time = time.time()
        allTime = float(end_time-start_time)
        FPS = (float(counter)/allTime)
        counter = 0
        start_time = time.time()

    counter = counter + 1

    cv2.putText(opencv_frame, "FPS: " + str(round(FPS, 2)), (50,100), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 255), 12)
    cv2.putText(opencv_frame, "FPS: " + str(round(FPS, 2)), (50,100), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 0), 4)
    cv2.imshow("opencv_frame", opencv_frame)
    
    cv2.waitKey(2)
    