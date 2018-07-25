import numpy as np
import cv2 as cv
cc_base = '/Users/minsubsim/work/opencv/data/haarcascades_cuda/'
face_cascade = cv.CascadeClassifier(cc_base + 'haarcascade_fullbody.xml')

cap = cv.VideoCapture('samples/redvelvet_view.mp4')
# params for ShiTomasi corner detection
fps = cap.get(cv.CAP_PROP_FPS)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)



print(1000.0 / fps)
wait_time = max(5, int(1000.0/fps) - 10)
while(cap.grab()):
    ret, frame = cap.retrieve()
    frame = cv.resize(frame, None, fx=0.7, fy=0.7, interpolation = cv.INTER_CUBIC)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    cv.imshow('frame', frame)
    k = cv.waitKey(wait_time) & 0xff
    if k == 27:
        break

cv.destroyAllWindows()
cap.release()
