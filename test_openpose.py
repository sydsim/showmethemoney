import numpy as np
import cv2 as cv
import json
cap = cv.VideoCapture('samples/rvview_c1.mp4')
# params for ShiTomasi corner detection
fps = cap.get(cv.CAP_PROP_FPS)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

color = np.random.randint(0,255,(100,3))

frame_index = 153
bplink = [(0,1),
(1,2),(2,3),(3,4),
(1,5),(5,6),(6,7),
(1,8),(8,9),(9,10),
(1,11),(11,12),(12,13)]

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    r = json.load(open('outputs/openpose/rvview/rvview_c1_%012d_keypoints.json' % frame_index, 'r'))
    for i, p in enumerate(r['people']):
        xl = p['pose_keypoints_2d'][0::3]
        yl = p['pose_keypoints_2d'][1::3]
        cl = p['pose_keypoints_2d'][2::3]
        for p1, p2 in bplink:
            if cl[p1] == 0 or cl[p2] == 0:
                continue
            x1 = int(round(xl[p1]))
            x2 = int(round(xl[p2]))
            y1 = int(round(yl[p1]))
            y2 = int(round(yl[p2]))
            frame = cv.line(frame, (x1,y1),(x2,y2), color[i].tolist(), 2)

    cv.imshow('frame',frame)
    k = cv.waitKey(100) & 0xff
    if k == 27:
        break
    frame_index += 1

cap.release()
cv.destroyAllWindows()
