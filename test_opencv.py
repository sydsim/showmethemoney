import numpy as np
import cv2 as cv
cap = cv.VideoCapture('samples/redvelvet_view.mp4')
# params for ShiTomasi corner detection
fps = cap.get(cv.CAP_PROP_FPS)
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.2,
                       minDistance = 5,
                       blockSize = 3 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
                  flags = cv.OPTFLOW_LK_GET_MIN_EIGENVALS,
                  )
# Create some random colors
color = np.random.randint(0,255,(10000,3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(len(p0))
pn = len(p0)
# Create a mask image for drawing purposes
mask = [np.zeros_like(old_frame) for _ in range(pn)]
count = [0] * pn
cont = np.arange(pn)
f_cnt = 0

def find_match(p1, p2):
    d = set()
    for t in p1:
        x, y = t[0]
        d.add((int(x),int(y)))
    r = []
    for t in p2:
        x, y = t[0]
        p = (int(x),int(y))
        if p in d:
            continue
        r.append((x,y))
    return r


while(cap.grab()):
    cont = cont.reshape(-1,1,1)
    ret, frame = cap.retrieve()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow

    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #print(len(p1), len(p0), len(st))
    #print('p0', p0)

    p2 = cv.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    print(p1.shape, p2.shape)
    p2 = find_match(p1, p2)
    print(len(p2))

    cont = cont[st==1]
    p0 = p0[st==1]
    p1 = p1[st==1]
    # draw the tracks
    for i, (ci, old, new) in enumerate(zip(cont, p0, p1)):
        a,b = new.ravel()
        c,d = old.ravel()
        ci = ci[0]
        mask[ci] = cv.line(mask[ci], (a,b),(c,d), color[i].tolist(), 2)
        frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
        frame = cv.add(frame,mask[ci])

    cv.imshow('frame',frame)
    wait_time = 1 # int(1000.0/fps) - 5
    k = cv.waitKey(wait_time) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = p1.reshape(-1,1,2)
    f_cnt += 1
    #if f_cnt % 100 == 0:
    #    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    #    mask = np.zeros_like(frame)
cv.destroyAllWindows()
cap.release()
