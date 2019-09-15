import dlib
import cv2
import time
from imutils import face_utils
from math import sqrt
from skimage.measure import compare_ssim
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
vs = cv2.VideoCapture(0)
check1,frame1 = vs.read()
default_img = frame1
default_img = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
default_img = cv2.resize(src=default_img, dsize=(50, 50))
while True:
    check, frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_size = cv2.resize(src=gray, dsize=(50, 50))
    (score, diff) = compare_ssim(gray_size, default_img, full=True)
    if score < 0.5:
        cv2.putText(frame, 'Move', (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Nothing', (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA)
    dets = detector(gray, 0)
    for rect in dets:
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
            break
cv2.destroyAllWindows()
