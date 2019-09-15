import dlib
import cv2
import time
from math import sqrt
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector() #Load face detector
vs = cv2.VideoCapture(0)
while True:
    check, frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 0)  #Xác định vị trí khuôn mặt trong bức ảnh
    for rect in dets:		
        landmark = predictor(gray, rect)
        lines = []
        # Xác định facial landmark trên khuôn mặt
        for k, d in enumerate(landmark.parts()):
            #xác định khung miệng
            if(k>=60 and k<=68):
                lines.append((d.x,d.y))

        #tìm điểm trung bình line
        x_line = round((lines[4][0]+lines[0][0])/2)
        y_line = round((lines[4][1]+lines[0][1])/2)

        #tính toán khoảng cách
        u_x = (lines[2][0]-x_line)*(lines[2][0]-x_line)
        u_y = (lines[2][1]-y_line)*(lines[2][1]-y_line)
        d_x = (lines[6][0]-x_line)*(lines[6][0]-x_line)
        d_y = (lines[6][1]-y_line)*(lines[6][1]-y_line)

        #kết luận
        if sqrt(u_x+u_y) < sqrt(d_x+d_y):
            cv2.putText(frame, 'Happy', (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
        elif sqrt(u_x+u_y) > sqrt(d_x+d_y):
            if sqrt(u_x+u_y) - sqrt(d_x+d_y) >= 1: #tính độ chênh lệch ( độ chênh lệch tương đương khoảng cách đỉnh môi trên và dưới. Lưu ý : Điểu chỉnh điều kiện để tạo độ nhạy !
                cv2.putText(frame, 'Sad', (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA) 
            else:
                cv2.putText(frame, 'Normal', (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0) , 2, cv2.LINE_AA)

    cv2.imshow('Face video', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows
