![](https://images.viblo.asia/2ce0132d-e67d-4808-a71c-6d99eb4ae2f4.PNG)
# Introduction
#### Facial emotion identification was a familiar problem in the world of computer vision. Although we have already many libraries that can help create an emotion detection system more convenient than in the past, we will depend on the ability of that and its pre-trained models. So, if you want to expand the recognition capabilities to your liking in the easiest way, welcome to my project. I will choose a very humanistic theme that is "Smiles".

# Ideas
The main idea here is that we will define the shape of the mouth and the threshold instead of using advanced algorithms like Machine Learning or Deep Learning for smile detection.

# About the maths
    I hope you can understand some of the formulas below:
* How to calculate length of vector?
 [
Formula for the Length of a Vector](https://www.wikihow.vn/T%C3%ADnh-%C4%91%E1%BB%99-l%E1%BB%9Bn-c%E1%BB%A7a-v%C3%A9c-t%C6%A1)
* Công thức tính điểm trung bình đoạn thẳng

    (x_tb, y_tb) = <img src="https://render.githubusercontent.com/render/math?math=$( \frac{(x1+x2)}{2} ,\frac{(y1+y2)}{2} )$">
    

# About the code
#### You need to install the libraries below:
   * Dlib
   `pip install dlib`
   * Opencv
   `pip install opencv-python`
   * Dowload the file shape_predictor_68_face_landmarks.dat [tại đây](https://drive.google.com/file/d/13OZVVPDcmIIBFIo4yqdEL_cK0A7Gik5A/view?usp=sharing) \
    The landmark file that look like this :
            ![Example of facial markers](https://scontent.fdad4-1.fna.fbcdn.net/v/t1.6435-9/71720201_392247348363407_1973497817078956032_n.jpg?_nc_cat=102&_nc_map=test-rt&ccb=1-3&_nc_sid=174925&_nc_ohc=BZ2AMUnTlDAAX9oINaE&_nc_ht=scontent.fdad4-1.fna&oh=5b0fdc516df9e82ae2cccd04e474cc3b&oe=60B62993)

     Each white point in the image is a landmark for your face, the effect of this file will give us a model that defines 68 landmarks on the face, we will use the markers to define the shape of the mouth!
 #### Let's coding:
 ##### Import the libraries
 ```python
import dlib
import cv2
from math import sqrt
```
##### Load file  shape_predictor_68_face_landmarks.dat with dlib for face and markers detection
```python
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
```
##### Turn on real-time opencv function 
```python
vs = cv2.VideoCapture(0)
while True:
```
##### Get frames
```python
check, frame = vs.read()
```
#### Convert image to gray image due to that is input of the dlib
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
##### Draw rectangle
```python
for rect in dets:
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
 ```
 ##### Draw landmark
 ```python
 landmark = predictor(gray, rect)
 ```
 ##### Get markers of mouth
 ```python
        for k, d in enumerate(landmark.parts()):
            if(k>=60 and k<=68):
                lines.append((d.x,d.y))
  ```
  ![markers of mouth](https://scontent.fdad4-1.fna.fbcdn.net/v/t1.6435-9/59786701_300642360857240_7484141720582488064_n.jpg?_nc_cat=108&_nc_map=test-rt&ccb=1-3&_nc_sid=730e14&_nc_ohc=iII6imlX-FQAX9mA0ur&_nc_ht=scontent.fdad4-1.fna&oh=d804e5d863423ca615c272a77379b519&oe=60B578EC)
  #### Calculation
  ...

[Read more at my Vietnamese post on Viblo](https://viblo.asia/p/computer-vision-phat-hien-guong-mat-va-nhan-dien-nu-cuoi-don-gian-cho-nguoi-moi-bat-dau-vyDZOwbGZwj#_=_)