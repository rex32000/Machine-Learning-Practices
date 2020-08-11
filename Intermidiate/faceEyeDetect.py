#HAAR cascade classifier
import numpy as np
import cv2
#face and eye detection
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

image = cv2.imread('images/Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3,5)

if faces is ():
	print("No faces found")

for (x, y, w, h) in faces:
	cv2.rectangle(image, (x,y), (x+w, y+h), (127,0,255), 2)
	cv2.imshow('Face Detection', image)
	cv2.waitKey(0)
	roi_gray = gray[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	eyes = eye_classifier.detectMultiScale(roi_gray)
	for (ex, ey, ew,eh) in eyes:
		cv2.rectangle(roi_color, (ex, ey), (ex+ew,ey+eh), (255, 255,0),2)
		cv2.imshow('img', img)
		cv2.waitKey(0)

cv2.destroyAllWindows()
