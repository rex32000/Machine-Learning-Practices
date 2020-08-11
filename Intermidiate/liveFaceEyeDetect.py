import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

def face_detector(img, size=0.5):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_classifier.detectMultiscale(gray, 1.3, 5)
	if faces is ():
		return img

	for (x, y, e, h) in faces:
		x = x-50
		y = y-50
		w = w+50
		h = h+50
		cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_classifier.detectMultiscale(roi_gray)

		for (ex, ey, ew,, eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+we, ey+eh),(0,0,255),2)

	roi_color = cv2.flip(roi_color, 1)
	return roi_color

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	cv2.imshow('Our face extractor', face_detector(frame))
	if cv2.waitKey(1) == 13:
		break

cv2.release()
cv2.destroyAllWindows()
