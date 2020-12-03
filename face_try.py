import cv2
from keras.models import load_model
model = load_model('my_model')
from keras.preprocessing import image
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
i=1
while(cap.isOpened()):
     ret,frame=cap.read()
     if(i!=1):
         i=i-1
         continue
     i=3
     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     faces=face_cascade.detectMultiScale(gray,1.1,4,minSize=(30,30))
     for(x,y,w,h) in faces:
             cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
             face=frame[x:x+w,y:y+h]
             height, width, channels = face.shape 
             if( width and height):
                 face=cv2.resize(face,(64,64))
                 face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                 n = image.img_to_array(face)
                 n = np.expand_dims(n, axis=0)

                 images = np.vstack([n])
                 classes = model.predict_classes(images, batch_size=1)
                 classes = classes[0][0]
        
                 if classes == 0:
                     print('with_mask')
                     frame=cv2.putText(frame,'With mask',(x,y),font,1,(255,0,0),2)
                 else:
                     print('without_mask')
                     frame=cv2.putText(frame,'Without mask',(x,y),font,1,(0,0,255),2)
     cv2.imshow('frame',frame)
     if cv2.waitKey(1)==27:
         break
cap.release()
cv2.destroyAllWindows()
