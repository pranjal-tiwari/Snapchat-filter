#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np


# In[2]:


#Init Webcam
cap=cv2.VideoCapture(0)

#Face Detection
eye_cascade = cv2.CascadeClassifier("C:/Users/HP/Downloads/haar-cascade-files-master/haar-cascade-files-master/haarcascade_eye_tree_eyeglasses.xml")
face_cascade = cv2.CascadeClassifier("C:/Users/HP/Downloads/haar-cascade-files-master/haar-cascade-files-master/haarcascade_frontalface_alt.xml")
nose_cascade = cv2.CascadeClassifier("C:/Users/HP/Downloads/haarcascade_mcs_nose.xml")
glasses=cv2.imread("C:/Users/HP/Downloads/Copy of glasses.png",-1)
mustache=cv2.imread("C:/Users/HP/Downloads/Copy of mustache.png",-1)

face_data = []

dataset_path = "C:/Users/HP/Downloads/"
file_name = input('Enter the name of the person:')


# In[ ]:


while True: #for infinite loops 
    ret, frame = cap.read()
    
    if ret == False:
        continue
        
    #gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,scalefactor=1.3, 5) #see
    
    if len(faces) == 0:
        continue
        
    #faces= sorted(faces, key = lambda f:f[2]*f[3])
    
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        
        # extract region of interest
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100,100))
        face_data.append(face_section)
        print(len(face_section))
        
        eyes = eye_cascade.detectMultiScale(face_section,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(face_section,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eye_section = frame[ey-offset:ey+eh+offset, ex-offset:ex+ew+offset]
            
        nose = nose_cascade.detectMultiScale(face_section,1.3,5)
        for (nx,ny,nw,nh) in nose:
            cv2.rectangle(face_section,(nx,ny),(nx+nw,ny+nh),(255,255,0),2) 
            nose_section = frame[ny-offset:ny+nh+offset, nx-offset:nx+nw+offset]
        
    
    cv2.imshow('Frame', frame)
    #qcv2.imshow('Gray_Frame', gray_frame)
    
    key_pressed = cv2.waitKey(1) & 0XFF
    
    if key_pressed == ord('q'):
        break
        
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

np.save(dataset_path+file_name+'.npy', face_data)
print("Data Saved Successfully:")
        
cap.release()
cv2.destroyAllWindows()
        


# In[ ]:


#knn code
def distance(v1,v2):
    return np.sqrt(sum((v1-v2)**2))

def knn(train, test, k=5):
    dis=[]
    
    for i in range(train.shape[0]):
        ix=train[i, :-1]
        iy=train[i, -1]
        d = distance(test, ix)
        dis.append([d, iy])
    dk=sorted(dis, key=lambda x: x[0])[:k]
    labels=np.array(dk)[:,-1]
    output= np.unique(labels, return_counts = True)
    index=np.argmax(output[1])
    return output[0][index]
        
    


# In[ ]:




