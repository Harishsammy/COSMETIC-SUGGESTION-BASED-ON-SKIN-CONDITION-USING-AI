# -*- coding: utf-8 -*-
# MLP for Pima Indians Dataset Serialize to JSON and HDF5
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from PIL import Image
import os
import cv2

json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

##label=['burger','chicken briyani','dosa','idly','pizza','pongal','poori','white rice']

def classify(img_file):
    img_name = img_file
    test_image = image.load_img(img_name, target_size = (128, 128))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = loaded_model.predict(test_image)
    a=np.round(result[0][0])
    b=np.round(result[0][1])
    c=np.round(result[0][2])
   

    print(a)
    print(b)
    print(c)
    

    if a == 1:
        try:
    
            prediction = 'Dry_skin'
            print(prediction)
            img = cv2.imread(img_name)
            cv2.imshow("Dry_skin",img)
            print(prediction,img_name)
            var2=cv2.imread("To prevent dry skin.jpg")
            cv2.imshow("To prevent dry skin.jpg",var2)
        except Exception as e:
            print(e)
            
            

    elif b == 1:
        prediction = 'Normal_skin'
        print(prediction)
        img = cv2.imread(img_name)
        cv2.imshow("Normal_skin",img)
        print(prediction,img_name)
        

        
    elif c == 1:
        prediction = 'Oil_skin'
        print(prediction)
        img = cv2.imread(img_name)
        cv2.imshow("Oil_skin",img)
        print(prediction,img_name)
        var1=cv2.imread("To prevent oil skin.jpg")
        cv2.imshow("To prevent oil skin.jpg",var1)

        
   
    


import os
path = 'data/test'
files = []
for r, d, f in os.walk(path):
   for file in f:
     if '.jpg' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')








