# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 20:37:18 2018

@author: sahithi
"""
#References
#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import os
import numpy as np
os.chdir("C:/Users/sahithi/Documents/GitHub/cvlib") #your project folder path 
from gtts import gTTS
import pyttsx3

    
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

    
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    #Text to Speech
    #label = "fruit"
    #engine = pyttsx3.init()
    #engine.say(label)
    #engine.runAndWait() 

    
    
    #tts = gTTS(text=label, lang='en')
    #tts.save("good.mp3")
    #os.system("mpg321 good.mp3")
    #os.startfile("C:/Users/sahithi/Documents/GitHub/cvlib/good.mp3")

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




# # Loading the network 

# Specify the paths for the 2 files
config = "yolov3.cfg"
weights = "yolov3.weights"
labels = "C:/Users/sahithi/Documents/GitHub/cvlib/yolov3.txt"

#  Reading input from webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, image = cap.read()
    cv2.imwrite("test.PNG", image)  
    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
    #classes = None
    
    with open(labels, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    
    net = cv2.dnn.readNet(weights, config)
    
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    
    net.setInput(blob)
    
    outs = net.forward(get_output_layers(net))
    #qprint(outs)
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    count = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
            
    cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)    
    cv2.imshow('Output',image)
    cv2.imwrite('Output.jpg',image)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
#out.release()
cv2.destroyAllWindows()

#from gtts import gTTS

