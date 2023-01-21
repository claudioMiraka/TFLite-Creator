import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',
                    default="test_image.png")
args = parser.parse_args()

min_conf_threshold = float(args.threshold)

IM_NAME = args.image

with open("labels.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]


interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5


image = cv2.imread(IM_NAME)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imH, imW, _ = image.shape 
image_resized = cv2.resize(image_rgb, (width, height))
input_data = np.expand_dims(image_resized, axis=0)

if floating_model:
    input_data = (np.float32(input_data) - input_mean) / input_std

interpreter.set_tensor(input_details[0]['index'],input_data)
interpreter.invoke()

boxes = interpreter.get_tensor(output_details[1]['index'])[0] 
classes = interpreter.get_tensor(output_details[3]['index'])[0] 
scores = interpreter.get_tensor(output_details[0]['index'])[0] 

for i in range(len(scores)):
    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

        ymin = int(max(1,(boxes[i][0] * imH)))
        xmin = int(max(1,(boxes[i][1] * imW)))
        ymax = int(min(imH,(boxes[i][2] * imH)))
        xmax = int(min(imW,(boxes[i][3] * imW)))
        
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

        object_name = labels[int(classes[i])] 
        label = '%s: %d%%' % (object_name, int(scores[i]*100)) 
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) 
        label_ymin = max(ymin, labelSize[1] + 10) 
        cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
        cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 

cv2.imshow('Object detector', image)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()     

