import os
import argparse
import cv2
import numpy as np
import time
from tqdm import tqdm
from tensorflow.lite.python.interpreter import Interpreter


LABEL_PATH = "labels.txt"
MODEL_PATH = "model.tflite"

DEFAULT_PADDING = 0.05
DEFAULT_THRESHOLD = 0.5


class ObjectDetectionResult(object):
    def __init__(self, xmin,ymin, xmax,ymax, confidence, label):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence
        self.label = label


class ObjectDector(object):
    def __init__(self, modelPath=MODEL_PATH, labelPath=LABEL_PATH, threshold=DEFAULT_THRESHOLD):

        if not os.path.exists(modelPath):
            raise ValueError('Model not found')
        
        with open(labelPath, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.interpreter = Interpreter(model_path=modelPath)
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        self.isFloatingModel = input_details[0]['dtype'] == np.float32 or input_details[0]['dtype'] == np.float16 
        self.tensorIndex = input_details[0]['index']

        self.inputHeight = input_details[0]['shape'][1]
        self.inputWidth = input_details[0]['shape'][2]
        
        self.outputDetails = self.interpreter.get_output_details()
        self.confidenceThreshold = threshold
        
        
    def detectObjects(self, image, padding=DEFAULT_PADDING):

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (self.inputWidth, self.inputHeight))
        input_data = np.expand_dims(image_resized, axis=0)

        input_mean = 127.5
        input_std = 127.5

        if self.isFloatingModel:
            input_data = (np.float32(input_data) - input_mean) / input_std

        self.interpreter.set_tensor(self.tensorIndex,input_data)
        self.interpreter.invoke()

        boxes = self.interpreter.get_tensor(self.outputDetails[1]['index'])[0]
        classes = self.interpreter.get_tensor(self.outputDetails[3]['index'])[0]
        scores = self.interpreter.get_tensor(self.outputDetails[0]['index'])[0]

        detections = []

        for i in range(len(scores)):
            if ((scores[i] > self.confidenceThreshold) and (scores[i] <= 1.0)):
                ymin = int(max(1,(boxes[i][0]-padding) * imH))
                xmin = int(max(1,(boxes[i][1]-padding) * imW))
                ymax = int(min(imH,(boxes[i][2]+padding) * imH))
                xmax = int(min(imW,(boxes[i][3]+padding) * imW))
                label = self.labels[int(classes[i])]

                detections.append(ObjectDetectionResult(xmin,ymin,xmax,ymax, scores[i], label))
        
        detections.sort(key=lambda x:x.confidence)
        
        return detections



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Full path of image')
    parser.add_argument('--threshold', help='Confidence threshold')

    parser.add_argument('--model', help='Full path of the model')
    parser.add_argument('--labels', help='Full path of the labels')

    args = parser.parse_args()

    labels = LABEL_PATH
    model = MODEL_PATH
    confidence = DEFAULT_THRESHOLD
    
    if args.labels != None :
        labels = args.labels
    if args.model != None :
        model = args.model
    if args.threshold != None:
        confidence = args.threshold


    if args.image != None :
        if not os.path.exists(args.image):
            raise ValueError('Image not found')

        objectDetector = ObjectDector(modelPath=model, labelPath=labels, threshold=confidence)

        image = cv2.imread(args.image)
        start_time = time.time()
        boxes = objectDetector.detectObjects(image)
        print("--- Inference time:  %s seconds ---" % (time.time() - start_time))
        
        for box in boxes :
            cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (10, 255, 0), 2)

            label = box.label + " " + str(int(box.confidence*100)) + "%"
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(box.ymin, labelSize[1] + 10)
            cv2.rectangle(image, (box.xmin, label_ymin-labelSize[1]-10), (box.xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (box.xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
        cv2.imshow('Object detector', image)
        
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()