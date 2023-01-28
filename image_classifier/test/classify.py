import os
import argparse
import cv2
import time
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter


LABEL_PATH = "labels.txt"
MODEL_PATH = "model.tflite"


class ImageClassifier(object):
    def __init__(self, modelPath=MODEL_PATH, labelPath=LABEL_PATH):

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


    def classifyImage(self, image):

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
        
        quantization = self.outputDetails[0]['quantization'][0]
        if self.isFloatingModel:
            quantization = 1.0
        scores = [ quantization * x for x in self.interpreter.get_tensor(self.outputDetails[0]['index'])[0]]
        
        max_value = max(scores)
        label = self.labels[scores.index(max_value)]
        
        return (label,max_value)


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Full path of image')
    parser.add_argument('--model', help='Full path of the model')
    parser.add_argument('--labels', help='Full path of the labels')

    args = parser.parse_args()

    labels = LABEL_PATH
    model = MODEL_PATH
    if args.labels != None :
        labels = args.labels

    if args.model != None :
        model = args.model


    if args.image != None :
        if not os.path.exists(args.image):
            raise ValueError('Image not found')

        image = cv2.imread(args.image)

        imageClassifier = ImageClassifier(modelPath=model, labelPath=labels)

        start_time = time.time()
        res_label, conf = imageClassifier.classifyImage(image)
        print("--- Inference time:  %s seconds ---" % (time.time() - start_time))

        label = '%s: %d%%' % (res_label, int(conf*100))
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(image, label, (0,labelSize[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        cv2.imshow('Image classifier', image)

        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    else:
        parser.print_help()