import argparse
import cv2

from object_detection.test.detect import ObjectDector
from image_classifier.test.classify import ImageClassifier


DETECTION_THRESHOLD = 0.5

DETECTION_LABELS = "detectionLabels.txt"
DETECTION_MODEL = "detectionModel.tflite"

CLASSIFICATION_LABELS = "classificationModel.txt"
CLASSIFICATION_MODEL = "classificationLabels.tflite"

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--detectionThreshold', help='Confidence threshold',  type=float, default=DETECTION_THRESHOLD)

    parser.add_argument('--detectionModel', help='Path of the object detection model', default=DETECTION_MODEL)
    parser.add_argument('--detectionLabels', help='Path of the object detection labels', default=DETECTION_LABELS)

    parser.add_argument('--classificationModel', help='Path of the classification model', default=CLASSIFICATION_MODEL)
    parser.add_argument('--classificationLabels', help='Path of the classification labels', default=CLASSIFICATION_MODEL)

    args = parser.parse_args()

    vid = cv2.VideoCapture(0)

    objectDetector = ObjectDector(modelPath=args.detectionModel, labelPath=args.detectionLabels, threshold=args.detectionThreshold)
    imageClassifier = ImageClassifier(modelPath=args.classificationModel, labelPath=args.classificationLabels)

    while(True):
          
        ret, image = vid.read()

        # detect the object and add 5% padding to bounding box
        objects = objectDetector.detectObjects(image, padding=0.05) 
        
        for object in objects :
            # draw bounding box
            cv2.rectangle(image, (object.xmin,object.ymin), (object.xmax,object.ymax), (10, 255, 0), 2)

            # classify cropped image
            res_label, conf = imageClassifier.classifyImage(image[object.ymin:object.ymax, object.xmin:object.xmax])
            
            label = '%s: %d%%' % (res_label, int(conf*100))
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(object.ymin, labelSize[1] + 10)
            cv2.rectangle(image, (object.xmin, label_ymin-labelSize[1]-10), (object.xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
            cv2.putText(image, label, (object.xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


        cv2.imshow('Detect me if you can!', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()







