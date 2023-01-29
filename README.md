# TFLite-Creator

Build ML model that can run on mobile devices without having an ML backgound!

This project uses [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker) to build models for object detection and image classification and [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) to create a data set.



## Getting Started

Each folder has its own readme file with the instructions. 

- [open_images_colletor](open_images_colletor) will create a data set.
- [object_detection/train](object_detection/train) will use the data set created to build an object detection model.
- [image_classifier/train](image_classifier/train) will use the data set created to build an image classifican model.


## Example


#### Classification Example

Trained on multiple classes of birds

<img src="media_example/classification.gif" width="50%" height="70%"/>


#### Objecti Detection Example

Trained to only reconize birds

<img src="media_example/object_detection.gif" width="50%" height="70%"/>

#### Objecti Detection with Classification Live Camera example Example
Combination of the above 2 models.
The object detection model was trained to only reconize only birds. Once a bird is reconized, the image is cropped then passed it to the classification model which was trained to reconize only specific birds (Blue jay, parrot, canary ...).
This combination gives better results than just training an object detection on those specific type of birds.

<img src="media_example/live_classification_and_detection.gif" width="50%" height="70%"/>

