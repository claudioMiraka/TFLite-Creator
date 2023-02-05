# TFLite-Creator

Build a TF lite model using a large dataset that can run on mobile devices with only few steps!

This project uses [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/models/modify/model_maker) to build models for object detection and image classification and [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) to create a data set.



## Getting Started

Each folder has its own readme file with the instructions. 

- [open_images_colletor](open_images_colletor) will create a data set.
- [object_detection/train](object_detection/train) will use the data set created to build an object detection model.
- [image_classifier/train](image_classifier/train) will use the data set created to build an image classifican model.

To test the models on PC use  [object_detection/test](object_detection/test) and [image_classifier/test](image_classifier/test); this will generate results like the examples below. 

If you want to test the model on a mobile device checkout [image_classification](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification) and [object_detection](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection)


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

## Note
Tested on Ubuntu and MacOS (Intel processor).

On MacOS with Apple Silicon, training is unsupported; I was unable to install tflite-model-maker (any help is appreciated!!).


