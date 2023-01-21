# Train Object Detection 

This is a simple tool to quickly create tflite models. Take a look at [tflite_model_maker](https://www.tensorflow.org/lite/models/modify/model_maker/object_detection) for more info.



### Instructions
This folder should contain the annotations.csv file and images/ folder which are created after running [run.sh](../open_images_colletor/run.sh) in open_images_colletor folder


Run ``` python3 train.py ```.

To test the model, copy the model and labels into object_detection/test/ and follow instructions there.


