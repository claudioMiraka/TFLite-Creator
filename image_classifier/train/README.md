# Train Image CLassifier

This is a simple tool to quickly create tflite models. Take a look at [tflite_model_maker]([https://www.tensorflow.org/lite/models/modify/model_maker/object_detection](https://www.tensorflow.org/lite/models/modify/model_maker/image_classification) for more info.


### Instructions
This folder should contain the data_set folder which are created after running [run.sh](../open_images_colletor/run.sh) in open_images_colletor folder.

If you are using your own dataset, crete a folder called data_set here, then for each class create a folder with all the images in it.

```
image_classifier/
    | ---> train.py
    | ---> data_set/
        | ---> class1/
            | 12345.png
            | 98239.png
        | ---> class2/
            | 35653.png
            | 43522.png
```



Run ``` python3 train.py ```.

To test the model, copy the model and labels into image_classifier/test/ and follow instructions there.


