# OIDv4_ToolKit
This is a modification of [this original repo](https://github.com/EscVM/OIDv4_ToolKit).

What I changed: store the files in a different folder structure and convert the annotations to be compatable with tflite-model-maker annotations.


## Getting Started
Open Images V4 contains [600 classes](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html) and it's a huge dataset.
The goal here is to create subset for tflite model training using only the classes you need, and use the annotantion that [tf-model-maker](https://cloud.google.com/vision/automl/object-detection/docs/label-images-edge-model#preparing_a_dataset) requires.

### Instructions


Choose the one you are interested and place the in [classes.txt](classes.txt) line by line. If a class a space replace it with underscore ( "Mobile phone" -> "Mobile_phone")

Run ```./run.sh ``` which will download the dataset related to the classes specified and will not re-download them in the future.

Once done, the images and annotation files will be organized in the object_detection/train/ and image_classifier/test/data_set/ . Those folder will contains instruction about how to then train the model.


### Note
There will be 3 different copies of the data set. One in each open_images_colletor, object_detection and image_classifier.
If you only need one of the the two (object detection or image classification), you can disable the other in [create_datasets.py ](create_datasets.py).

Do not delete the images in the OID folder otherwise they will have to be re-downloaded.


