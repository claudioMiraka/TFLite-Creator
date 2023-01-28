import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

image_path = "data_set/"

data = DataLoader.from_folder(image_path)
train_data, rest_data = data.split(0.7)
validation_data, test_data = rest_data.split(0.5)


model = image_classifier.create(train_data, model_spec='efficientnet_lite0',use_augmentation=True, validation_data=validation_data)
loss, accuracy = model.evaluate(test_data)

model.export(export_dir='.', tflite_filename='model.tflite', label_filename='labels.txt', export_format=[ExportFormat.LABEL,ExportFormat.LABEL,ExportFormat.SAVED_MODEL,ExportFormat.TFLITE])
model.evaluate_tflite('model.tflite', test_data)

config = QuantizationConfig.for_float16()
model.export(export_dir='.', tflite_filename='model_fp16.tflite', quantization_config=config)

config = QuantizationConfig.for_dynamic()
model.export(export_dir='.', tflite_filename='model_dynamic.tflite', quantization_config=config)





