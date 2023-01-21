import numpy as np
import os

from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


spec = model_spec.get('efficientdet_lite0')

train_data, validation_data, test_data = object_detector.DataLoader.from_csv('annotations.csv')

model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)

model.evaluate(test_data)

model.export(export_dir='.', export_format=[ExportFormat.SAVED_MODEL, ExportFormat.LABEL, ExportFormat.TFLITE])

model.evaluate_tflite('model.tflite', test_data)

config = QuantizationConfig.for_float16()
model.export(export_dir='.', tflite_filename='model_fp16.tflite', quantization_config=config)

config = QuantizationConfig.for_dynamic()
model.export(export_dir='.', tflite_filename='model_dynamic.tflite', quantization_config=config)
  
