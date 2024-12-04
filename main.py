import os
import numpy as np
import mlflow
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import image_classifier
from sklearn.utils.class_weight import compute_class_weight

mlflow.set_tracking_uri(uri='http://localhost:5000')
mlflow.set_experiment('PS Workbooks v2007')
mlflow.sklearn.autolog(disable=True)
mlflow.autolog()

image_path = '/home/corey/Documents/workbook_classification'

data = image_classifier.Dataset.from_folder(image_path)
train_data, remaining_data = data.split(0.8)
test_data, validation_data = remaining_data.split(0.5)

# Assuming your tf.data.Dataset contains (features, label) pairs
labels = []
for _, label in train_data._dataset:
    labels.append(label.numpy())

# Convert labels to a NumPy array
labels_np = np.array(labels)

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_np),
    y=labels_np
)

# Create a dictionary to map class labels to their weights
class_weights_dict = dict(zip(np.unique(labels_np), class_weights))
print(class_weights_dict)

#spec = image_classifier.SupportedModels.MOBILENET_V2
spec = image_classifier.SupportedModels.EFFICIENTNET_LITE4
# Trying without data augmentation 
hparams = image_classifier.HParams(
    export_dir="exported_model_2",
    do_data_augmentation=False,
    class_weights=class_weights_dict,
)
options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)

model = image_classifier.ImageClassifier.create(
    train_data = train_data,
    validation_data = validation_data,
    options=options, 
)

loss, acc = model.evaluate(test_data)
print(f'Test loss:{loss}, Test accuracy:{acc}')

model.export_model(model_name='efficientnet_lite4_psworkbook_weighted.tflite')
model.export_labels(export_dir='exported_model_2')

