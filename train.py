import os
import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from model import *
from preprocessing import *
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

gc.collect()
tf.keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

############### PREPROCCESING 
start_time = time.time()

scales = None
average = False
apply_filter = False
band = False
fs = 200

additional_columns = ['AGE', 'SEX']  # Columns you want to include
categorical_columns = ['SEX']
numerical_columns = ['AGE']

# Seed for the random train/test split
seed = 42
random.seed(seed)

img_models = {
    "DenseNet201": tf.keras.applications.densenet.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    "InceptionV3": tf.keras.applications.inception_v3.preprocess_input,
    "Xception": tf.keras.applications.xception.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
    "VGG19": tf.keras.applications.vgg19.preprocess_input,
    "ResNet101": tf.keras.applications.resnet.preprocess_input,
    "InceptionResNetV21": tf.keras.applications.inception_resnet_v2.preprocess_input,
    "MobileNet": tf.keras.applications.mobilenet.preprocess_input
}

############### DESCRIPTION AND HYPERPARAMETERS

# Directory where the CSV files for training are located
train_path = r''
save_path = r''

# Name, hyperparameters and description of the experiment
name_exp = 'EXP 1'
description = ''
base_img_model = "VGG16"
backbone = img_models[base_img_model]
max_freq_FFT = 40
dropout = 0.2
epochs = 1000
batch_size = 16
patience = 15

# Window size and step size between windows
window_size = round(fs * 15)
step_size = round(fs * 1)

file = r''

X_train, X_test, X_img_train, X_img_test, y_train, y_test, da_train, da_test, num_classes, clases = preprocess_data(
    train_path,
    window_size,
    step_size,
    scales=None,
    master_path=file,
    additional_columns=additional_columns,
    categorical_columns=categorical_columns,
    numerical_columns=numerical_columns,
    img_size=(224, 224),
    image_preprocessor=backbone
)

if num_classes == 2:
    num_classes = 1

input_shape = X_train.shape[1:]

############### TRAINING

model = multimodal_erg_model(input_shape, (224, 224), num_classes, window_size, num_variables_adicionales=len(additional_columns))

checkpoint_val = ModelCheckpoint(save_path+name_exp+'_best_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_freq='epoch')
checkpoint_loss = ModelCheckpoint(save_path+name_exp+'_best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')
earlystop = CustomEarlyStopping(patience=patience)

reduce_lr = ReduceLROnPlateauConUmbral(
    monitor="val_loss",  
    threshold=0.5,       
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    verbose=1
)

callbacks = [checkpoint_val, checkpoint_loss, earlystop, reduce_lr]

if num_classes == 1:
    loss = "binary_crossentropy"
else:
    loss = "sparse_categorical_crossentropy"

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss,
    metrics=["accuracy"],
)

######## avoid overfitting 

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

history = model.fit(
    [X_train, da_train, X_img_train], 
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=([X_test, da_test, X_img_test], y_test), 
    class_weight=class_weights,
    shuffle=True,
    verbose=1,
)

loss = history.history['loss']
val_loss = history.history['val_loss']

######## Graph
epochs_range = range(1, len(loss) + 1)
epoch_best_model = len(epochs_range) - patience
plt.plot(epochs_range, loss, 'y', label='Training loss')
plt.plot(epochs_range, val_loss, 'r', label='Test loss')
plt.title('Training and test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.axvline(x=epoch_best_model, color='blue', linestyle='--', label='Best model saved')
plt.legend()
plt.close()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs_range, acc, 'y', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'r', label='Test Accuracy')
plt.title('Training and test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.axvline(x=epoch_best_model, color='blue', linestyle='--', label='Best model saved')
plt.legend()
plt.show()

#### SAVE MODEL
model.save(save_path +name_exp+'_final_model.hdf5')

end_time = time.time()

print(end_time - start_time)