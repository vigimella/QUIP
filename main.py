import importlib, pkg_resources

importlib.reload(pkg_resources)
import os, glob, shutil, pickle, datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
from distutils import config

import cirq
import argparse
import time
import sympy
import numpy as np
import seaborn as sns
import collections

import random

from tqdm import tqdm
from keras import models
from keras import layers
from keras.metrics import Precision, Recall, AUC
from tensorflow.keras.models import Model

# visualization tools
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
files_folder = os.path.join(APP_ROOT, 'exp_archive')


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # cast to float32 for one_hot encode (otherwise TRUE/FALSE tensor)
    return tf.cast(parts[-2] == CLASS_NAMES, tf.float32)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=1)  # tf.image.decode_jpeg(img, channels=CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_DIM, IMG_DIM])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, loop=False):
    # IF it is a small dataset, only load it once and keep it in memory.
    # OTHERWISE use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    if loop:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


def parse_args():
    parser = argparse.ArgumentParser(prog="QNN", description='Docker to execute Quantum Neural Network')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str, help='dataset')
    group.add_argument('-e', '--epochs', required=False, type=int, default=10,
                       help='number of epochs')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=32)
    group.add_argument('-r', '--learning_rate', required=False, type=float, default=0.01,
                       help="Learning rate for training models")
    group.add_argument('-t', '--threshold', required=False, type=float, default=0.5, help='dataset')

    group.set_defaults(classAnalysis=True)
    arguments = parser.parse_args()
    return arguments


def build_plot(folder, file_name, first_metric_list, second_metric_list, metric_name):
    plt.rcParams["figure.autolayout"] = True

    file_name = file_name.replace('.results', '')
    plot_name = os.path.join(folder, f'{metric_name}_{file_name}')

    # plot creation
    num_epochs = len(first_metric_list) + 1
    epochs = range(1, num_epochs)

    fig, ax = plt.subplots()

    ax.plot(epochs, first_metric_list, color='#BF2A15', linestyle=':', label=f'Training {metric_name}')
    ax.plot(epochs, second_metric_list, 'o-', label=f'Validation {metric_name}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(f'{metric_name}')
    ax.set_xticks(range(1, len(first_metric_list) + 1))
    ax.legend()

    # set plot colors

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('#e5e5e5')
    ax.tick_params(color='#797979', labelcolor='#797979')
    ax.patch.set_edgecolor('black')
    ax.grid(True, color='white')

    if not num_epochs >= 20:
        plt.rcParams["figure.figsize"] = [(num_epochs / 3.22), 5.50]

    else:
        plt.tick_params(axis='x', which='major', labelsize=5)
        plt.rcParams["figure.figsize"] = [8.50, 5.50]

    plt.draw()

    fig.savefig(plot_name, dpi=180)

    print(f"{metric_name}'s plot created and stored at following path: {folder}.")


args = parse_args()

config.DATASET = args.dataset
config.THRESHOLD = args.threshold
config.LEARNING_RATE = args.learning_rate
config.EPOCHS = args.epochs
config.BATCH_SIZE = args.batch_size

dataset_folder = os.path.join(APP_ROOT, config.DATASET)

# dataset path declaration
train_path_dir = tf.data.Dataset.list_files(os.path.join(dataset_folder, "training/train/*/*"))
val_path_dir = tf.data.Dataset.list_files(os.path.join(dataset_folder, "training/val/*/*"))
test_path_dir = tf.data.Dataset.list_files(os.path.join(dataset_folder, "test/*/*"))

train_path_dir_elm = os.path.join(dataset_folder, 'training/train')

templist = list()

for folder in os.walk(train_path_dir_elm):
    templist.append(folder[0].replace(train_path_dir_elm, '').replace('/', ''))

start_time = time.time()

# hyperparameter setting
CLASS_NAMES = list(filter(None, templist))
IMG_DIM = 28
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE
THRESHOLD = config.THRESHOLD
NUM_CLASSES = len(CLASS_NAMES)

print(
    f'Experiment submitted | Batch Size: {str(BATCH_SIZE)}, Epochs: {str(EPOCHS)}, Learning Rate: {str(LEARNING_RATE)}')

lab_train = train_path_dir.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
lab_val = val_path_dir.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
lab_test = test_path_dir.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_ds = prepare_for_training(lab_train)
val_ds = prepare_for_training(lab_val)
test_ds = prepare_for_training(lab_test)

train_data = [(images.numpy(), labels.numpy()) for images, labels in train_ds]
val_data = [(images.numpy(), labels.numpy()) for images, labels in val_ds]
test_data = [(images.numpy(), labels.numpy()) for images, labels in test_ds]

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

for element in train_data:
    x_train.append(element[0])
    y_train.append(element[1])
for element in val_data:
    x_val.append(element[0])
    y_val.append(element[1])
for element in test_data:
    x_test.append(element[0])
    y_test.append(element[1])

fin_train_data = [(images.numpy(), labels.numpy()) for images, labels in train_ds]
fin_val_data = [(images.numpy(), labels.numpy()) for images, labels in val_ds]
fin_test_data = [(images.numpy(), labels.numpy()) for images, labels in test_ds]

del x_train[-1]
del y_train[-1]
del x_val[-1]
del y_val[-1]
del x_test[-1]
del y_test[-1]


def convert_to_circuit(image):
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def quantum_preprocess(image):
    small_image = tf.image.resize(image, (4, 4)).numpy()

    binary_image = np.array(small_image > THRESHOLD, dtype=np.float32)

    circ_image = [convert_to_circuit(x) for x in binary_image]

    tfcirc_img = tfq.convert_to_tensor(circ_image)

    del binary_image, circ_image

    return tfcirc_img


x_train_tfcirc = [quantum_preprocess(train) for train in x_train]
x_val_tfcirc = [quantum_preprocess(val) for val in x_val]
x_test_tfcirc = [quantum_preprocess(test) for test in x_test]

x_train_tfcirc = np.vstack(train for train in x_train_tfcirc).flatten()
x_val_tfcirc = np.vstack(val for val in x_val_tfcirc).flatten()

x_test_tfcirc = np.vstack(val for val in x_test_tfcirc).flatten()
y_train = np.array(y_train)

y_val = np.array(y_val)
y_test = np.array(y_test)

y_train = y_train.reshape(y_train.shape[0] * y_train.shape[1], y_train.shape[2])
y_val = y_val.reshape(y_val.shape[0] * y_val.shape[1], y_val.shape[2])
y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2])


class CircuitLayerBuilder:
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout) ** symbol)


def create_qnn_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)  # a single qubit at [-1,-1]
    circ = cirq.Circuit()

    # Prepare the readout qubit.
    circ.append(cirq.X(readout))
    circ.append(cirq.H(readout))

    builder = CircuitLayerBuilder(data_qubits=data_qubits, readout=readout)

    # Then add layers (TODO experiment by adding more).
    builder.add_layer(circ, cirq.XX, "xx1")
    builder.add_layer(circ, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circ.append(cirq.H(readout))

    return circ, cirq.Z(readout)


model_circuit, model_readout = create_qnn_model()

model = tf.keras.Sequential([
    # The input is the data-circuit, encoded as a tf.string
    tf.keras.layers.Input(shape=(), dtype=tf.string),

    # The PQC layer returns the expected value of the readout gate, range [-1,1].
    tfq.layers.PQC(model_circuit, model_readout),

    layers.Dense(NUM_CLASSES, activation='softmax'),
])

opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=['acc', Precision(name="prec"), Recall(name="rec"), AUC(name='auc')])

qnn_training = model.fit(x=x_train_tfcirc,
                         y=y_train,
                         batch_size=BATCH_SIZE,
                         epochs=EPOCHS,
                         validation_data=(x_val_tfcirc, y_val))

print('Test \n')
qnn_test = model.evaluate(x_test_tfcirc, y_test)

# Save Trained Model

qnn_training.save('qnn_model.h5')

LEARNING_RATE = str(LEARNING_RATE).replace('0.', '')
THRESHOLD = str(THRESHOLD).replace('0.', '')
current_timestamp = datetime.datetime.now()
file_name = f'exp{str(BATCH_SIZE)}{str(EPOCHS)}{LEARNING_RATE}{str(current_timestamp)}T{THRESHOLD}'

history = model.history

# Save Results
print('Saving Results...')

new_folder = os.path.join(files_folder, file_name)

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

output_file_training = os.path.join(new_folder, 'ex-time.txt')

time_seconds = time.time() - start_time

# Convert to hours, minutes, and seconds
time_hours = int(time_seconds // 3600)
time_seconds %= 3600
elapsed_minutes = int(time_seconds // 60)
time_seconds %= 60

# Open the file for writing and save the data
with open(output_file_training, "w") as file:
    for key, value in history.history.items():
        file.write(f"{key}:[{', '.join(map(str, value))}]\n")
    file.write(f'Execution Time: {time_hours} H, {elapsed_minutes} M, {int(time_seconds)} S')

build_plot(folder=new_folder, file_name=file_name, metric_name='accuracy', first_metric_list=qnn_training.history['acc'],
           second_metric_list=qnn_training.history['val_acc'])
build_plot(folder=new_folder, file_name=file_name, metric_name='loss', first_metric_list=qnn_training.history['loss'],
           second_metric_list=qnn_training.history['val_loss'])

shutil.make_archive(new_folder, 'zip', new_folder)

print(f'All files were successfully saved to the following directory: {files_folder}')
