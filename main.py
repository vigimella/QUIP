import os, time, datetime, re, warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import argparse
import time
import sympy
import numpy as np
import collections

from tensorflow.keras.layers import Layer

from distutils import config
from tqdm import tqdm
from keras import models
from keras import layers
from keras.metrics import Precision, Recall, AUC
import io
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import register_keras_serializable


import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

from save_results import save_exp

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
files_folder = os.path.join(APP_ROOT, 'exp_archive')

@register_keras_serializable()
class CustomPQC(tfq.layers.PQC):
    def __init__(self, model_circuit, operators, **kwargs):
        super().__init__(model_circuit, operators, **kwargs)
        self.model_circuit = model_circuit
        self.operators = operators

    def get_config(self):
        config = super().get_config()
        # Convert to JSON strings
        config.update({
            'model_circuit': cirq.to_json(self.model_circuit),
            'operators': [cirq.to_json(op) for op in self.operators],
        })
        return config

    @classmethod
    def from_config(cls, config):

        model_circuit_json = config.pop('model_circuit')
        operators_json = config.pop('operators')

        model_circuit = cirq.read_json(io.StringIO(model_circuit_json))
        raw_operators = [cirq.read_json(io.StringIO(op)) for op in operators_json]

        # Ensure all operators are valid Pauli types
        operators = [ensure_pauli(op) for op in raw_operators]

        for i, op in enumerate(operators):
            print(f"[DEBUG] Operator {i}: {op} | Type: {type(op)}")

        return cls(model_circuit=model_circuit, operators=operators, **config)

def ensure_pauli(op):
        if isinstance(op, (cirq.PauliString, cirq.PauliSum)):
            return op
        elif isinstance(op, cirq.Operation):
            return cirq.PauliString(op)
        elif isinstance(op, cirq.Qid):
            # Default to Z if you just have a qubit
            return cirq.PauliString(cirq.Z(op))
        else:
            raise TypeError(f"Cannot convert to PauliString: {op!r} (type: {type(op)})")
        
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # cast to float32 for one_hot encode (otherwise TRUE/FALSE tensor)
    return tf.cast(parts[-2] == CLASS_NAMES, tf.float32)

def get_classes(dataset_path):

    # List all subdirectories in the training folder
    train_path_dir = os.path.join(dataset_path, "training/train")

    class_labels = [class_name for class_name in os.listdir(train_path_dir) if
                    os.path.isdir(os.path.join(train_path_dir, class_name))]

    return class_labels

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

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds


class CircuitLayerBuilder:
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout) ** symbol)

def create_qnn_model(img_size):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(img_size, img_size)
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

def convert_to_circuit(image, img_size):
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(img_size, img_size)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

def quantum_preprocess(image, img_size):
    small_image = tf.image.resize(image, (img_size, img_size)).numpy()

    binary_image = np.array(small_image > THRESHOLD, dtype=np.float32)

    circ_image = [convert_to_circuit(x, IMG_DIM) for x in binary_image]

    tfcirc_img = tfq.convert_to_tensor(circ_image)

    del binary_image, circ_image

    return tfcirc_img

def train_validation_phase(x_train_tfcirc, y_train, BATCH_SIZE, EPOCHS, x_val_tfcirc, y_val):

    model_circuit, model_readout = create_qnn_model(IMG_DIM)

    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),

        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        CustomPQC(model_circuit, model_readout),

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

    return model, qnn_training

def test_phase(model, x_test_tfcirc, y_test):
    
    qnn_test = model.evaluate(x_test_tfcirc, y_test, verbose=0)
    
    return qnn_test

def generate_cm(y_test, x_test_tfcirc, model):

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test_tfcirc), axis=1)
    confusion = confusion_matrix(y_true, y_pred)
    
    return confusion

def parse_args():
    parser = argparse.ArgumentParser(prog="QNN", description='Docker to execute Quantum Neural Network')
    group = parser.add_argument_group('Arguments')
    group.add_argument('-d', '--dataset', required=True, type=str, help='dataset')
    group.add_argument('-i', '--image', required=True, type=int, help='image size')
    group.add_argument('-e', '--epochs', required=False, type=int, default=10,
                       help='number of epochs')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=32)
    group.add_argument('-r', '--learning_rate', required=False, type=float, default=0.01,
                       help="Learning rate for training models")
    group.add_argument('-t', '--threshold', required=False, type=float, default=0.5, help='dataset')
    group.add_argument('-lm', '--loadmodel', required=False, type=str, default='', help='load model saved during training')
    group.add_argument('-m', '--mode', required=False, type=str, default='complete', help='mode of execution. complete --> train-val-test; train-val; test',)

    group.set_defaults(classAnalysis=True)
    arguments = parser.parse_args()
    return arguments

args = parse_args()

config.DATASET = args.dataset
config.IMAGE_SIZE = args.image
config.THRESHOLD = args.threshold
config.LEARNING_RATE = args.learning_rate
config.EPOCHS = args.epochs
config.BATCH_SIZE = args.batch_size

dataset_main_folder = os.path.join(APP_ROOT, 'datasets')

if not os.path.exists(dataset_main_folder):
    os.makedirs(dataset_main_folder)

dataset_folder = os.path.join(dataset_main_folder, config.DATASET)

# dataset path declaration
train_path_dir = tf.data.Dataset.list_files(os.path.join(dataset_folder, "training/train/*/*"))
val_path_dir = tf.data.Dataset.list_files(os.path.join(dataset_folder, "training/val/*/*"))
test_path_dir = tf.data.Dataset.list_files(os.path.join(dataset_folder, "test/*/*"))

train_path_dir_elm = os.path.join(dataset_folder, 'training/train')

templist = list()

for folder in os.walk(train_path_dir_elm):
    templist.append(folder[0].replace(train_path_dir_elm, '').replace('/', ''))


# hyperparameter setting
CLASS_NAMES = list(filter(None, templist))
IMG_DIM = config.IMAGE_SIZE
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
LEARNING_RATE = config.LEARNING_RATE
THRESHOLD = config.THRESHOLD
NUM_CLASSES = len(CLASS_NAMES)

print(
    f'Experiment submitted | Batch Size: {str(BATCH_SIZE)}, Epochs: {str(EPOCHS)}, Learning Rate: {str(LEARNING_RATE)},'
    f'THRESHOLD: {str(THRESHOLD)}')

current_timestamp = datetime.datetime.now()
start_time = time.time()

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

x_train_tfcirc = [quantum_preprocess(train, IMG_DIM) for train in x_train]
x_val_tfcirc = [quantum_preprocess(val, IMG_DIM) for val in x_val]
x_test_tfcirc = [quantum_preprocess(test, IMG_DIM) for test in x_test]

x_train_tfcirc = np.concatenate([train.numpy().flatten() for train in x_train_tfcirc])
x_val_tfcirc   = np.concatenate([val.numpy().flatten() for val in x_val_tfcirc])
x_test_tfcirc  = np.concatenate([test.numpy().flatten() for test in x_test_tfcirc])

y_train = np.concatenate([y for y in y_train], axis=0)
y_val = np.concatenate([y for y in y_val], axis=0)
y_test = np.concatenate([y for y in y_test], axis=0)

if args.mode == 'complete':
    
    model, qnn_training = train_validation_phase(x_train_tfcirc, y_train, BATCH_SIZE, EPOCHS, x_val_tfcirc, y_val)

    class_labels = get_classes(dataset_path=dataset_folder)
    qnn_test = test_phase(model, x_test_tfcirc, y_test)
    qnn_test = f'Test Results: Loss {qnn_test[0]}, Accuracy {qnn_test[1]}, Precision {qnn_test[2]}, Recall {qnn_test[3]}, AUC {qnn_test[4]}'
    print(qnn_test)
    confusion = generate_cm(y_test, x_test_tfcirc, model)
    
elif args.mode == 'train-val':
    
    model, qnn_training = train_validation_phase(x_train_tfcirc, y_train, BATCH_SIZE, EPOCHS, x_val_tfcirc, y_val)
    class_labels = get_classes(dataset_path=dataset_folder)
    qnn_test = ''
    confusion = ''
    
elif args.mode == 'test':
    
    qnn_training = ''
    
    model = models.load_model(args.loadmodel, custom_objects={'CustomPQC': CustomPQC})
    
    class_labels = get_classes(dataset_path=dataset_folder)
    qnn_test = test_phase(model, x_test_tfcirc, y_test)
    qnn_test = f'Test Results: Loss {qnn_test[0]}, Accuracy {qnn_test[1]}, Precision {qnn_test[2]}, Recall {qnn_test[3]}, AUC {qnn_test[4]}'
    print(qnn_test)
    confusion = generate_cm(y_test, x_test_tfcirc, model)

end_time = time.time()
total_seconds = end_time - start_time
execution_time = str(datetime.timedelta(seconds=total_seconds))

print(f'Execution Time: {execution_time}')

timestamp = re.sub(r'[.:\s-]', '', str(current_timestamp))
LEARNING_RATE = str(LEARNING_RATE).replace('0.', '')
THRESHOLD = str(THRESHOLD).replace('0.', '')

# Save Results
print('Saving Results...')

dest_folder = save_exp(args.mode, files_folder, BATCH_SIZE, EPOCHS, LEARNING_RATE, timestamp, IMG_DIM, execution_time, THRESHOLD, qnn_training, qnn_test, confusion, class_labels)

model_path = os.path.join(dest_folder, f'model_{timestamp}.h5')
model.save(model_path)

print(f'All files were successfully saved to the following directory: {dest_folder}')