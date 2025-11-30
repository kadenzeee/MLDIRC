#!/usr/bin/env python3


import tensorflow as tf
from tensorflow import keras
import time
import numpy as np
import sys
import os

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

os.environ["ROBCLAS_VERBOSE"] = "0"
os.environ["ROCM_INFO_LEVEL"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(f"[INFO] Environment set")

program_start = time.time()

infile = "8K22TO90DEG.npz"

f = np.load(infile, mmap_mode='r')
TIMES, ANGLES, LABELS = f["TIMES"], f["ANGLES"], f["LABELS"]
nevents = TIMES.shape[0]
input_dim = TIMES.shape[1]

print(f"[INFO] Data size: {sys.getsizeof(TIMES)//10**6} MB")

# ---------------------------------------------------------------
#
#                       PARAMETERS
#
# ---------------------------------------------------------------
class_names = ['Pi+', 'Proton']
num_classes = len(class_names) # Pions or kaons?


batch_size  = 128 # How many events to feed to NN at a time?
nepochs     = 40 # How many epochs?

trainfrac   = 0.75
valfrac     = 0.1
testfrac    = 0.15
# ---------------------------------------------------------------

perm = np.random.permutation(nevents)
TIMES_shuf  = TIMES[perm]
ANGLES_shuf = ANGLES[perm]
LABELS_shuf = LABELS[perm]

trainend    = int(np.floor(nevents * trainfrac))
valend      = int(trainend + np.floor(nevents * valfrac))

traintimes  = TIMES_shuf[:trainend]
trainangles = ANGLES_shuf[:trainend]
trainlabels = LABELS_shuf[:trainend]
valtimes    = TIMES_shuf[trainend:valend]
valangles   = ANGLES_shuf[trainend:valend]
vallabels   = LABELS_shuf[trainend:valend]
testtimes   = TIMES_shuf[valend:]
testangles  = ANGLES_shuf[valend:]
testlabels  = LABELS_shuf[valend:]

class ScheduledFiLM(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ScheduledFiLM, self).__init__(**kwargs)
        self.lambda_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.ranp_rate = 0.01
        

    def call(self, inputs):
        x, gamma, beta = inputs
        lam = tf.clip_by_value(self.lambda_var, 0.0, 1.0)
        return (1.0 + lam * gamma) * x + lam * beta



class ScheduledFiLMCallback(keras.callbacks.Callback):
    def __init__(self, film_layer, nepochs):
        super(ScheduledFiLMCallback, self).__init__()
        self.film_layer = film_layer
        self.nepochs = nepochs

    def on_epoch_end(self, epoch, logs=None):
        new_lambda = (2*epoch + self.nepochs) / (epoch + self.nepochs) - 0.5
        self.film_layer.lambda_var.assign(new_lambda)
        print(f'\nUpdated FiLM lambda to {new_lambda:.4f}')



class SparseBatchGenerator(keras.utils.Sequence):
    """
    Converts sparse matrices to dense batches for training during runtime. Full sparse dataset is loaded into memory, and dense batches are created using this class.
    Class keeps ordering and supports shuffling each epoch.

    
    Parameters
    ----------
    *args : ndarray, sparseMatrix, awkwardArray 
        Sparse or dense matrices to be placed into batches. The arguments should include label data as the last argument.
    >>> train_gen = SparseBatchGenerator(times, angles, labels, *kwargs)
    batch_size : int
        Number of matrices to batch.
    shuffle : bool
        Whether or not to shuffle data on epoch end.
    """

    def __init__(self, *args, batch_size=256, shuffle=True):
        self.args = args
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.args[0].shape[0]
        self.indices = np.arange(self.n)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_list = []
        
        for data in self.args:
            arr = np.asarray(data[batch_idx])
            batch_list.append(arr)
            
        inputs = batch_list[:-1]
        labels = batch_list[-1]


        # keras needs a tuple of the inputs if there are multiple inputs, can't handle list of inputs. 
        if len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs = tuple(inputs)
        
        return inputs, labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


num_nodes = 80

# Time Data Branch
hist_input = keras.Input(shape=(input_dim,))

h = keras.layers.Dropout(0.15)(hist_input)
h = keras.layers.Dense(num_nodes, activation='relu')(h)
h = keras.layers.Dense(num_nodes, activation='relu')(h)


# Angle Data Branch
angle_input = keras.Input(shape=(7,))
a = keras.layers.Dense(num_nodes, activation='relu')(angle_input)
a = keras.layers.Dense(num_nodes, activation='relu')(a)

# Produce FiLM parameters
gamma = keras.layers.Dense(num_nodes, activation='linear', name='gamma')(a)
beta  = keras.layers.Dense(num_nodes, activation='linear', name='beta')(a)
#gamma = keras.layers.Lambda(lambda g: 1.0 + 1.5*g)(gamma)
#beta  = keras.layers.Lambda(lambda b: 1.5*b)(beta)

# FiLM layer
h_mod = keras.layers.Multiply()([h, gamma])
h_mod = keras.layers.Add()([h_mod, beta])

# Combined layers and output
x = keras.layers.Dense(16, activation='relu')(h_mod)
out = keras.layers.Dense(num_classes, activation='softmax', name='output')(x)

model = keras.Model(inputs=[hist_input, angle_input], outputs=out)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.summary()



film = ScheduledFiLM()

train_gen   = SparseBatchGenerator(traintimes, trainangles, trainlabels, batch_size=batch_size, shuffle=True)
val_gen     = SparseBatchGenerator(valtimes, valangles, vallabels, batch_size=batch_size, shuffle=True)
test_gen    = SparseBatchGenerator(testtimes, testangles, testlabels, batch_size=batch_size, shuffle=True)



model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=nepochs, 
    #callbacks=[ScheduledFiLMCallback(film, nepochs)],
)

test_loss, test_acc = model.evaluate(
    test_gen, verbose=2
)

print('\nTest accuracy:', test_acc)
print('Test loss:', test_loss)
program_end = time.time()
print(f"Done in {program_end - program_start}s")