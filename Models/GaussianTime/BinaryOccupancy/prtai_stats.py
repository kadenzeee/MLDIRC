import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker



program_start = time.time()

infile = "GaussianTime/BinaryOccupancy/80DEG.npz"

with np.load(infile) as f:
    TIMES, LABELS = f["TIMES"], f["LABELS"]
    nevents = TIMES.shape[0]

# ---------------------------------------------------------------
#
#                       PARAMETERS
#
# ---------------------------------------------------------------

classnames = ['Pi+', 'Proton']
numclasses = len(classnames) # Pions or kaons?


batchsize  = 64 # How many events to feed to NN at a time?
nepochs    = 10 # How many epochs?

# Splits
trainfrac  = 0.7
valfrac    = 0.15
testfrac   = 0.15

# ---------------------------------------------------------------

trainend   = int(nevents*trainfrac)
valend     = trainend + int(nevents*valfrac)

X_train, Y_train    = TIMES[:trainend]       , LABELS[:trainend]
X_val, Y_val        = TIMES[trainend:valend] , LABELS[trainend:valend]
X_test, Y_test      = TIMES[valend:]         , LABELS[valend:]

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_ds = train_ds.batch(batchsize)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

val_ds   = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_ds   = val_ds.batch(batchsize)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)

test_ds  = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_ds  = test_ds.batch(batchsize)
test_ds  = test_ds.prefetch(tf.data.AUTOTUNE)

print(f'Training on {trainend} events')
input_dim = TIMES.shape[1]
print(f'Input dimension: {input_dim}')

search = 64
heatmap = np.zeros((search, search))
sizes = np.zeros((search, search))

for i in range(0, search):
    for j in range(0, search):

        model = tf.keras.Sequential(
            [tf.keras.layers.Flatten(input_shape=(input_dim,))] +
            [tf.keras.layers.Dense(i + 1, activation='relu') for _ in range(j + 1)] +
            [tf.keras.layers.Dense(numclasses)]
            )

        modelsize = sum(np.prod(w.shape) for w in model.trainable_weights)


        model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

        model.fit(train_ds, validation_data=val_ds, epochs=nepochs)

        test_loss, test_acc = model.evaluate(test_ds, verbose=2)
        heatmap[i, j] = test_acc
        sizes[i, j] = modelsize

program_end = time.time()

print(heatmap)
print(sizes)
fig, ax = plt.subplots(figsize=(10,7))
map = ax.pcolor(heatmap, vmin=0.5, vmax=1)
ax.set_xlabel('Number of Nodes')
ax.set_ylabel('Number of Layers')

loc = plticker.MultipleLocator(base=1, offset=0.5); ax.xaxis.set_major_locator(loc); ax.yaxis.set_major_locator(loc); 
f = lambda x, _: int(x + 0.5); ax.xaxis.set_major_formatter(plticker.FuncFormatter(f)); ax.yaxis.set_major_formatter(plticker.FuncFormatter(f))

traditional_acc = 0.86
max_acc = np.max(heatmap)

cbar = fig.colorbar(map, ax=ax)
cbar.ax.hlines(traditional_acc, 0, 1, colors='k', linestyles='-', linewidth=1)
cbar.ax.text(2, traditional_acc, 'Time Imaging', va='center', ha='left', fontsize=8, transform=cbar.ax.transData)
cbar.ax.hlines(max_acc, 0, 1, colors='k', linestyles='-', linewidth=1)
cbar.ax.text(2, max_acc, 'DNN', va='center', ha='left', fontsize=8, transform=cbar.ax.transData)

plt.savefig(f'accuracies.png')

fig, ax = plt.subplots(figsize=(10,7))
map = ax.pcolor(sizes)
ax.set_title('Number of Parameters')
ax.set_xlabel('Number of Nodes')
ax.set_ylabel('Number of Layers')

loc = plticker.MultipleLocator(base=1, offset=0.5); ax.xaxis.set_major_locator(loc); ax.yaxis.set_major_locator(loc); 
f = lambda x, _: int(x + 0.5); ax.xaxis.set_major_formatter(plticker.FuncFormatter(f)); ax.yaxis.set_major_formatter(plticker.FuncFormatter(f))

max_acc = np.max(sizes)

cbar = fig.colorbar(map, ax=ax)

plt.savefig(f'modelsizes.png')


print(f'Done in {program_end-program_start} s')