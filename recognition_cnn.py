import numpy as np
import tensorflow as tf
import dill as pickle
from math import *
from keras.datasets import cifar10
from keras.utils import to_categorical

''''
The CIFAR-10 dataset consists of 
- 60000 32x32 colour images 
- 10 classes
- with 6000 images per class
- There are 50000 training images and 
- 10000 test images. 

http://www.cs.toronto.edu/~kriz/cifar.html
'''

# setup paths
import os

batch_size = 128
num_classes = 10
epochs_shortrun = 5
epochs_longrun = 500

save_dir = os.path.expanduser("~") + "/PycharmProjects/self-tests/work"
res_dir = os.path.expanduser("~") + "/PycharmProjects/self-tests/results"
model_name = "convnet_cifar10"


ckpt_dir = os.path.join(save_dir,"checkpoints")
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)

model_picture_path = os.path.join(res_dir, model_name + ".svg")
model_path = os.path.join(res_dir, model_name + ".kerasave")
hist_path = os.path.join(res_dir, model_name + ".kerashist")

def setup_tf():
    # set random seeds for reproducibility
    tf.reset_default_graph()
    tf.set_random_seed(343)
    np.random.seed(343)


def setup_load_cifar(verbose=False):
    datadir = os.path.expanduser("~") + "/.keras/datasets/"
    datafile = datadir + "cifar-10-batches-py.tar.gz"  # the name keras looks for

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if verbose:
        print("x_train shape: {}, {} train samples, {} test samples.\n".format(
            x_train.shape, x_train.shape[0], x_test.shape[0]))

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0

    # Load label names to use in prediction results
    label_list_path = "datasets/cifar-10-batches-py/batches.meta"

    keras_dir = os.path.expanduser(os.path.join("~", ".keras"))
    datadir_base = os.path.expanduser(keras_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join("/tmp", ".keras")
    label_list_path = os.path.join(datadir_base, label_list_path)

    with open(label_list_path, mode="rb") as f:
        labels = pickle.load(f)

    return x_train, y_train, x_test, y_test, labels

def setup_data_aug():
    print("Using real-time data augmentation.\n")
    # This will do preprocessing and realtime data augmentation:
    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range
        # (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally
        # (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically
        # (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False  # randomly flip images
    )

    return datagen

# Function to find latest checkpoint file
def last_ckpt(dir):
    fl = os.listdir(dir)
    fl = [x for x in fl if x.endswith(".hdf5")]
    cf = ""
    if len(fl) > 0:
        accs = [float(x.split("-")[3][0:-5]) for x in fl]
        m = max(accs)
        iaccs = [i for i, j in enumerate(accs) if j == m]
        fl = [fl[x] for x in iaccs]
        epochs = [int(x.split("-")[2]) for x in fl]
        cf = fl[epochs.index(max(epochs))]
        cf = os.path.join(dir, cf)

    return cf

# Visualizing CIFAR 10, takes indicides and shows in a grid
def cifar_grid(X, Y, inds, n_col, predictions=None):
    import matplotlib.pyplot as plt
    if predictions is not None:
        if Y.shape != predictions.shape:
            print("Predictions must equal Y in length!\n")
            return (None)
    N = len(inds)
    n_row = int(ceil(1.0 * N / n_col))
    fig, axes = plt.subplots(n_row, n_col, figsize=(10, 10))

    clabels = labels["label_names"]
    for j in range(n_row):
        for k in range(n_col):
            i_inds = j * n_col + k
            i_data = inds[i_inds]

            axes[j][k].set_axis_off()
            if i_inds < N:
                axes[j][k].imshow(X[i_data, ...], interpolation="nearest")
                label = clabels[np.argmax(Y[i_data, ...])]
                axes[j][k].set_title(label)
                if predictions is not None:
                    pred = clabels[np.argmax(predictions[i_data, ...])]
                    if label != pred:
                        label += " n"
                        axes[j][k].set_title(pred, color="red")

    fig.set_tight_layout(True)
    return fig


######################


x_train, y_train, x_test, y_test, labels = setup_load_cifar(verbose=True)

#indices = [np.random.choice(range(len(x_train))) for i in range(36)]
#fig = cifar_grid(x_train,y_train,indices,6)
#fig.show()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
datagen = setup_data_aug()
# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)




######################

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation="relu",
                 input_shape=x_train.shape[1:]))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


# initiate Adam optimizer
opt = Adam(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# checkpoint callback
filepath = os.path.join(ckpt_dir, "weights-improvement-{epoch:02d}-{val_acc:.6f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
print("Saving improvement checkpoints to \n\t{0}".format(filepath))

# early stop callback, given a bit more leeway
stahp = EarlyStopping(min_delta=0.00001, patience=25)


############################
## FITTING THE DATA TO model
############################
#
# epochs = epochs_longrun
#
# cpf = last_ckpt(ckpt_dir)
# if cpf != "":
#   print("Loading starting weights from \n\t{0}".format(cpf))
#   model.load_weights(cpf)
#
# # Fit the model on the batches generated by datagen.flow().
# hist = model.fit_generator(datagen.flow(x_train, y_train,
#     batch_size=batch_size,shuffle=True),
#     steps_per_epoch=x_train.shape[0] // batch_size,
#     epochs=epochs,verbose=2,
#     validation_data=(x_test, y_test),
#     workers=4, callbacks=[checkpoint,stahp])
#
# # Save model and weights
# model.save(model_path)
# #print('Saved trained model at %s ' % model_path)
#
# with open(hist_path, 'wb') as f:
#   pickle.dump(hist.history, f)


######################

# from keras.utils import plot_model
# plot_model(model, to_file=model_picture_path,
#            show_layer_names=True, show_shapes=True, rankdir="TB")
#print(model.summary())
#################


##########################################
####### Evaluation & Predictions #########
##########################################


# from keras.models import load_model
#
# x_train, y_train, x_test, y_test, labels = setup_load_cifar()
# datagen = setup_data_aug()
# datagen.fit(x_train)
#
# model = load_model(model_path)

# # Evaluate model with test data set
# evaluation = model.evaluate_generator(datagen.flow(x_test, y_test,
#     batch_size=batch_size, shuffle=False),
#     steps=x_test.shape[0] // batch_size, workers=4)
#
# # Print out final values of all metrics
# key2name = {'acc':'Accuracy', 'loss':'Loss',
#     'val_acc':'Validation Accuracy', 'val_loss':'Validation Loss'}
# results = []
# for i,key in enumerate(model.metrics_names):
#     results.append('%s = %.2f' % (key2name[key], evaluation[i]))
# print(", ".join(results))


# # Predicting on the model
# num_predictions = 36
# predict_gen = model.predict_generator(datagen.flow(x_test, y_test,
#     batch_size=batch_size, shuffle=False),
#     steps=(x_test.shape[0] // batch_size)+1, workers=4)
#
# indices = [np.random.choice(range(len(x_test)))
#            for i in range(num_predictions)]
#
# cifar_grid(x_test,y_test,indices,6, predictions=predict_gen).show()






import matplotlib.pyplot as plt

with open(hist_path, 'rb') as f:
  hist = pickle.load(f)

key2name = {'acc':'Accuracy', 'loss':'Loss',
    'val_acc':'Validation Accuracy', 'val_loss':'Validation Loss'}

fig = plt.figure()

things = ['acc','loss','val_acc','val_loss']
for i,thing in enumerate(things):
  trace = hist[thing]
  plt.subplot(2,2,i+1)
  plt.plot(range(len(trace)),trace)
  plt.title(key2name[thing])

fig.set_tight_layout(True)
fig.show()