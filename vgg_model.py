from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
import cv2, numpy as np
import load_data
from keras.utils import to_categorical
import load_data as compcar


def setup_load_compcar(verbose=False):
    global label_encoder
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test), num_classes, label_encoder = compcar.load_data()
    if verbose:
        print("x_train shape: {}, {} train samples, {} test samples.\n".format(
            x_train.shape, x_train.shape[0], x_test.shape[0]))

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Load label names to use in prediction results
    labels = compcar.load_labels()

    return x_train, y_train, x_test, y_test, labels


def VGG_16(x_train, weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=x_train.shape[1:]))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

# im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
# im[:,:,0] -= 103.939
# im[:,:,1] -= 116.779
# im[:,:,2] -= 123.68
# im = im.transpose((2,0,1))
# im = np.expand_dims(im, axis=0)

batch_size = 128
num_classes = None
epochs_shortrun = 5
epochs_longrun = 500
label_encoder = None
x_train, y_train, x_test, y_test, labels = setup_load_compcar(verbose=True)

# Test pretrained model
model = VGG_16(x_train)
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])

# early stop callback, given a bit more leeway
stahp = EarlyStopping(min_delta=0.00001, patience=25)

############################
## FITTING THE DATA TO model
############################

epochs = epochs_longrun
hist = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[stahp], batch_size=128)

print(np.argmax(hist))