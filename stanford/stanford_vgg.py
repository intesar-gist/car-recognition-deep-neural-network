from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.optimizers import SGD
from keras.layers.pooling import MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
import cv2, numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import stanford.load_data_stanford as ld
import helper

label_encoder = None
batch_size = 128
num_classes = None
epochs_shortrun = 5
epochs_longrun = 500
label_encoder = None

save_dir = os.path.expanduser("~") + "/PycharmProjects/car-recognition-cnn/stanford/work"
res_dir = os.path.expanduser("~") + "/PycharmProjects/car-recognition-cnn/stanford/results"
model_name = "vgg_stanford_bmw"

ckpt_dir = os.path.join(save_dir,"checkpoints")
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)

model_picture_path = os.path.join(res_dir, model_name + ".svg")
model_path = os.path.join(res_dir, model_name + ".kerasave")
hist_path = os.path.join(res_dir, model_name + ".kerashist")


def VGG_16(input_img_shape):
    print("Input image shape: " + str(input_img_shape) + "\n")
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=input_img_shape))
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
    model.add(Dense(10, activation='softmax'))

    return model

# im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
# im[:,:,0] -= 103.939
# im[:,:,1] -= 116.779
# im[:,:,2] -= 123.68
# im = im.transpose((2,0,1))
# im = np.expand_dims(im, axis=0)

########################################
## # LOADING DATA FROM FILES
########################################
x_train, y_train, x_test, y_test, labels = ld.setup_and_load_data(verbose=True)


# #######################################
# # SHOWING RANDOM CARD LOADED IN A GRID
# #######################################
# indices = [np.random.choice(range(len(x_train))) for i in range(36)]
# fig = helper.cifar_grid(x_train,y_train,indices,6, labels)
# fig.show()


################################
## CREATING AND COMPILING MODEL
################################
model = VGG_16(x_train.shape[1:])
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])


##########################################
## SETTING UP CHECKPOINTS & EARLY STOPPING
##########################################
filepath = os.path.join(ckpt_dir, "weights-improvement-{epoch:02d}-{val_acc:.6f}.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
print("Will save improvement checkpoints to \n\t{0}".format(filepath))

# early stop callback, given a bit more leeway
stahp = EarlyStopping(min_delta=0.00001, patience=200)


############################
## FITTING THE DATA TO model
############################

cpf = helper.last_ckpt(ckpt_dir)
if cpf != "":
    print("Loading starting weights from \n\t{0}".format(cpf))
    model.load_weights(cpf)

epochs = epochs_longrun
hist = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), callbacks=[checkpoint, stahp],
                 batch_size=64)


############################
## SAVE MODEL & WEIGHTS
############################
model.save(model_path)
print('Saved trained model at %s ' % model_path)

with open(hist_path, 'wb') as f:
    pickle.dump(hist.history, f)


print(np.argmax(hist))