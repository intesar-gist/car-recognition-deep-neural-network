import os
import scipy.io as cio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle
from sklearn import preprocessing
from keras.utils import to_categorical
import helper

# setup paths
standford_car = os.path.expanduser("~") + "/PycharmProjects/datasets/bmw10_release"
matlab_file = standford_car + "/bmw10_annos.mat"
image_folder = standford_car + "/bmw10_ims"
dir_export = standford_car + "/export"
num_classes = 10


def read_matlab_file():
    from scipy.io import loadmat
    file = loadmat(matlab_file)
    train_indices = []
    test_indices = []

    train = file['train_indices']
    for i in train:
        train_indices.append(i[0])

    test = file['test_indices']
    for i in test:
        test_indices.append(i[0])

    images_path_data = file['annos'][0]

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for i, image in enumerate(images_path_data):
        index = i+1

        if index in train_indices:
            x_train.append(image[0][0])
            y_train.append(image[1][0][0]-1)
        elif index in test_indices:
            x_test.append(image[0][0])
            y_test.append(image[1][0][0]-1)

    x_train = process_images(x_train)
    y_train = np.array(y_train)
    x_test = process_images(x_test)
    y_test = np.array(y_test)

    return (x_train, y_train), (x_test, y_test)


def process_images(images_paths):
    images = []
    for path in images_paths:
        img_path = os.path.join(image_folder, path.rstrip("\n\r"))
        images.append(resize_image_nopad(img_path))

    return np.array(images)


def write_file(data_to_dump, file_name):
    print("exporting file: " + file_name + "\n")
    # dumping images and label data for training
    path = os.path.join(dir_export, file_name)
    filehandler = open(path, 'wb')
    pickle.dump(data_to_dump, filehandler)
    filehandler.close()


def read_file(file_name):
    path = os.path.join(dir_export, file_name)
    filehandler = open(path, 'rb')
    object = pickle.load(filehandler)
    filehandler.close()
    return object


# re-sizing images
def resize_image_nopad(img_path, image_size=244):
    img = mpimg.imread(img_path)
    img = img.astype("float32")
    img /= 255.0

    img = cv2.cv2.resize(img, (image_size, image_size))
    return img


def resize_image_pad(img_path, image_size=244):
    im = mpimg.imread(img_path)
    desired_size = image_size

    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_im


def load_labels():
    labels = ["Folder 1", "Folder 2", "Folder 3", "Folder 4", "Folder 5", "Folder 6", "Folder 7", "Folder 8",
              "Folder 10", "Folder 11"]
    return labels


def load_data():
    print("Loading stanford car data..\n")

    training_data = read_file("training_data")
    testing_data = read_file("testing_data")

    print("\nTraining Data shape: " + str(training_data['data'].shape))
    print("Testing Data shape: " + str(testing_data['data'].shape))

    return (training_data['data'], training_data['label']), (testing_data['data'], testing_data['label'])


def read_process_export_data():
    print("Reading data from matlab file and images folders")
    (x_train, y_train), (x_test, y_test) = read_matlab_file()

    # exporting car images and labels
    train_data = {'data': x_train, 'label': y_train}
    write_file(train_data, "training_data")
    test_data = {'data': x_test, 'label': y_test}
    write_file(test_data, "testing_data")# # reading files

    # data1 = read_file('test_data_export')
    print(x_train.shape)
    plt.imshow(x_train[0])
    plt.show()


def setup_and_load_data(verbose=False):
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = load_data()
    if verbose:
        print("x_train shape: {}, {} train samples, {} test samples.\n".format(
            x_train.shape, x_train.shape[0], x_test.shape[0]))

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Load label names to use in prediction results
    labels = load_labels()

    return x_train, y_train, x_test, y_test, labels

def test_data():
    x_train, y_train, x_test, y_test, labels = setup_and_load_data(verbose=True)
    indices = [np.random.choice(range(len(x_train))) for i in range(36)]
    fig = helper.cifar_grid(x_train, y_train, indices, 6, labels)
    fig.show()
