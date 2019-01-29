import os
import scipy.io as cio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import pickle
from sklearn import preprocessing

# setup paths
comp_car = os.path.expanduser("~") + "/PycharmProjects/datasets/CompCars/data"
dir_data_split = comp_car + "/train_test_split/classification/bmw"
dir_export = comp_car + "/train_test_split/classification"
dir_misc = comp_car + "/misc"
dir_images = comp_car + "/image"

# export files namings
files_to_export = {"train_data_export": "/train_bmw.txt", "test_data_export": "/test_bmw.txt"}
model_names_dictionary = "labels_dictionary_export"


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


def load_bmw_models_dict():
    mat = cio.loadmat(dir_misc + '/make_model_name.mat')
    key = 'model_names'
    mat = mat[key]
    bmw_model_dict = {}

    # only consider BMW cars
    for i in range(len(mat)):
        if(i >= 67 and i <= 122):
            if(mat[i][0].size > 0):
                bmw_model_dict[i+1] = mat[i][0][0]

    print("Total BMW Cars in CompCar dataset: " + str(len(bmw_model_dict)) + "\n")

    return bmw_model_dict


# re-sizing images
def load_and_process_image(img_path, image_size=255):
    img = mpimg.imread(img_path)
    img = img.astype("float32")
    img /= float(image_size)

    img = cv2.cv2.resize(img, (image_size, image_size))
    return img


# returns numpy array of images and corresponding labels
def load_bmw_data(file_name):
    img_matrix = []
    car_model_labels = []

    f = open(dir_data_split + file_name, "r")
    for img_path_data in f:
        path_data = img_path_data.split("/")
        car_model_id = path_data[1]
        #car_model_year = path_data[2]
        #car_img_name = path_data[3].rstrip("\n\r")

        car_model_labels.append(car_model_id)

        # load images from given path
        img_path = os.path.join(dir_images, img_path_data.rstrip("\n\r"))
        img_matrix.append(load_and_process_image(img_path))

    f.close()

    # convert to numpy array
    img_matrix = np.array(img_matrix)
    car_model_labels = np.array(car_model_labels)

    print("shape of images: {}, {} shape of labels, {} num. of unique labels/classes.\n".format(
        img_matrix.shape, car_model_labels.shape, len(set(car_model_labels))))

    return img_matrix, car_model_labels


def load_labels():
    return read_file(model_names_dictionary)


def hot_encode_labels(train_labels, test_labels):

        if np.array_equal(set(train_labels), set(test_labels)):
            print("\n train and test labels/classes are exactly equal, We good!!! \n")
        else:
            print("\nDiscrepancies between train and test labels/classes, program halted !!!!! \n")

        le = preprocessing.LabelEncoder()

        # fitting transforming on train labels
        transformed_train_labels = le.fit_transform(train_labels)

        # transforming test labels assuming it is subset or exactly equals of train labels
        transformed_test_labels = le.transform(test_labels)

        return transformed_train_labels, transformed_test_labels, len(set(transformed_train_labels)), le


def load_data():
    x_train, y_train, x_test, y_test = ([] for i in range(4))
    for output_file_name in files_to_export.keys():
        print("reading file: " + output_file_name)
        data = read_file(output_file_name)
        print("shape of images: {}, {} shape of labels, {} num. of unique labels/classes.\n".format(
            data['data'].shape, data['label'].shape, len(set(data['label']))))

        if output_file_name == "train_data_export":
            x_train = data['data']
            y_train = data['label']
        else:  # test_data_export
            x_test = data['data']
            y_test = data['label']

    y_train, y_test, num_classes, label_encoder = hot_encode_labels(y_train, y_test)

    return (x_train, y_train), (x_test, y_test), num_classes, label_encoder


# # exporting labels dictionary
# model_names = load_bmw_models_dict()
# write_file(model_names, model_names_dictionary)
#
# # exporting car images and labels
# for output_name, file_name in files_to_export.items():
#     images, car_model_labels = load_bmw_data(file_name)
#     data = {'data': images, 'label': car_model_labels}
#     write_file(data, output_name)


# # reading files
# data1 = read_file('test_data_export')
# print(data1['data'].shape)
# plt.imshow(data1['data'][300])
# plt.show()

