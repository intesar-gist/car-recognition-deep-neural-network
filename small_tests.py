import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

im = "/Users/ihaider/PycharmProjects/datasets/CompCars/data/image/11/739/2010/d7a8728fa79260.jpg"


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

    new_im = new_im.astype("float32")
    new_im /= float(255)

    return new_im


# re-sizing images
def resize_image_nopad(img_path, image_size=244):
    img = mpimg.imread(img_path)
    img = img.astype("float32")
    img /= float(image_size)

    img = cv2.cv2.resize(img, (image_size, image_size))
    return img


def test_image_augmentation():
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

    img = resize_image_pad(im)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='results', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 50:
            break  # otherwise the generator would loop indefinitely


import numpy as np
from scipy import ndimage
a = np.array([[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0],
              [0, 0, 1, 1, 1, 0, 0],
              [0, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]])

# Find the location of all objects
objs = ndimage.find_objects(a)

# Get the height and width
height = int(objs[0][0].stop - objs[0][0].start)
width = int(objs[0][1].stop - objs[0][1].start)

print(str(height)+ '----' + str(width))

plt.imshow(objs)
plt.show()