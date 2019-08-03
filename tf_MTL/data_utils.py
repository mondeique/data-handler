
import tensorflow as tf
import numpy as np
import random
import scipy
from typing import Tuple


def gauss(mean, stddev, ksize):
    """
    Use Tensorflow to compute a Gaussian Kernel.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian (e.g. 0.0).
    stddev : float
        Standard Deviation of the Gaussian (e.g. 1.0).
    ksize : int
        Size of kernel (e.g. 16).

    Returns
    -------
    kernel : np.ndarray
        Computed Gaussian Kernel using Tensorflow.
    """
    g = tf.Graph()
    with tf.Session(graph=g):
        x = tf.linspace(-3.0, 3.0, ksize)
        z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0) /
                           (2.0 * tf.pow(stddev, 2.0)))) *
             (1.0 / (stddev * tf.sqrt(2.0 * 3.1415))))
        return z.eval()


def gauss2d(mean, stddev, ksize):
    """Use Tensorflow to compute a 2D Gaussian Kernel.

    Parameters
    ----------
    mean : float
        Mean of the Gaussian (e.g. 0.0).
    stddev : float
        Standard Deviation of the Gaussian (e.g. 1.0).
    ksize : int
        Size of kernel (e.g. 16).

    Returns
    -------
    kernel : np.ndarray
        Computed 2D Gaussian Kernel using Tensorflow.
    """
    z = gauss(mean, stddev, ksize)
    g = tf.Graph()
    with tf.Session(graph=g):
        z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))
        return z_2d.eval()


def random_rotation(img: tf.Tensor, max_rotation: float = 0.1, crop: bool = False) -> tf.Tensor:
    """
    Rotates an image with a random angle.

    :param img: (tf.Tensor) input image to apply random rotation
    :param max_rotation: (float) maximum angle to rotate (radians)
    :param crop: (boolean) to crop or not the image after rotation

    :return: rotated image

    """
    with tf.name_scope('RandomRotation'):
        rotation = tf.random_uniform([], -max_rotation, max_rotation, name='pick_random_angle')
        rotated_image = tf.contrib.image.rotate(img, rotation, interpolation='BILINEAR')
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2*rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h-new_h)/2), tf.int32), tf.cast(tf.ceil((w-new_w)/2), tf.int32)
            # Test sliced
            rotated_image_crop = tf.cond(
                tf.logical_and(bb_begin[0] < h - bb_begin[0], bb_begin[1] < w - bb_begin[1]),
                true_fn=lambda: rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :],
                false_fn=lambda: img,
                name='check_slices_indices'
            )

            # If crop removes the entire image, keep the original image
            rotated_image = tf.cond(tf.equal(tf.size(rotated_image_crop), 0),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop,
                                    name='check_size_crop')

        return rotated_image


def random_padding(image: tf.Tensor, max_pad_w: int = 10, max_pad_h: int = 10) -> tf.Tensor:
    """
    Given an image will pad its border adding a random number of rows and columns

    :param image: image to pad
    :param max_pad_w: maximum padding in width
    :param max_pad_h: maximum padding in height

    :return: a padded image
    """

    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    return tf.pad(image, paddings, mode='REFLECT', name='random_padding')


def random_scaling(image: tf.Tensor, minscale: float = 0.8, maxscale: float = 1.2) -> tf.Tensor:
    """
    Randomly scales the images between 0.8 to 1.2 times the original size.

    :param image: (tf.Tensor) input image to apply random-scaling
    :param minscale: (float) minimum size of scaling
    :param maxscale: (float) maximum size of scaling

    :return: (tf.Tensor) scalied image
    """

    scale = tf.random_uniform([1], minval=minscale, maxval=maxscale, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]))
    scaled_image = tf.image.resize_images(image, new_shape)

    return scaled_image


def augment_data(image: tf.Tensor, max_rotation: float = 0.1) -> tf.Tensor:
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)

    :param image: (tf.Tensor) input image
    :param max_rotation: (float) maximum permitted rotation (in radians)

    :return: image: (tf.Tensor) output image

    """
    with tf.name_scope('DataAugmentation'):

        # Random padding
        image = random_padding(image)

        image = random_scaling(image, minscale=0.8, maxscale=1.2)
        image = random_rotation(image, max_rotation, crop=True)

        return image


def padding_inputs(image: tf.Tensor, target_shape: Tuple[int, int, int]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given an input image, will pad it to return a target_shape size padded image.

    There are 4 cases:
         - image size > target size : simple resizing to shrink the image
         - image width > target width & image height < target height : pad the image height
         - image width < target width & image height > target height : pad the image width
         - image size <= target size : pad the image width & height

    :param image: (tf.Tensor) Tensor of shape [H,W,C]
    :param target_shape: (list) final shape after padding [NEW_H,NEW_W,C]

    :return: (image padded, output shape)
    """

    # target_ratio = target_shape[1]/target_shape[0]
    # Compute ratio to keep the same ratio in new image and get the size of padding
    # necessary to have the final desired shape
    shape = tf.shape(image)
    # ratio = tf.divide(shape[1], shape[0], name='ratio')
    new_h = shape[0]
    new_w = shape[1]
    target_h = target_shape[0]
    # new_w = tf.cast(tf.round((ratio * new_h) / increment) * increment, tf.int32)
    # f1 = lambda: (new_w, ratio)
    # f2 = lambda: (new_h, tf.constant(1.0, dtype=tf.float64))
    # new_w, ratio = tf.case({tf.greater(new_w, 0): f1,
    #                         tf.less_equal(new_w, 0): f2},
    #                        default=f1, exclusive=True)
    target_w = target_shape[1]

    # Definitions for cases
    def pad_height_fn():
        with tf.name_scope('mirror_padding'):
            pad_h = tf.subtract(target_h, new_h)

            img_resized = tf.image.resize_images(image, [new_h, target_shape[1]])

            # Padding to have the desired width
            paddings = [[0, pad_h], [0, 0], [0, 0]]
            pad_height_image = tf.pad(img_resized, paddings, mode='SYMMETRIC', name=None)

            # Set manually the shape
            pad_height_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_height_image, (new_h, new_w)

    def pad_width_fn():
        with tf.name_scope('mirror_padding'):
            pad_w = tf.subtract(target_w, new_w)

            img_resized = tf.image.resize_images(image, [target_shape[0], new_w])

            # Padding to have the desired width
            paddings = [[0, 0], [0, pad_w], [0, 0]]
            pad_width_image = tf.pad(img_resized, paddings, mode='SYMMETRIC', name=None)

            # Set manually the shape
            pad_width_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_width_image, (new_h, new_w)

    def pad_fn():
        with tf.name_scope('mirror_padding'):
            pad_h = tf.subtract(target_h, new_h)
            pad_w = tf.subtract(target_w, new_w)

            img_resized = tf.image.resize_images(image, [new_h, new_w])

            # Padding to have the desired width
            paddings = [[0, pad_h], [0, pad_w], [0, 0]]
            pad_image = tf.pad(img_resized, paddings, mode='SYMMETRIC', name=None)

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def simple_resize():
        with tf.name_scope('simple_resize'):
            img_resized = tf.image.resize_images(image, target_shape)

            img_resized.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return img_resized, tuple(target_shape)

    # 4 cases
    final_image, (new_h, new_w) = tf.case(
        {  # case 1 : new_size >= target_size
            tf.logical_and(tf.greater_equal(new_w, target_w), tf.less(new_h, target_h)): simple_resize,
            # case 2 : new_w >= target_w & new_h < target_h
            tf.logical_and(tf.greater_equal(new_h, target_h), tf.less(new_w, target_w)): pad_height_fn,
            # case 3 : new_w < target_w & new_h > target_h
            tf.logical_and(tf.greater_equal(new_w, target_w), tf.less(new_h, target_h)): pad_width_fn,
            # case 4 : new_size < target_size
            tf.logical_and(tf.less(new_h, target_h), tf.less(new_w, target_w)): pad_fn
        },
        default=simple_resize, exclusive=True)
    return final_image, final_image.shape


################################ Numpy Functions #######################################

def random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])
    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad, mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
    return new_batch


def random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch


def random_flip_updown(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.flipud(batch[i])
    return batch


def random_90degrees_rotation(batch, rotations=[0, 1, 2, 3]):
    for i in range(len(batch)):
        num_rotations = random.choice(rotations)
        batch[i] = np.rot90(batch[i], num_rotations)
    return batch


def random_rotation(batch, max_angle):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            angle = random.uniform(-max_angle, max_angle)
            batch[i] = scipy.ndimage.interpolation.rotate(batch[i], angle, reshape=False)
    return batch


def random_blur(batch, sigma_max=5.0):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            sigma = random.uniform(0., sigma_max)
            batch[i] = scipy.ndimage.filters.gaussian_filter(batch[i], sigma)
    return batch


def augmentation(batch, img_size):
    batch = random_crop(batch, (img_size, img_size), 10)
    #batch = random_blur(batch)
    batch = random_flip_leftright(batch)
    batch = random_rotation(batch, 10)

    return batch


def normalize(x, s = 0.1):
    """
    Normalize the image range for visualization

    :param x: (np.array) input array to apply normalization
    :param s: (float) percentage of normalization

    :return: (np.array) normalized array
    """
    return np.uint8(np.clip(
        (x - x.mean()) / max(x.std(), 1e-4) * s + 0.5,
        0, 1) * 255)
