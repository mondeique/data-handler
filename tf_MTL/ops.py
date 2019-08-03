import tensorflow as tf
import numpy as np
from .const import *
import requests
import os
import sys


def download(path):
    """
    Use urllib to download a file

    :param path: (str) url to download

    :return: (str) Location of downloaded file.

    """
    filename = path.split('/')[-1]
    if os.path.exists(filename):
        return filename
    print('Downloading ' + path)
    with open(filename, "wb") as f:
        response = requests.get(path, stream=True)
        total_length = response.headers.get('content-length')
        if total_length is None:
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s] %.2f of %.2f MB" % (
                    '=' * done,
                    ' ' * (50-done),
                    dl / 1024.0 / 1024.0,
                    total_length / 1024.0 / 1024.0))
                sys.stdout.flush()
    return filename


def download_and_extract_tar(path, dst='./'):
    """
    Download and extract tar file from path

    :param path: (str) Url to tar file to download
    :param dst: (str) Location where tar file extracted

    """
    import tarfile
    filepath = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
        tarfile.open(filepath, 'r:gz').extractall(dst)


def download_and_extract_zip(path, dst='./'):
    """
    Download and extract zip file from path

    :param path: (str) Url to zip file to download
    :param dst: (str) Location where tar file extracted

    """
    import zipfile
    filepath = download(path)
    if not os.path.exists(dst):
        os.makedirs(dst)
        zf = zipfile.ZipFile(file=filepath)
        zf.extractall(dst)


def interp(l, r, n_samples):
    """
    Interpolate between the arrays l and r, n_sample times.

    :param l: (np.ndarray) left side index of numpy array
    :param r: (np.ndarray) right side index of numpy array
    :param n_samples: (int) number of interpolation samples

    :return: (np.array) Interpolated array

    """
    return np.array([
        l + step_i / (n_samples - 1) * (r - l)
        for step_i in range(n_samples)])


def weight_variable(shape, **kwargs):
    """
    Helper function to create a weight variable initialized with a normal distribution.

    :param shape: (list) size of weight variable
    :param kwargs: keyword argument -> key=value 의 형태로 이루어짐

    :return: variable tensor

    """
    if isinstance(shape, list):
        initial = tf.random_normal(tf.stack(shape), mean=0.0, stddev=WEIGHT_INIT)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=WEIGHT_INIT)
    return tf.Variable(initial, **kwargs)


def bias_variable(shape, **kwargs):
    """
    Helper function to create a bias variable initialized with a normal distribution.

    :param shape: (list) size of bias variable
    :param kwargs: keyword argument -> key=value 의 형태로 이루어짐

    :return: variable tensor

    """
    if isinstance(shape, list):
        initial = tf.random_normal(tf.stack(shape), mean=0.0, stddev=WEIGHT_INIT)
        initial.set_shape(shape)
    else:
        initial = tf.random_normal(shape, mean=0.0, stddev=WEIGHT_INIT)
    return tf.Variable(initial, **kwargs)


def conv2d(name, x, filter_size, in_filters, out_filters, strides):
    """
    Helper function to create 2d convolution operation.

    :param x: (tf.Tensor) input tensor to convolve
    :param filter_size: (int) filter size ex) 3*3 filter => filter size = 3
    :param in_filters: (int) number of input tensor
    :param out_filters: (int) number of output tensor
    :param strides: (int) how much you want to stride

    :return: (tf.Tensor) output tensor with convolution

    """
    with tf.variable_scope(name):
        # n = filter_size * filter_size * out_filters
        filter = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], tf.float32,
                                 tf.random_normal_initializer(stddev=WEIGHT_INIT))
        return tf.nn.conv2d(x, filter, [1, strides, strides, 1], 'SAME')


def lrelu(x, leak=0.2):
    """
    Helper function to apply leaky-relu

    :param x: (tf.Tensor) input tensor to apply leaky-relu
    :param leak: (optional) (float) percentage of leaky

    :return: (tf.Tensor) output tensor that apply leaky-relu

    """
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def fc_layer(name, x, out_dim, activation='relu'):
    """
    Helper function to apply fully connected

    :param x: (tf.Tensor) input tensor to connect
    :param out_dim: (int) output dimension to apply
    :param keep_rate: (float) dropout ratio
    :param activation: (str) activation name (relu, softmax, linear)

    :return: (tf.Tensor) output tensor that connected

    """
    assert (activation == 'relu') or (activation == 'softmax') or (activation == 'linear')
    with tf.variable_scope(name):
        dim = x.get_shape().as_list()
        dim = np.prod(dim[1:])
        x = tf.reshape(x, [-1, dim])
        w = tf.get_variable('DW', [x.get_shape()[1], out_dim],
                            initializer=tf.random_normal_initializer(stddev=WEIGHT_INIT))
        b = tf.get_variable('bias', [out_dim], initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, w, b)
        if activation == 'relu':
            x = lrelu(x)
        else:
            if activation == 'softmax':
                x = tf.nn.softmax(x)

        if activation != 'relu':
            return x
        else:
            return tf.nn.dropout(x, DROP_OUT_PROB)


def max_pool(x, filter, stride):
    """
    Helper function to apply max pooling

    :param x: (tf.Tensor) input tensor to apply max pooling
    :param filter: (int) size of max-pooling filter
    :param stride: (int) number of stride

    :return: (tf.Tensor) output tensor that applied max pooling

    """
    return tf.nn.max_pool(x, [1, filter, filter, 1], [1, stride, stride, 1], 'SAME')


def batch_norm(x, n_out, is_training=True, scope='bn'):
    """
    Batch normalization on convolutional maps.

    :param x: (tf.Tensor) input tensor to apply batch_norm
    :param n_out:
    :param is_training:

    :return: (tf.Tensor) applied batch normalization tensor

    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def vgg_block(name, x, in_filters, out_filters, repeat, strides, is_training):
    """
    Helper function vgg block which all layer added

    :param x: (tf.Tensor) input tensor to apply vgg network
    :param in_filters: (int) input dimension
    :param out_filters: (int) output dimension
    :param repeat:(int) number of vgg block
    :param strides: (int) number of strides
    :param is_training: (boolean) whether training is applying or not

    :return: (tf.Tensor) output tensor which applied vgg network

    """
    with tf.variable_scope(name):
        for layer in range(repeat):
            scope_name = name + '_' + str(layer)
            x = conv2d(scope_name, x, 3, in_filters, out_filters, strides)
            if USE_BN:
                x = batch_norm(x, out_filters, is_training)
            x = lrelu(x)

            in_filters = out_filters

        x = max_pool(x, 2, 2)
        return x


def flatten(x, name=None):
    """
    Helper function flatten Tensor to 2-dimensions.

    :param x: (tf.Tensor) input tensor to flatten to 2d tensor

    :return: (tf.Tensor) output tensor

    """
    with tf.variable_scope(name):
        dims = x.get_shape().as_list()
        if len(dims) == 4:
            flattened = tf.reshape(
                x,
                shape=[-1, dims[1] * dims[2] * dims[3]])
        elif len(dims) == 2 or len(dims) == 1:
            flattened = x
        else:
            raise ValueError('Expected n dimensions of 1, 2 or 4.  Found:',
                             len(dims))

        return flattened


def to_tensor(x):
    """
    Helper function to convert 2 dim Tensor to a 4 dim Tensor ready for convolution.

    Performs the opposite of flatten(x).  If the tensor is already 4-D, this
    returns the same as the input, leaving it unchanged.

    :param x: (tf.Tensor) input tensor to reverse-flatten to 4d tensor for applying convolution

    :return: (tf.Tensor) output tensor

    """
    if len(x.get_shape()) == 2:
        n_input = x.get_shape().as_list()[1]
        x_dim = np.sqrt(n_input)
        if x_dim == int(x_dim):
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 1], name='reshape')
        elif np.sqrt(n_input / 3) == int(np.sqrt(n_input / 3)):
            x_dim = int(np.sqrt(n_input / 3))
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, 3], name='reshape')
        else:
            x_tensor = tf.reshape(
                x, [-1, 1, 1, n_input], name='reshape')
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    return x_tensor
