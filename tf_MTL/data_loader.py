
import numpy as np
import pandas as pd
import cv2

# pandas read csv -> cv2로 각각 imread -> array data 저장


def getBagImage():
    print('Reading csv................')
    df = pd.read_csv('./data/bag.csv')
    data_mask = np.random.rand(len(df)) < 0.8
    train_df = df[data_mask]
    train_label_df = train_df.drop(['filename'], axis=1)
    test_df = df[~data_mask]
    test_label_df = test_df.drop(['filename'], axis=1)
    train_data = []
    test_data = []
    print('Load bag image...................')
    for i in range(train_df.shape[0]):
        train_image = cv2.imread(train_df['filename'][i])
        train_label = train_label_df.values.tolist()
        train_data.append(train_image[i])
        train_data.append(train_label[i])
    for i in range(test_df.shape[0]):
        test_image = cv2.imread(test_df['filename'][i])
        test_label = test_label_df.values.tolist()
        test_data.append(test_image[i])
        test_data.append(test_label[i])

    print('Number of bag train data: ', str(len(train_data)))
    print('Number of bag test data: ', str(len(test_data)))
    print('---------------------------------------------------------------')
    return train_data, test_data

#########################################################################################
#
#
# def create_input_pipeline(image_paths, labels, batch_size, n_epochs, shape,
#                           crop_factor=1.0, n_threads=4, training=True, randomize=False):
#     """Creates a pipefile from a list of image files.
#     Includes batch generator/central crop/resizing options.
#     The resulting generator will dequeue the images batch_size at a time until
#     it throws tf.errors.OutOfRangeError when there are no more images left in
#     the queue.
#     Parameters
#     ----------
#     files : list
#         List of paths to image files.
#     batch_size : int
#         Number of image files to load at a time.
#     n_epochs : int
#         Number of epochs to run before raising tf.errors.OutOfRangeError
#     shape : list
#         [height, width, channels]
#     crop_shape : list
#         [height, width] to crop image to.
#     n_threads : int, optional
#         Number of threads to use for batch shuffling
#     """
#
#     # We first create a "producer" queue.  It creates a production line which
#     # will queue up the file names and allow another queue to deque the file
#     # names all using a tf queue runner.
#     # Put simply, this is the entry point of the computational graph.
#     # It will generate the list of file names.
#     # We also specify it's capacity beforehand.
#
#     # image_paths_tf = tf.convert_to_tensor(image_paths, dtype=string, name='image_paths')
#     # labels_tf = tf.convert_to_tensor(labels, dtype=string, name='labels')
#
#     if training:
#         # Remove num_epochs to continue indefinitely
#         input_queue = tf.train.slice_input_producer(
#             [image_paths, labels], shuffle=training)
#     else:
#         input_queue = tf.train.slice_input_producer(
#         [image_paths, labels], num_epochs=n_epochs, shuffle=training)
#
#     # We pass the filenames to this object which can read the file's contents.
#     # This will create another queue running which dequeues the previous queue.
#     file_contents = tf.read_file(input_queue[0])
#
#     # And then have to decode its contents as we know it is a jpeg image
#     imgs = tf.image.decode_jpeg(
#         file_contents,
#         channels=3 if len(shape) > 2 and shape[2] == 3 else 0)
#
#     # We have to explicitly define the shape of the tensor.
#     # This is because the decode_jpeg operation is still a node in the graph
#     # and doesn't yet know the shape of the image.  Future operations however
#     # need explicit knowledge of the image's shape in order to be created.
#     imgs.set_shape(shape)
#
#     # Next we'll centrally crop the image to the size of 100x100.
#     # This operation required explicit knowledge of the image's shape.
#
#     rsz_shape = [int(shape[0] * crop_factor),
#                      int(shape[1] * crop_factor)]
#     imgs = tf.image.resize_images(imgs, rsz_shape)
#
#     # TODO: Scale image by 1 +/- .150
#     # tf.image.central_crop(imgs, central_fraction)
#     # tf.image.resize_image_with_crop_or_pad(imgs, target_height, target_width)
#     #uint8image = tf.random_crop(uint8image, (224, 224, 3))
#
#     if randomize:
#         imgs = tf.image.random_flip_left_right(imgs)
#         imgs = tf.image.random_flip_up_down(imgs, seed=None)
#
#         # TODO: Random Rotation
#         # random_rot = random.randint(1,5)
#         # imgs = tf.image.rot90(imgs)
#
#         if (random.randint(1,3) == 1):
#             imgs = tf.image.transpose_image(imgs)
#
#
#     # Now we'll create a batch generator that will also shuffle our examples.
#     # We tell it how many it should have in its buffer when it randomly
#     # permutes the order.
#     min_after_dequeue = len(image_paths) // 10
#
#     # The capacity should be larger than min_after_dequeue, and determines how
#     # many examples are prefetched.  TF docs recommend setting this value to:
#     # min_after_dequeue + (num_threads + a small safety margin) * batch_size
#     capacity = min_after_dequeue + (n_threads + 1) * batch_size
#
#     if training:
#         # Randomize the order and output batches of batch_size.
#         batch, batchlabels, batchfilenames = tf.train.shuffle_batch([imgs, input_queue[1], input_queue[0]],
#                                     enqueue_many=False,
#                                     batch_size=batch_size,
#                                     capacity=capacity,
#                                     min_after_dequeue=min_after_dequeue,
#                                     num_threads=n_threads)
#     else:
#         batch, batchlabels, batchfilenames = tf.train.batch([imgs, input_queue[1], input_queue[0]],
#                                     enqueue_many=False,
#                                     batch_size=batch_size,
#                                     capacity=capacity,
#                                     num_threads=n_threads,
#                                     allow_smaller_final_batch=True)
#
#     # alternatively, we could use shuffle_batch_join to use multiple reader
#     # instances, or set shuffle_batch's n_threads to higher than 1.
#
#     return batch, batchlabels, batchfilenames
#
# def get_data_files_paths():
#     """
#     Returns the input file folders path
#
#     :return: list of strings
#         The input file paths as list [train_jpg_dir, test_jpg_dir, train_csv_file, test_csv_template_file]
#     """
#
#     data_root_folder = os.path.abspath("input/")
#     train_jpg_dir = os.path.join(data_root_folder, 'train-jpg')
#     test_jpg_dir = os.path.join(data_root_folder, 'test-jpg')
#     train_csv_file = os.path.join(data_root_folder, 'train_tags.csv')
#     test_csv_template_file = os.path.join(data_root_folder, 'test_tags_blank.csv')
#     return [train_jpg_dir, test_jpg_dir, train_csv_file, test_csv_template_file]
#
#
# def dense_to_one_hot(labels, n_classes=2):
#     """Convert class labels from scalars to one-hot vectors.
#     Parameters
#     ----------
#     labels : array
#         Input labels to convert to one-hot representation.
#     n_classes : int, optional
#         Number of possible one-hot.
#     Returns
#     -------
#     one_hot : array
#         One hot representation of input.
#     """
#     return np.eye(n_classes).astype(np.float32)[labels]
#
# def multilabeldense_to_one_hot(tags, labels_map):
#     """Convert class labels from scalars to one-hot vectors.
#     Parameters
#     ----------
#     tags : array
#         Input labels to convert to one-hot representation.
#     labels_map :
#         Number of possible one-hot.
#     Returns
#     -------
#     one_hot : array
#         One hot representation of input.
#     """
#     targets = np.zeros(len(labels_map), np.int)
#     for t in tags.split(' '):
#         targets[labels_map[t]] = 1
#     return targets.tolist()
