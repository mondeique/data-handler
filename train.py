import os
import numpy as np
import tensorflow as tf
from .tf_MTL import data_loader
from .tf_MTL import data_utils
from .tf_MTL import model
from .tf_MTL import const

# prepare data
train_data, test_data = data_loader.getBagImage()


def one_hot(index, num_classes):
    assert index < num_classes and index >= 0
    tmp = np.zeros(num_classes, dtype=np.float32)
    tmp[index] = 1.0
    return tmp


sess = tf.InteractiveSession()
global_step = tf.contrib.framework.get_or_create_global_step()

x, y, mask = model.input_tensor()

y_color_conv, y_shape_conv, y_opening_conv, y_strap_conv, y_pattern_conv, y_material_conv, is_training, keep_prob = model.multi_label_net(x)

color_loss, shape_loss, opening_loss, strap_loss, pattern_loss, material_loss, l2_loss, total_loss = model.selective_loss(y_color_conv, y_shape_conv,
                                                                                                                          y_opening_conv, y_strap_conv,
                                                                                                                          y_pattern_conv, y_material_conv,
                                                                                                                          y, mask)

train_step = model.train_op(total_loss, global_step)

color_mask = tf.get_collection('color_mask')[0]
shape_mask = tf.get_collection('shape_mask')[0]
opening_mask = tf.get_collection('opening_mask')[0]
strap_mask = tf.get_collection('strap_mask')[0]
pattern_mask = tf.get_collection('pattern_mask')[0]
material_mask = tf.get_collection('material_mask')[0]

y_color = tf.get_collection('y_color')[0]
y_shape = tf.get_collection('y_shape')[0]
y_opening = tf.get_collection('y_opening')[0]
y_strap = tf.get_collection('y_strap')[0]
y_pattern = tf.get_collection('y_pattern')[0]
y_material = tf.get_collection('y_material')[0]

color_correct_prediction = tf.equal(tf.argmax(y_color_conv, 1), tf.argmax(y_color, 1))
shape_correct_prediction = tf.equal(tf.argmax(y_shape_conv, 1), tf.argmax(y_shape, 1))
opening_correct_prediction = tf.equal(tf.argmax(y_opening_conv, 1), tf.argmax(y_opening, 1))
strap_correct_prediction = tf.equal(tf.argmax(y_strap_conv, 1), tf.argmax(y_strap, 1))
pattern_correct_prediction = tf.equal(tf.argmax(y_pattern_conv, 1), tf.argmax(y_pattern, 1))
material_correct_prediction = tf.equal(tf.argmax(y_material_conv, 1), tf.argmax(y_material, 1))

color_true_pred = tf.reduce_sum(tf.cast(color_correct_prediction, dtype=tf.float32) * color_mask)
shape_true_pred = tf.reduce_sum(tf.cast(shape_correct_prediction, dtype=tf.float32) * shape_mask)
opening_true_pred = tf.reduce_sum(tf.cast(opening_correct_prediction, dtype=tf.float32) * opening_mask)
strap_true_pred = tf.reduce_sum(tf.cast(strap_correct_prediction, dtype=tf.float32) * strap_mask)
pattern_true_pred = tf.reduce_sum(tf.cast(pattern_correct_prediction, dtype=tf.float32) * pattern_mask)
material_true_pred = tf.reduce_sum(tf.cast(material_correct_prediction, dtype=tf.float32) * material_mask)

real_train_data = []
# TODO : 이 code가 맞는지 한번 실제 데이터로 돌려봐야함
# Mask : color -> 0 , shape -> 1, opening -> 2, strap -> 3, pattern -> 4, material -> 5
for i in range(len(train_data)):
    img = (train_data[i][0] - 128) / 255.0
    label = train_data[i][1]
    real_train_data.append((img, one_hot(label, 7), 0.0))
for i in range(len(train_data)):
    img = (train_data[i][0] - 128) / 255.0
    label = train_data[i][1]
    real_train_data.append((img, one_hot(label, 7), 1.0))
for i in range(len(train_data)):
    img = (train_data[i][0] - 128) / 255.0
    label = train_data[i][1]
    real_train_data.append((img, one_hot(label, 7), 2.0))
for i in range(len(train_data)):
    img = (train_data[i][0] - 128) / 255.0
    label = train_data[i][1]
    real_train_data.append((img, one_hot(label, 7), 3.0))
for i in range(len(train_data)):
    img = (train_data[i][0] - 128) / 255.0
    label = train_data[i][1]
    real_train_data.append((img, one_hot(label, 7), 4.0))
for i in range(len(train_data)):
    img = (train_data[i][0] - 128) / 255.0
    label = train_data[i][1]
    real_train_data.append((img, one_hot(label, 7), 5.0))

saver = tf.train.Saver()

if not os.path.isfile(const.SAVE_FOLDER + 'model.ckpt.index'):
    print('Create new model')
    sess.run(tf.global_variables_initializer())
    print('OK')
else:
    print('Restoring existed model')
    saver.restore(sess, const.SAVE_FOLDER + 'model.ckpt')
    print('OK')

loss_summary_placeholder = tf.placeholder(tf.float32)
tf.summary.scalar('loss', loss_summary_placeholder)
merge_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./summary/")

learning_rate = tf.get_collection('learning_rate')[0]

current_epoch = (int)(global_step.eval() / (real_train_data // const.BATCH_SIZE))
for epoch in range(current_epoch + 1, const.NUM_EPOCHS):
    print('Epoch:', str(epoch))
    np.random.shuffle(real_train_data)
    train_img = []
    train_label = []
    train_mask = []

    for i in range(len(real_train_data)):
        train_img.append(real_train_data[i][0])
        train_label.append(real_train_data[i][1])
        train_mask.append(real_train_data[i][2])

    number_batch = len(real_train_data) // const.BATCH_SIZE

    avg_ttl = []
    avg_rgl = []
    avg_color_loss = []
    avg_shape_loss = []
    avg_opening_loss = []
    avg_strap_loss = []
    avg_pattern_loss = []
    avg_material_loss = []

    color_nb_true_pred = 0
    shape_nb_true_pred = 0
    opening_nb_true_pred = 0
    strap_nb_true_pred = 0
    pattern_nb_true_pred = 0
    material_nb_true_pred = 0

    color_nb_train = 0
    shape_nb_train = 0
    opening_nb_train = 0
    strap_nb_train = 0
    pattern_nb_train = 0
    material_nb_train = 0

    print("Learning rate: %f" % learning_rate.eval())
    for batch in range(number_batch):
        # print('Training on batch {0}/{1}'.format(str(batch + 1), str(number_batch)))
        top = batch * const.BATCH_SIZE
        bot = min((batch + 1) * const.BATCH_SIZE, len(real_train_data))
        batch_img = np.asarray(train_img[top:bot])
        batch_label = np.asarray(train_label[top:bot])
        batch_mask = np.asarray(train_mask[top:bot])

        for i in range(const.BATCH_SIZE):
            if batch_mask[i] == 0.0:
                color_nb_train += 1
            else:
                if batch_mask[i] == 1.0:
                    shape_nb_train += 1
                else:
                    if batch_mask[i] == 2.0:
                        opening_nb_train += 1
                    else:
                        if batch_mask[i] == 3.0:
                            strap_nb_train += 1
                        else:
                            if batch_mask[i] == 4.0:
                                pattern_nb_train += 1
                            else:
                                material_nb_train += 1

        batch_img = data_utils.augmentation(batch_img, 227)

        ttl, colorl, shapel, openingl, strapl, patternl, materiall, l2l, _ = sess.run([total_loss, color_loss, shape_loss, opening_loss, strap_loss,
                                                                                      pattern_loss, material_loss, l2_loss, train_step],
                                                                                      feed_dict={x: batch_img, y: batch_label, mask: batch_mask,
                                                                                                 is_training: True,
                                                                                                 keep_prob: const.DROP_OUT_PROB})

        color_nb_true_pred += sess.run(color_true_pred, feed_dict={x: batch_img, y: batch_label, mask: batch_mask,
                                                                   is_training: True,keep_prob: const.DROP_OUT_PROB})

        shape_nb_true_pred += sess.run(shape_true_pred, feed_dict={x: batch_img, y: batch_label, mask: batch_mask,
                                                                   is_training: True, keep_prob: const.DROP_OUT_PROB})

        opening_nb_true_pred += sess.run(opening_true_pred, feed_dict={x: batch_img, y: batch_label, mask: batch_mask,
                                                                       is_training: True, keep_prob: const.DROP_OUT_PROB})

        strap_nb_true_pred += sess.run(strap_true_pred, feed_dict={x: batch_img, y: batch_label, mask: batch_mask,
                                                                   is_training: True, keep_prob: const.DROP_OUT_PROB})

        pattern_nb_true_pred += sess.run(pattern_true_pred, feed_dict={x: batch_img, y: batch_label, mask: batch_mask,
                                                                       is_training: True, keep_prob: const.DROP_OUT_PROB})

        material_nb_true_pred += sess.run(material_true_pred, feed_dict={x: batch_img, y: batch_label, mask: batch_mask,
                                                                         is_training: True, keep_prob: const.DROP_OUT_PROB})

        avg_ttl.append(ttl)
        avg_color_loss.append(colorl)
        avg_shape_loss.append(shapel)
        avg_opening_loss.append(openingl)
        avg_strap_loss.append(strapl)
        avg_pattern_loss.append(patternl)
        avg_material_loss.append(materiall)

        avg_rgl.append(l2l)

    color_train_accuracy = color_nb_true_pred * 1.0 / color_nb_train
    shape_train_accuracy = shape_nb_true_pred * 1.0 / shape_nb_train
    opening_train_accuracy = opening_nb_true_pred * 1.0 / opening_nb_train
    strap_train_accuracy = strap_nb_true_pred * 1.0 / strap_nb_train
    pattern_train_accuracy = pattern_nb_true_pred * 1.0 / pattern_nb_train
    material_train_accuracy = material_nb_true_pred * 1.0 / material_nb_train


#     print('Avg_ttl: ' + str(avg_ttl))
#     print('loss_summary_placeholder: ' + str(loss_summary_placeholder))
#     print('merge_summary: ' + str(merge_summary))

    summary = sess.run(merge_summary, feed_dict={loss_summary_placeholder: avg_ttl})
    writer.add_summary(summary, global_step=epoch)

    with open('log.csv', 'w+') as f:
        # epochs, color_train_accuracy, shape_train_accuracy, opening_train_accuracy,
        # avg_color_loss, avg_shape_loss, avg_opening_loss, avg_ttl, avg_rgl
        f.write('{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}\n'.format(current_epoch, color_train_accuracy, shape_train_accuracy, opening_train_accuracy,
                                                                                            strap_train_accuracy, pattern_train_accuracy, material_train_accuracy,
                                                                                            avg_color_loss, avg_shape_loss, avg_opening_loss, avg_strap_loss, avg_pattern_loss,
                                                                                            avg_material_loss, avg_ttl, avg_rgl))

    print('color task train accuracy: ' + str(color_train_accuracy * 100))
    print('shape task train accuracy: ' + str(shape_train_accuracy * 100))
    print('opening task train accuracy: ' + str(opening_train_accuracy * 100))
    print('strap task train accuracy: ' + str(strap_train_accuracy * 100))
    print('pattern task train accuracy: ' + str(pattern_train_accuracy * 100))
    print('material task train accuracy: ' + str(material_train_accuracy * 100))

    print('Total loss: ' + str(avg_ttl) + '. L2-loss: ' + str(avg_rgl))
    print('Color loss: ' + str(avg_color_loss))
    print('Shape loss: ' + str(avg_shape_loss))
    print('opening loss: ' + str(avg_opening_loss))
    print('strap loss: ' + str(avg_strap_loss))
    print('pattern loss: ' + str(avg_pattern_loss))
    print('material loss: ' + str(avg_material_loss))

    print('\n')

    saver.save(sess, const.SAVE_FOLDER + 'model.ckpt')
