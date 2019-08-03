
import tensorflow as tf
from .const import *
from .ops import fc_layer, vgg_block


def input_tensor():
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y = tf.placeholder(tf.float32, [None, 7])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE])

    return x, y, mask


def multi_label_net(x):

    is_training = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    x = vgg_block('Block1', x, 1, 32, 2, 1, is_training)
    # print(x.get_shape())

    x = vgg_block('Block2', x, 32, 64, 2, 1, is_training)
    # print(x.get_shape())

    x = vgg_block('Block3', x, 64, 128, 2, 1, is_training)
    # print(x.get_shape())

    x = vgg_block('Block4', x, 128, 256, 3, 1, is_training)
    # print(x.get_shape())

    # TODO : fully-connected layer dimension 개수 바꿔야 함
    # color branch
    color_fc1 = fc_layer('color_fc1', x, 256, keep_prob)
    color_fc2 = fc_layer('color_fc2', color_fc1, 256, keep_prob)
    y_color_conv = fc_layer('color_softmax', color_fc2, 15, keep_prob, 'softmax')

    # shape branch
    shape_fc1 = fc_layer('shape_fc1', x, 256, keep_prob)
    shape_fc2 = fc_layer('shape_fc2', shape_fc1, 256, keep_prob)
    y_shape_conv = fc_layer('shape_softmax', shape_fc2, 13, keep_prob, 'softmax')

    # opening_type branch
    opening_fc1 = fc_layer('opening_fc1', x, 256, keep_prob)
    opening_fc2 = fc_layer('opening_fc2', opening_fc1, 256, keep_prob)
    y_opening_conv = fc_layer('opening_softmax', opening_fc2, 6, keep_prob, 'softmax')

    # strap branch
    strap_fc1 = fc_layer('strap_fc1', x, 256, keep_prob)
    strap_fc2 = fc_layer('strap_fc2', strap_fc1, 256, keep_prob)
    y_strap_conv = fc_layer('strap_softmax', strap_fc2, 6, keep_prob, 'softmax')

    # pattern branch
    pattern_fc1 = fc_layer('pattern_fc1', x, 256, keep_prob)
    pattern_fc2 = fc_layer('pattern_fc2', pattern_fc1, 256, keep_prob)
    y_pattern_conv = fc_layer('pattern_softmax', pattern_fc2, 12, keep_prob, 'softmax')

    # material branch
    material_fc1 = fc_layer('material_fc1', x, 256, keep_prob)
    material_fc2 = fc_layer('material_fc2', material_fc1, 256, keep_prob)
    y_material_conv = fc_layer('material_softmax', material_fc2, 7, keep_prob, 'softmax')

    return y_color_conv, y_shape_conv, y_opening_conv, y_strap_conv, y_pattern_conv, y_material_conv, is_training, keep_prob


def selective_loss(y_color_conv, y_shape_conv, y_opening_conv, y_strap_conv, y_pattern_conv, y_material_conv, y, mask):

    vector_color = tf.constant(0., tf.float32, [BATCH_SIZE])
    vector_shape = tf.constant(1., tf.float32, [BATCH_SIZE])
    vector_opening = tf.constant(2., tf.float32, [BATCH_SIZE])
    vector_strap = tf.constant(3., tf.float32, [BATCH_SIZE])
    vector_pattern = tf.constant(4., tf.float32, [BATCH_SIZE])
    vector_material = tf.constant(5., tf.float32, [BATCH_SIZE])

    color_mask = tf.cast(tf.equal(mask, vector_color), tf.float32)
    shape_mask = tf.cast(tf.equal(mask, vector_shape), tf.float32)
    opening_mask = tf.cast(tf.equal(mask, vector_opening), tf.float32)
    strap_mask = tf.cast(tf.equal(mask, vector_strap), tf.float32)
    pattern_mask = tf.cast(tf.equal(mask, vector_pattern), tf.float32)
    material_mask = tf.cast(tf.equal(mask, vector_material), tf.float32)

    tf.add_to_collection('smile_mask', color_mask)
    tf.add_to_collection('shape_mask', shape_mask)
    tf.add_to_collection('opening_mask', opening_mask)
    tf.add_to_collection('strap_mask', strap_mask)
    tf.add_to_collection('pattern_mask', pattern_mask)
    tf.add_to_collection('material_mask', material_mask)

    y_color = tf.slice(y, [0, 0], [BATCH_SIZE, 2])
    y_shape = tf.slice(y, [0, 0], [BATCH_SIZE, 2])
    y_opening = tf.slice(y, [0, 0], [BATCH_SIZE, 4])
    y_strap = tf.slice(y, [0, 0], [BATCH_SIZE, 4])
    y_pattern = tf.slice(y, [0, 0], [BATCH_SIZE, 4])
    y_material = tf.slice(y, [0, 0], [BATCH_SIZE, 4])

    tf.add_to_collection('y_color', y_color)
    tf.add_to_collection('y_shape', y_shape)
    tf.add_to_collection('y_opening', y_opening)
    tf.add_to_collection('y_strap', y_strap)
    tf.add_to_collection('y_pattern', y_pattern)
    tf.add_to_collection('y_material', y_material)

    color_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_color * tf.log(y_color_conv), axis=1) * color_mask) / tf.clip_by_value(
        tf.reduce_sum(color_mask), 1, 1e9)
    shape_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_shape * tf.log(y_shape_conv), axis=1) * shape_mask) / tf.clip_by_value(
        tf.reduce_sum(shape_mask), 1, 1e9)
    opening_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_opening * tf.log(y_opening_conv), axis=1) * opening_mask) / tf.clip_by_value(
        tf.reduce_sum(opening_mask), 1, 1e9)
    strap_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_strap * tf.log(y_strap_conv), axis=1) * strap_mask) / tf.clip_by_value(
        tf.reduce_sum(strap_mask), 1, 1e9)
    pattern_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_pattern * tf.log(y_pattern_conv), axis=1) * pattern_mask) / tf.clip_by_value(
        tf.reduce_sum(pattern_mask), 1, 1e9)
    material_cross_entropy = tf.reduce_sum(
        tf.reduce_sum(-y_material * tf.log(y_material_conv), axis=1) * material_mask) / tf.clip_by_value(
        tf.reduce_sum(material_mask), 1, 1e9)

    l2_loss = []
    for var in tf.trainable_variables():
        if var.op.name.find(r'DW') > 0:
            l2_loss.append(tf.nn.l2_loss(var))
    l2_loss = WEIGHT_DECAY * tf.add_n(l2_loss)

    total_loss = color_cross_entropy + shape_cross_entropy + opening_cross_entropy + strap_cross_entropy + pattern_cross_entropy + material_cross_entropy + l2_loss

    return color_cross_entropy, shape_cross_entropy, opening_cross_entropy, strap_cross_entropy, pattern_cross_entropy, material_cross_entropy, l2_loss, total_loss


def train_op(loss, global_step):
    learning_rate = tf.train.exponential_decay(INIT_LR, global_step, DECAY_STEP, DECAY_LR_RATE, staircase=True)
    train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss,
                                                                                                                   global_step=global_step)
    tf.add_to_collection('learning_rate', learning_rate)
    return train_step
