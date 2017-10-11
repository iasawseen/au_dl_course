import tensorflow as tf
import os, sys
import numpy as np

from datasets import dataset_utils
from preprocessing import vgg_preprocessing
from functools import partial

slim = tf.contrib.slim

CHECKPOINTS_DIR = '/home/ivan/Data/deep/course_au/au_dl_course/' \
                  'seminar_4/obj_det/checkpoints'
K = 9


if not tf.gfile.Exists(CHECKPOINTS_DIR):
    tf.gfile.MakeDirs(CHECKPOINTS_DIR)


def build_faster_rcnn(images, qtys, bndboxes, labels):

    with slim.arg_scope(vgg_arg_scope()):
        conv = vgg_16(images,
                      num_classes=1000,
                      is_training=True)

    # print(slim.get_model_variables())
    # print(slim.get_model_variables('vgg_16'))

    init_assign_op, init_feed_dict = \
        slim.assign_from_checkpoint(os.path.join(CHECKPOINTS_DIR, 'vgg_16.ckpt'),
                                       slim.get_model_variables('vgg_16'))
        # slim.get_model_variables('vgg_16'))

    # print('init_feed_dict:')
    # print(init_feed_dict)


    cls_preds, bndbox_coors = create_roi(conv)

    # print(cls_preds)
    # print(bndbox_coors)

    loss_roi = create_roi_loss(cls_preds, bndbox_coors, qtys, bndboxes, labels)
    optimizer = tf.train.AdamOptimizer(learning_rate=.001)
    train_op = optimizer.minimize(loss_roi)

    return train_op, init_assign_op, init_feed_dict


def vgg_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID'):

    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            conv_out = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(conv_out, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers.
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                               scope='dropout7')
            net = slim.conv2d(net, num_classes, [1, 1],
                              activation_fn=None,
                              normalizer_fn=None,
                              scope='fc8')
            # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
            end_points[sc.name + '/fc8'] = net

        return conv_out


vgg_16.default_image_size = 224

FEATURE_MAP_SIZE = 12
IOU_THRESHOLD = 0.3


def get_offsets():
    offsets = np.zeros(shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 4], dtype=np.float32)

    offset = 224 / FEATURE_MAP_SIZE

    for i in range(FEATURE_MAP_SIZE):
        for j in range(FEATURE_MAP_SIZE):
            offsets[i, j, :, 0] += j * offset
            offsets[i, j, :, 1] += (FEATURE_MAP_SIZE - i - 1) * offset
            offsets[i, j, :, 2] += j * offset
            offsets[i, j, :, 3] += (FEATURE_MAP_SIZE - i - 1) * offset

    return tf.constant(offsets)


def create_roi(inputs):
    with slim.arg_scope(vgg_arg_scope()):
        fc = slim.conv2d(inputs, 512, [3, 3], padding='VALID', scope='roi')

        cls_preds = slim.conv2d(fc, 2 * K, [1, 1], scope='cls_preds')
        cls_preds = tf.reshape(cls_preds, [-1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 2])
        cls_preds = tf.nn.softmax(cls_preds)

        bndbox_coors = slim.conv2d(fc, 4 * K, [1, 1], scope='coor_regs')
        bndbox_coors = tf.reshape(bndbox_coors, [-1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 4])

        return cls_preds, bndbox_coors


def get_iou(a, b):
    x_left = tf.maximum(a[0], b[0])
    y_left = tf.maximum(a[1], b[1])
    x_right = tf.minimum(a[2], b[2])
    y_right = tf.minimum(a[3], b[3])

    intersection = (x_right - x_left) * (y_right - y_left)

    a_area = (a[2] - a[0]) * (a[3] - a[1])
    b_area = (b[2] - b[0]) * (b[3] - b[1])

    return intersection / (a_area + b_area - intersection)


def process_anchor(anchor, bndboxes):
    ious = tf.map_fn(partial(get_iou, b=anchor), bndboxes, back_prop=True, dtype=tf.float32)
    argmax = tf.argmax(ious, output_type=tf.int32)

    return tf.cond(ious[argmax] > IOU_THRESHOLD,
                   # anchor covers any of bndboxes
                   lambda: (tf.stack([1.0, 0.0]), tf.stack([1, 1, 1, 1]), bndboxes[argmax]),
                   # anchor doesn't cover any of bndboxes
                   lambda: (tf.stack([0.0, 1.0]), tf.stack([0, 0, 0, 0]), tf.stack([0.0, 0.0, 0.0, 0.0])))


def get_smooth_l1(ground_truths, preds, mask):
    diff = tf.abs(ground_truths - preds)
    # diff = tf.reshape(diff, shape=[-1])

    condition = tf.less(diff, 1.0)

    loss = tf.where(condition, diff, tf.square(diff))
    # loss = tf.reshape(loss, shape=tf.shape(ground_truths))
    loss = tf.multiply(loss, tf.cast(mask, dtype=tf.float32))
    return tf.reduce_mean(loss)


# def get_roi_loss_per_image(batch_cls_preds, batch_bndbox_coors, batch_qtys, bndboxes, batch_labels):
def get_roi_loss_per_image(single_image):

    batch_cls_preds, batch_bndbox_coors, batch_qtys, bndboxes, batch_labels = single_image

    # batch_anchors = batch_bndbox_coors + get_offsets()
    batch_anchors = tf.reshape(batch_bndbox_coors, [-1, 4])

    ground_truth_preds, anchor_mask, ground_truth_boxes = \
        tf.map_fn(partial(process_anchor, bndboxes=bndboxes), batch_anchors,
                  back_prop=True, dtype=(tf.float32, tf.int32, tf.float32))

    ground_truth_preds = tf.reshape(ground_truth_preds, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 2])
    anchor_mask = tf.reshape(anchor_mask, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 4])
    ground_truth_boxes = tf.reshape(ground_truth_boxes, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 4])

    # print('losses:')
    # print(batch_cls_preds)
    # print(ground_truth_preds)
    # print(anchor_mask)
    # print(ground_truth_boxes)

    loss_preds = tf.losses.softmax_cross_entropy(batch_cls_preds, ground_truth_preds)
    loss_smooth = get_smooth_l1(ground_truth_boxes, batch_bndbox_coors, anchor_mask)

    print(loss_preds)
    print(loss_smooth)

    return loss_preds + loss_smooth


def create_roi_loss(cls_preds, coor_regs, qtys, bndboxes, labels):

    with slim.arg_scope(vgg_arg_scope()):

        offsets = get_offsets()
        coor_regs += offsets

        losses = tf.map_fn(get_roi_loss_per_image, (cls_preds, coor_regs, qtys, bndboxes, labels),
                           dtype=tf.float32, back_prop=True)

        # print(losses)

        return tf.reduce_mean(losses)
