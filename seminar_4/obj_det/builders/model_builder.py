import tensorflow as tf
import os, sys
import numpy as np

from datasets import dataset_utils
from preprocessing import vgg_preprocessing
from functools import partial
from .VOC_utils import NUM_CLASSES

slim = tf.contrib.slim

CHECKPOINTS_DIR = '/home/ivan/Data/deep/course_au/au_dl_course/' \
                  'seminar_4/obj_det/checkpoints'
K = 9
FEATURE_MAP_SIZE = 12
IOU_THRESHOLD_LOW = 0.3
IOU_THRESHOLD_HIGH = 0.7
vgg_16_default_image_size = 224
PREDS_NUMBER = 64
BATCH_SIZE = 16


if not tf.gfile.Exists(CHECKPOINTS_DIR):
    tf.gfile.MakeDirs(CHECKPOINTS_DIR)


class FasterRCNN:
    def __init__(self):
        self.images = None
        self.gnd_truth_boxes = None
        self.gnd_truth_labels = None
        self.conv = None
        self.init_assign_op = None
        self.init_feed_dict = None
        self.features_map = None
        self.anchors_logits = None
        self.anchors_coors = None
        self.top_anchors_coors = None
        self.top_anchors_labels = None
        self.classification_masks = None
        self.mask = None
        self.features_map_masked = None
        self.roi_loss = None
        self.class_loss = None
        self.total_loss = None

    def build_faster_rcnn(self, images, bndboxes, labels):

        self.images = images
        self.gnd_truth_boxes = bndboxes
        self.gnd_truth_labels = labels

        with slim.arg_scope(FasterRCNN.vgg_arg_scope()):
            self.conv = FasterRCNN.vgg_16(images,
                                          num_classes=1000,
                                          is_training=True)

        self.init_assign_op, self.init_feed_dict = \
            slim.assign_from_checkpoint(os.path.join(CHECKPOINTS_DIR, 'vgg_16.ckpt'),
                                        slim.get_model_variables('vgg_16'))

        self.create_roi()
        self.create_roi_loss()
        self.create_masks()
        self.create_classification()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        self.total_loss = tf.losses.get_total_loss(add_regularization_losses=False)

        train_op = slim.learning.create_train_op(self.total_loss, optimizer)

        return train_op, self.init_assign_op, self.init_feed_dict

    def create_masks(self):
        with tf.control_dependencies([self.top_anchors_coors]):
            def generate_image_masks(image_anchors_coors):
                image_masks = tf.map_fn(FasterRCNN.generate_anchor_mask, image_anchors_coors, dtype=tf.float32)
                return image_masks
            self.classification_masks = tf.map_fn(generate_image_masks, self.top_anchors_coors, dtype=tf.float32)

    @staticmethod
    def generate_anchor_mask(anchor_coors):
        x_min = tf.cast(tf.round(tf.maximum(anchor_coors[0], 0.0) * FEATURE_MAP_SIZE), dtype=tf.int32)
        y_min = tf.cast(tf.round(tf.maximum(anchor_coors[1], 0.0) * FEATURE_MAP_SIZE), dtype=tf.int32)
        x_max = tf.cast(tf.round(tf.minimum(anchor_coors[2], 1.0) * FEATURE_MAP_SIZE), dtype=tf.int32)
        y_max = tf.cast(tf.round(tf.minimum(anchor_coors[3], 1.0) * FEATURE_MAP_SIZE), dtype=tf.int32)

        height = y_max - y_min
        width = x_max - x_min
        left_margin = x_min
        right_margin = FEATURE_MAP_SIZE - x_max
        bottom_margin = y_min
        top_margin = FEATURE_MAP_SIZE - y_max

        mask = tf.ones(shape=[width, height], dtype=tf.float32)

        mask = tf.cond(left_margin > 0,
                       lambda: tf.concat([tf.zeros(shape=[left_margin, height], dtype=tf.float32), mask], axis=1),
                       lambda: mask)

        mask = tf.cond(right_margin > 0,
                       lambda: tf.concat([mask, tf.zeros(shape=[right_margin, height], dtype=tf.float32)], axis=1),
                       lambda: mask)

        mask = tf.cond(bottom_margin > 0,
                       lambda: tf.concat([mask, tf.zeros(shape=[FEATURE_MAP_SIZE, bottom_margin], dtype=tf.float32)], axis=0),
                       lambda: mask)

        mask = tf.cond(top_margin > 0,
                       lambda: tf.concat([tf.zeros(shape=[FEATURE_MAP_SIZE, top_margin], dtype=tf.float32), mask], axis=0),
                       lambda: mask)

        with tf.control_dependencies([tf.assert_equal(tf.shape(mask), [FEATURE_MAP_SIZE, FEATURE_MAP_SIZE])]):
            mask = tf.tile(mask, tf.stack([1, 512]))
            mask = tf.reshape(mask, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, 512])

        return mask

    def create_classification(self):
        with tf.control_dependencies([self.classification_masks, self.features_map]):
            self.classification_masks = tf.reshape(self.classification_masks,
                                                   shape=[PREDS_NUMBER, BATCH_SIZE, FEATURE_MAP_SIZE,
                                                          FEATURE_MAP_SIZE, 512])

            layer = tf.multiply(self.classification_masks, self.features_map, name='layer')
            layer = tf.reshape(layer, shape=[BATCH_SIZE * PREDS_NUMBER, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, 512])

            top_anchors_labels = tf.reshape(self.top_anchors_labels, shape=[BATCH_SIZE * PREDS_NUMBER, NUM_CLASSES + 1])

            logits = slim.conv2d(layer, NUM_CLASSES + 1, [FEATURE_MAP_SIZE, FEATURE_MAP_SIZE],
                                 padding='VALID', scope='classification')

            logits = tf.squeeze(logits, name='logits')

            class_loss = tf.nn.softmax_cross_entropy_with_logits(labels=top_anchors_labels, logits=logits)
            self.class_loss = tf.reduce_mean(class_loss)

            tf.losses.add_loss(self.class_loss)

    def apply_masks(self):
        def apply_masks_per_image(tup):
            masks, features_map = tup

        self.features_map_masked = tf.map_fn(apply_masks_per_image, self.classification_masks, self.features_map)

    @staticmethod
    def vgg_arg_scope(weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    @staticmethod
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

    def create_roi(self):
        with slim.arg_scope(FasterRCNN.vgg_arg_scope()):
            self.features_map = slim.conv2d(self.conv, 512, [3, 3], padding='VALID', scope='roi')

            cls_preds = slim.conv2d(self.features_map, 2 * K, [1, 1], scope='cls_preds')
            cls_preds = tf.reshape(cls_preds, [-1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 2])

            bndbox_coors = slim.conv2d(self.features_map, 4 * K, [1, 1], scope='coor_regs')
            bndbox_coors = tf.nn.sigmoid(bndbox_coors)
            bndbox_coors = tf.reshape(bndbox_coors, [-1, FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 4])

            self.anchors_logits = cls_preds
            self.anchors_coors = bndbox_coors

    @staticmethod
    def get_iou(a, b):
        x_left = tf.maximum(a[0], b[0])
        y_left = tf.maximum(a[1], b[1])
        x_right = tf.minimum(a[2], b[2])
        y_right = tf.minimum(a[3], b[3])

        intersection = (x_right - x_left) * (y_right - y_left)

        a_area = (a[2] - a[0]) * (a[3] - a[1])
        b_area = (b[2] - b[0]) * (b[3] - b[1])

        return intersection / (a_area + b_area - intersection)

    @staticmethod
    def process_anchor(anchor, bndboxes, labels):
        ious = tf.map_fn(partial(FasterRCNN.get_iou, b=anchor), bndboxes, back_prop=True, dtype=tf.float32)
        argmax = tf.argmax(ious, output_type=tf.int32)

        def IOU_greater_0_3(ious_array, argmax_index):
            return tf.cond(ious[argmax] > IOU_THRESHOLD_HIGH,
                           # anchor covers any of bndboxes
                           lambda: (tf.stack([1.0, 0.0]), tf.stack([1]), ious[argmax],
                                    bndboxes[argmax], labels[argmax]),
                           # anchor doesn't cover any of bndboxes well
                           lambda: (tf.stack([0.0, 1.0]), tf.stack([0]), ious[argmax],
                                    tf.stack([0.0, 0.0, 0.0, 0.0]), tf.stack([0])))

        return tf.cond(ious[argmax] < IOU_THRESHOLD_LOW,
                       # anchor doesn't cover any of bndboxes
                       lambda: (tf.stack([0.0, 1.0]), tf.stack([1]), ious[argmax],
                                tf.stack([0.0, 0.0, 0.0, 0.0]), tf.stack([0])),
                       # consider iou > 0.3
                       partial(IOU_greater_0_3, ious_array=ious, argmax_index=argmax))

    @staticmethod
    def get_smooth_l1(ground_truths, preds, mask):
        diff = tf.abs(ground_truths - preds)
        condition = tf.less(diff, 1.0)
        loss = tf.where(condition, diff, tf.square(diff))
        loss = tf.multiply(loss, tf.cast(mask, dtype=tf.float32))

        return tf.reduce_mean(loss, name='smooth_l1')

    @staticmethod
    def get_roi_loss_per_image(single_image):

        feature_map, anchors_logits, anchors_coors, gnd_truth_bndboxes, gnd_truth_labels = single_image

        anchors_coors = tf.reshape(anchors_coors, [-1, 4])

        gnd_truth_preds, anchor_mask, max_ious, ground_truth_boxes, gnd_truth_labels = \
            tf.map_fn(partial(FasterRCNN.process_anchor, bndboxes=gnd_truth_bndboxes, labels=gnd_truth_labels),
                      anchors_coors, back_prop=True, dtype=(tf.float32, tf.int32, tf.float32, tf.float32, tf.int32))

        _, top_indices = tf.nn.top_k(max_ious, k=PREDS_NUMBER, name='top_anchors')
        top_bndbox_coors = tf.gather(anchors_coors, indices=top_indices, name='top_bndbox_coors')
        top_gnd_truth_labels = tf.gather(gnd_truth_labels, indices=top_indices, name='top_pred_labels')
        top_gnd_truth_labels = tf.one_hot(top_gnd_truth_labels, depth=NUM_CLASSES + 1, on_value=1.0, off_value=0.0)

        gnd_truth_preds = tf.reshape(gnd_truth_preds, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 2])
        anchor_mask = tf.reshape(anchor_mask, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 1])
        ground_truth_boxes = tf.reshape(ground_truth_boxes, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 4])
        anchors_coors = tf.reshape(anchors_coors, shape=[FEATURE_MAP_SIZE, FEATURE_MAP_SIZE, K, 4])

        gnd_truth_preds = tf.reshape(gnd_truth_preds, shape=[-1, 2])
        anchors_logits = tf.reshape(anchors_logits, shape=[-1, 2])

        loss_preds = tf.nn.softmax_cross_entropy_with_logits(labels=gnd_truth_preds, logits=anchors_logits)

        loss_preds = tf.multiply(loss_preds, tf.cast(anchor_mask, tf.float32))
        loss_preds = tf.reduce_mean(loss_preds)

        loss_smooth = FasterRCNN.get_smooth_l1(ground_truth_boxes, anchors_coors, anchor_mask)

        roi_loss = loss_preds + loss_smooth

        return roi_loss, top_bndbox_coors, top_gnd_truth_labels

    def create_roi_loss(self):

        with slim.arg_scope(FasterRCNN.vgg_arg_scope()):
            losses, self.top_anchors_coors, self.top_anchors_labels \
                = tf.map_fn(FasterRCNN.get_roi_loss_per_image, (self.features_map, self.anchors_logits,
                                                                self.anchors_coors, self.gnd_truth_boxes,
                                                                self.gnd_truth_labels),
                            dtype=(tf.float32, tf.float32, tf.float32), back_prop=True, name='losses_mapping')

            self.roi_loss = tf.reduce_mean(losses)
            tf.losses.add_loss(self.roi_loss)
