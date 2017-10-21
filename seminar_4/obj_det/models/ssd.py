import numpy as np
import tensorflow as tf
import os


from functools import partial
from builders.VOC_utils import NUM_CLASSES
from nets import vgg

slim = tf.contrib.slim

CHECKPOINTS_DIR = '/checkpoints'
K = 9
FEATURE_MAP_SIZE = 12
IOU_THRESHOLD_LOW = 0.3
IOU_THRESHOLD_HIGH = 0.7
IOU_THRESHOLD = 0.5

VVG_DEFAULT_IMAGE_SIZE = 224
PREDS_NUMBER = 64
BATCH_SIZE = 16
BOXES = 3117


class SSD:
    def __init__(self):
        self.images = None
        self.gnd_truth_boxes = None
        self.gnd_truth_labels = None

        self.conv_4_3 = None
        self.conv_5_3 = None
        self.conv_7 = None
        self.conv_8_2 = None
        self.conv_9_2 = None
        self.conv_10_2 = None
        self.conv_4_3_detector = None
        self.conv_7_detector = None
        self.conv_8_detector = None
        self.conv_9_detector = None
        self.conv_10_detector = None

        self.default_boxes = None
        self.pred_boxes_offsets = None
        self.pred_boxes_coors = None
        self.pred_boxes_norm = None

        self.pred_class_logits = None
        self.pred_classes = None

        # self.pred_classes_suppressed = None

        self.result_images = None

        self.init_assign_op = None
        self.init_feed_dict = None

        self.total_loss = None

    def build(self, images, bndboxes, labels):
        self.images = images
        self.gnd_truth_boxes = bndboxes
        self.gnd_truth_labels = tf.one_hot(labels, depth=NUM_CLASSES + 1, on_value=1.0, off_value=0.0)

        with slim.arg_scope(SSD.vgg_arg_scope()):
            self.vgg_16(images, num_classes=1000, is_training=True)

        self.init_assign_op, self.init_feed_dict = \
            slim.assign_from_checkpoint(os.path.join(os.getcwd()+CHECKPOINTS_DIR, 'vgg_16.ckpt'),
                                        slim.get_model_variables('vgg_16'))
        self.build_ssd_layers()

        self.conv_4_3 = tf.nn.tanh(self.conv_4_3)

        self.conv_4_3_detector = SSD.build_detector_classifier(self.conv_4_3, boxes=3, scope='conv_4_3_detector',
                                                               map_size=(28, 28), box_def_size=25)

        self.conv_7_detector = SSD.build_detector_classifier(self.conv_7, boxes=3, scope='conv_7_detector',
                                                             map_size=(14, 14), box_def_size=50)

        self.conv_8_detector = SSD.build_detector_classifier(self.conv_8_2, boxes=3, scope='conv_8_detector',
                                                             map_size=(7, 7), box_def_size=80)

        self.conv_9_detector = SSD.build_detector_classifier(self.conv_9_2, boxes=3, scope='conv_9_detector',
                                                             map_size=(3, 3), box_def_size=120)

        self.conv_10_detector = SSD.build_detector_classifier(self.conv_10_2, boxes=3, scope='conv_10_detector',
                                                              map_size=(1, 1), box_def_size=150)

        self.default_boxes = self.concat_by_index(0, axis=0)
        self.pred_boxes_offsets = self.concat_by_index(1, axis=1)
        self.pred_boxes_coors = self.concat_by_index(2, axis=1)
        self.pred_boxes_norm = SSD.get_box_norm(self.pred_boxes_coors)
        self.pred_class_logits = self.concat_by_index(3, axis=1)
        self.pred_classes = tf.nn.softmax(self.pred_class_logits)

        self.suppress_boxes()

        self.create_loss()

        tf.losses.add_loss(self.total_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        total_loss = tf.losses.get_total_loss(add_regularization_losses=False)

        train_op = slim.learning.create_train_op(total_loss, optimizer)

        return train_op, self.init_assign_op, self.init_feed_dict

    def get_images_with_boxes(self):
        return self.result_images

    def suppress_boxes(self):
        masks, best_classes = tf.map_fn(SSD.suppress_image, (self.pred_boxes_norm, self.pred_classes),
                                        dtype=(tf.bool, tf.int32), name='suppressing')

        self.result_images = tf.map_fn(SSD.draw_bndboxes, (self.images, self.pred_boxes_norm, best_classes, masks),
                                       dtype=tf.float32)



    @staticmethod
    def draw_bndboxes(tup):
        image, pred_boxes_norm, best_classes, mask = tup
        pred_boxes_norm = tf.boolean_mask(pred_boxes_norm, mask)
        best_classes = tf.boolean_mask(best_classes, mask)
        pred_boxes_norm = tf.reshape(pred_boxes_norm, shape=[1, tf.shape(pred_boxes_norm)[0], 4])
        image = tf.reshape(image, shape=[1, VVG_DEFAULT_IMAGE_SIZE, VVG_DEFAULT_IMAGE_SIZE, 3])
        image = tf.image.draw_bounding_boxes(images=image, boxes=pred_boxes_norm)
        image = tf.reshape(image, shape=[VVG_DEFAULT_IMAGE_SIZE, VVG_DEFAULT_IMAGE_SIZE, 3])

        return image

    @staticmethod
    def suppress_image(tup):
        image_boxes, image_classes = tup

        def process_class(class_scores, boxes):
            indices = tf.image.non_max_suppression(boxes, scores=class_scores, max_output_size=16)
            size = tf.shape(indices)[0]
            indices = tf.reshape(indices, shape=[size, 1])
            updates = tf.ones(shape=[size], dtype=tf.float32)
            shape = tf.constant([BOXES])
            mask = tf.scatter_nd(indices, updates, shape)

            return tf.multiply(class_scores, mask)

        transposed = tf.transpose(image_classes)

        supp_classes = tf.map_fn(partial(process_class, boxes=image_boxes), transposed, dtype=tf.float32)
        supp_classes = tf.transpose(supp_classes)

        def get_mask(sup_scores):
            argmax = tf.cast(tf.argmax(sup_scores), dtype=tf.int32)

            return tf.cond(sup_scores[argmax] > 0.5, lambda: (True, argmax), lambda: (False, 0))

        mask, best_class = tf.map_fn(get_mask, supp_classes, dtype=(tf.bool, tf.int32))
        return mask, best_class

    @staticmethod
    def get_box_norm(batch_boxes):
        batch_boxes = tf.divide(batch_boxes, VVG_DEFAULT_IMAGE_SIZE)

        def process_image(boxes):
            return tf.map_fn(lambda box: tf.stack(SSD.restrict_coors(
                SSD.convert_box_to_coors(box), minimum=0.0, maximum=1.0)), boxes)

        norm_boxes = tf.map_fn(process_image, batch_boxes)
        return norm_boxes

    @staticmethod
    def build_detector_classifier(input_layer, boxes, scope, map_size, box_def_size):
        with tf.variable_scope(scope) as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                coors_offsets = slim.conv2d(input_layer, boxes * 4, [3, 3],
                                            padding='SAME', scope=scope + '_coors')
                coors_offsets = tf.reshape(coors_offsets, shape=[BATCH_SIZE, -1, 4])

                default_boxes = SSD.generate_3_boxes(map_size, box_def_size, name=scope + '_default_boxes')

                coors = tf.add(coors_offsets, default_boxes)

                logits = slim.conv2d(input_layer, boxes * (NUM_CLASSES + 1), [3, 3],
                                     padding='SAME', scope=scope + '_labels')

                logits = tf.reshape(logits, shape=[BATCH_SIZE, -1, 21])

        return default_boxes, coors_offsets, coors, logits

    @staticmethod
    def generate_3_boxes(map_size, def_size, name):
        map_width, map_height = map_size

        sizes = [(def_size, def_size),
                 (def_size * np.sqrt(2), def_size / np.sqrt(2)),
                 (def_size / np.sqrt(2), def_size * np.sqrt(2))]

        result = np.zeros(shape=(map_height, map_width, 3, 4))

        width_offset = VVG_DEFAULT_IMAGE_SIZE / (map_width * 2)
        height_offset = VVG_DEFAULT_IMAGE_SIZE / (map_height * 2)

        for index, tup in zip(range(len(sizes)), sizes):
            width, height = tup

            result[:, 0, index, 0] = width_offset
            result[0, :, index, 1] = VVG_DEFAULT_IMAGE_SIZE - height_offset
            result[:, :, index, 2] = width
            result[:, :, index, 3] = height

            for j in range(1, map_height):
                result[:, j, index, 0] = result[:, j - 1, index, 0] + width_offset * 2

            for i in range(1, map_width):
                result[i, :, index, 1] = result[i - 1, :, index, 1] - width_offset * 2

        def process_boxes(boxes):
            for i in range(map_height):
                for j in range(map_width):
                    for k in range(len(sizes)):
                        box = boxes[i, j, k, :]
                        x_min = box[0] - box[2] / 2
                        y_min = box[1] - box[3] / 2
                        x_max = box[0] + box[2] / 2
                        y_max = box[1] + box[3] / 2

                        x_min = max(x_min, 0.0)
                        y_min = max(y_min, 0.0)
                        x_max = min(x_max, VVG_DEFAULT_IMAGE_SIZE)
                        y_max = min(y_max, VVG_DEFAULT_IMAGE_SIZE)

                        upd_w = x_max - x_min
                        upd_h = y_max - y_min
                        upd_x = x_min + upd_w / 2
                        upd_y = y_min + upd_h / 2

                        box[0], box[1], box[2], box[3] = upd_x, upd_y, upd_w, upd_h
            return boxes
        result = process_boxes(result)

        default_boxes = tf.constant(result, name=name, dtype=tf.float32)

        return tf.reshape(default_boxes, shape=[-1, 4])

    def concat_by_index(self, index, axis):
        return tf.concat([self.conv_4_3_detector[index], self.conv_7_detector[index], self.conv_8_detector[index],
                          self.conv_9_detector[index], self.conv_10_detector[index]], axis=axis)

    def create_loss(self):
        losses = tf.map_fn(self.get_loss_per_image, (self.pred_boxes_offsets, self.pred_boxes_coors, self.pred_class_logits,
                                                     self.gnd_truth_boxes, self.gnd_truth_labels),
                           dtype=tf.float32, name='map_images')

        self.total_loss = tf.reduce_mean(losses)

    def get_loss_per_image(self, tup):

        pred_boxes_offsets, pred_boxes_coors, pred_boxes_logits, gnd_truth_boxes, gnd_truth_labels = tup

        mask, argmaxes, boxes_loss = tf.map_fn(partial(SSD.find_best_gnd_box, gnd_truth_boxes=gnd_truth_boxes),
                                               (self.default_boxes, pred_boxes_offsets, pred_boxes_coors),
                                               dtype=(tf.bool, tf.int32, tf.float32), name='map_boxes', back_prop=False)

        boxes_loss = tf.map_fn(partial(SSD._find_best_gnd_box, gnd_truth_boxes=gnd_truth_boxes),
                               (argmaxes, self.default_boxes, pred_boxes_offsets, pred_boxes_coors),
                               dtype=tf.float32, name='map_boxes')

        classification_loss = tf.map_fn(partial(SSD.get_class_loss_for_box, gnd_truth_labels=gnd_truth_labels),
                                        (pred_boxes_logits, argmaxes), dtype=tf.float32, name='map_labels')

        loss = tf.boolean_mask(boxes_loss, mask=mask) + tf.boolean_mask(classification_loss, mask=mask)

        n = tf.reduce_sum(tf.cast(mask, dtype=tf.float32))
        loss = tf.cond(n > tf.constant(0.01, dtype=tf.float32), lambda: tf.div(loss, n), lambda: tf.constant(0.0, dtype=tf.float32))

        return tf.reduce_mean(loss)

    @staticmethod
    def _find_best_gnd_box(tup, gnd_truth_boxes):
        argmax, default_box, pred_box_offsets, pred_box_coors = tup

        box_loss = SSD.get_box_loss(gnd_truth_box=gnd_truth_boxes[argmax], default_box=default_box,
                                    pred_box_coors=pred_box_coors)

        return box_loss

    @staticmethod
    def find_best_gnd_box(tup, gnd_truth_boxes):
        default_box, pred_box_offsets, pred_box_coors = tup

        sub_ious = tf.map_fn(partial(SSD.get_iou, pred_box_coors=pred_box_coors),
                             gnd_truth_boxes, dtype=tf.float32, name='map_for_ious')

        # argmax = tf.argmax(sub_ious, output_type=tf.int32)
        argmax = tf.cast(tf.argmax(sub_ious), dtype=tf.int32)

        mask, best_gnd_box = tf.cond(sub_ious[argmax] > IOU_THRESHOLD,
                                     lambda: (tf.constant(True, dtype=tf.bool), gnd_truth_boxes[argmax]),
                                     lambda: (tf.constant(False, dtype=tf.bool), gnd_truth_boxes[argmax]))

        box_loss = SSD.get_box_loss(gnd_truth_box=best_gnd_box, default_box=default_box,
                                    pred_box_coors=pred_box_coors)

        return mask, argmax, box_loss

    @staticmethod
    def compare_box_with_gnd_truth_boxes(tup, gnd_truth_boxes):
        default_box, pred_box_offsets, pred_box_coors = tup

        sub_ious = tf.map_fn(partial(SSD.get_iou, pred_box_coors=pred_box_coors),
                             gnd_truth_boxes, dtype=tf.bool, name='map_for_ious')

        sub_loss = tf.map_fn(partial(SSD.get_box_loss, default_box=default_box, pred_box_coors=pred_box_coors),
                             gnd_truth_boxes, dtype=tf.float32, name='map_for_loss')

        return sub_ious, sub_loss

    @staticmethod
    def get_class_loss_for_box(tup, gnd_truth_labels):
        pred_logits, argmax = tup

        return tf.nn.softmax_cross_entropy_with_logits(logits=pred_logits, labels=gnd_truth_labels[argmax])

    @staticmethod
    def convert_box_to_coors(coors):
        pred_x = coors[0]
        pred_y = coors[1]
        pred_width = coors[2]
        pred_height = coors[3]

        pred_xmin = pred_x - pred_width / 2
        pred_ymin = pred_y - pred_height / 2
        pred_xmax = pred_x + pred_width / 2
        pred_ymax = pred_y + pred_height / 2

        return pred_xmin, pred_ymin, pred_xmax, pred_ymax

    @staticmethod
    def restrict_coors(coors, minimum, maximum):
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = coors
        pred_xmin = tf.maximum(pred_xmin, tf.constant(minimum, dtype=tf.float32))
        pred_ymin = tf.maximum(pred_ymin, tf.constant(minimum, dtype=tf.float32))

        pred_xmax = tf.minimum(pred_xmax, tf.cast(maximum, tf.float32))
        pred_ymax = tf.minimum(pred_ymax, tf.cast(maximum, tf.float32))
        return pred_xmin, pred_ymin, pred_xmax, pred_ymax


    @staticmethod
    def get_iou(gnd_truth_box, pred_box_coors):
        # pred_x = pred_box_coors[0]
        # pred_y = pred_box_coors[1]
        # pred_width = pred_box_coors[2]
        # pred_height = pred_box_coors[3]
        #
        # pred_xmin = pred_x - pred_width / 2
        # pred_ymin = pred_y - pred_height / 2
        # pred_xmax = pred_x + pred_width / 2
        # pred_ymax = pred_y + pred_height / 2
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = SSD.restrict_coors(
            SSD.convert_box_to_coors(pred_box_coors), minimum=0.0, maximum=VVG_DEFAULT_IMAGE_SIZE)
        # pred_xmin = tf.maximum(pred_xmin, tf.constant(0.0, dtype=tf.float32))
        # pred_ymin = tf.maximum(pred_ymin, tf.constant(0.0, dtype=tf.float32))
        #
        # pred_xmax = tf.minimum(pred_xmax, tf.cast(VVG_DEFAULT_IMAGE_SIZE, tf.float32))
        # pred_ymax = tf.minimum(pred_ymax, tf.cast(VVG_DEFAULT_IMAGE_SIZE, tf.float32))

        x_left = tf.maximum(pred_xmin, gnd_truth_box[0])
        y_left = tf.maximum(pred_ymin, gnd_truth_box[1])
        x_right = tf.minimum(pred_xmax, gnd_truth_box[2])
        y_right = tf.minimum(pred_ymax, gnd_truth_box[3])

        x_diff = x_right - x_left
        y_diff = y_right - y_left

        intersection = x_diff * y_diff

        a_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
        b_area = (gnd_truth_box[2] - gnd_truth_box[0]) * (gnd_truth_box[3] - gnd_truth_box[1])

        union_area = a_area + b_area - intersection

        return tf.cond(tf.logical_or(x_diff < 0.0, y_diff < 0.0), lambda: 0.0, lambda: intersection / union_area)

    @staticmethod
    def get_box_loss(gnd_truth_box, default_box, pred_box_coors):

        pred_x = pred_box_coors[0]
        pred_y = pred_box_coors[1]
        pred_width = pred_box_coors[2]
        pred_height = pred_box_coors[3]

        gnd_truth_width = gnd_truth_box[2] - gnd_truth_box[0]
        gnd_truth_x = gnd_truth_box[0] + gnd_truth_width / 2

        gnd_truth_height = gnd_truth_box[3] - gnd_truth_box[1]
        gnd_truth_y = gnd_truth_box[1] + gnd_truth_height / 2

        g_x = (gnd_truth_x - default_box[0]) / default_box[2]
        g_y = (gnd_truth_y - default_box[1]) / default_box[3]

        def check_log(v):
            return tf.cond(tf.logical_or(tf.is_nan(v),  tf.is_inf(v)), lambda: tf.constant(0.0, dtype=tf.float32), lambda: v)

        g_w = tf.log(gnd_truth_width / default_box[2])
        g_w = check_log(g_w)
        g_h = tf.log(gnd_truth_height / default_box[3])
        g_h = check_log(g_h)

        l_x = (pred_x - default_box[0]) / default_box[2]
        l_y = (pred_y - default_box[1]) / default_box[3]

        l_w = tf.log(pred_width / default_box[2])
        l_w = check_log(l_w)
        l_h = tf.log(pred_height / default_box[3])
        l_h = check_log(l_h)

        return SSD.get_smooth_l1(ground_truths=tf.stack([g_x, g_y, g_w, g_h]),
                                 preds=tf.stack([l_x, l_y, l_w, l_h]))


    @staticmethod
    def get_smooth_l1(ground_truths, preds):
        diff = tf.abs(ground_truths - preds)
        condition = tf.less(diff, tf.constant(1.0, dtype=tf.float32))
        loss = tf.where(condition, diff, tf.square(diff))

        return tf.reduce_sum(loss, name='smooth_l1')

    def build_ssd_layers(self):
        with tf.variable_scope('ssd') as sc:
            end_points_collection = sc.name + '_end_points'
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                conv_6 = slim.conv2d(self.conv_5_3, 1024, [3, 3], padding='SAME', scope='conv6')
                self.conv_7 = slim.conv2d(conv_6, 1024, [1, 1], padding='SAME', scope='conv7')

                conv_8_1 = slim.conv2d(self.conv_7, 256, [1, 1], padding='SAME', scope='conv8_1')
                conv_8_2 = slim.conv2d(conv_8_1, 512, [3, 3], padding='SAME', scope='conv8_2')
                self.conv_8_2 = slim.max_pool2d(conv_8_2, [2, 2], scope='pool8')

                conv_9_1 = slim.conv2d(self.conv_8_2, 128, [1, 1], padding='SAME', scope='conv9_1')
                conv_9_2 = slim.conv2d(conv_9_1, 256, [3, 3], padding='SAME', scope='conv9_2')
                self.conv_9_2 = slim.max_pool2d(conv_9_2, [2, 2], scope='pool9')

                conv_10_1 = slim.conv2d(self.conv_9_2, 128, [1, 1], padding='SAME', scope='conv10_1')
                self.conv_10_2 = slim.conv2d(conv_10_1, 256, [3, 3], padding='VAlID', scope='conv10_2')

    @staticmethod
    def vgg_arg_scope(weight_decay=0.0005):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def vgg_16(self, inputs,
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
                self.conv_4_3 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(self.conv_4_3, [2, 2], scope='pool4')
                self.conv_5_3 = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(self.conv_5_3, [2, 2], scope='pool5')
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
