import tensorflow as tf

from builders import dataset_builder, model_builder
from preprocessing import vgg_preprocessing
from functools import partial

slim = tf.contrib.slim


FILEPATH_FROM = '/home/ivan/Data/deep/course_au/au_dl_course/' \
                'seminar_4/obj_det/data/VOC_train/VOCdevkit/'

FILEPATH_TO = '/home/ivan/Data/deep/course_au/au_dl_course/' \
              'seminar_4/obj_det/data/'


def load_batch(dataset, batch_size=32, is_training=True):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image_raw, shape, qty, xmin, ymin, xmax, ymax, \
    label = data_provider.get(['image', 'shape', 'qty',
                               'xmin', 'ymin', 'xmax', 'ymax',
                               'label'])

    shape = tf.cast(shape, dtype=tf.int32)
    qty = tf.cast(qty, dtype=tf.int32)
    label = tf.cast(label, dtype=tf.int32)
    # print(image_raw)
    # print(shape)
    # print(qty)
    # print(xmin)
    # print(ymin)
    # print(xmax)
    # print(ymax)

    image = vgg_preprocessing.preprocess_image(image_raw,
                                               model_builder.vgg_16.default_image_size,
                                               model_builder.vgg_16.default_image_size,
                                               is_training=is_training)

    bndbox = tf.stack([xmin, ymin, xmax, ymax], name='stacking_coors')
    bndbox = tf.transpose(bndbox)

    images, images_raw, qtys, bndboxes, labels = tf.train.batch(
        [image, image_raw, qty, bndbox, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size,
        dynamic_pad=True)

    # print(images)
    # print(images_raw)
    # print(qtys)
    # print(bndboxes)
    # print(labels)

    return images, images_raw, qtys, bndboxes, labels


def main():

    with tf.Graph().as_default():
        dataset = dataset_builder.build([FILEPATH_FROM + 'VOC2007/'], FILEPATH_TO, 'train')
        images, _, qtys, bndboxes, labels = load_batch(dataset)

        train_op, init_assign_op, init_feed_dict = \
            model_builder.build_faster_rcnn(images, qtys, bndboxes, labels)

        # print(init_feed_dict)

        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)
            tf.train.start_queue_runners(sess)

        slim.learning.train(
            train_op=train_op,
            logdir=model_builder.CHECKPOINTS_DIR,
            log_every_n_steps=1,
            init_fn=InitAssignFn,
            number_of_steps=1,
            save_summaries_secs=300,
            save_interval_secs=600)


if __name__ == '__main__':
    main()
