import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
import os
import skimage.io as io

from builders import dataset_builder, model_builder
from preprocessing import vgg_preprocessing
from functools import partial
from models import ssd

slim = tf.contrib.slim

# FILEPATH_FROM = '/home/ivan/Data/deep/course_au/au_dl_course/' \
#                 'seminar_4/obj_det/data/VOC_train/VOCdevkit/'

FILEPATH_FROM = '/data/VOC_train/VOCdevkit/'

FILEPATH_TO = '/data/'
CHECKPOINTS_DIR = '/checkpoints/'


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def load_batch(dataset, batch_size=ssd.BATCH_SIZE, is_training=True):
    data_provider = slim.dataset_data_provider.DatasetDataProvider(dataset)

    image_raw, shape, qty, xmin, ymin, xmax, ymax, \
    label = data_provider.get(['image', 'shape', 'qty',
                               'xmin', 'ymin', 'xmax', 'ymax',
                               'label'])

    shape = tf.cast(shape, dtype=tf.int32)
    qty = tf.cast(qty, dtype=tf.int32)
    label = tf.cast(label, dtype=tf.int32)

    image = preprocess_image(image_raw,
                             ssd.VVG_DEFAULT_IMAGE_SIZE,
                             ssd.VVG_DEFAULT_IMAGE_SIZE,
                             is_training=is_training)

    bndbox = tf.stack([xmin, ymin, xmax, ymax], name='stacking_coors')
    bndbox = tf.transpose(bndbox)

    images, images_raw, qtys, bndboxes, labels = tf.train.batch(
        [image, image_raw, qty, bndbox, label],
        batch_size=batch_size,
        num_threads=10,
        capacity=2*batch_size,
        dynamic_pad=True)

    return images, images_raw, qtys, bndboxes, labels


def preprocess_image(image, output_height, output_width, is_training):

    image = tf.image.resize_image_with_crop_or_pad(image=image, target_width=output_width, target_height=output_height)
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    image = tf.image.random_flip_left_right(image)

    # means = [_R_MEAN, _G_MEAN, _B_MEAN]

    return image

    # channels = tf.split(axis=2, num_or_size_splits=3, value=image)
    # for i in range(3):
    #     channels[i] -= means[i]
    # return tf.concat(axis=2, values=channels)


def main():
    print('starting')
    with tf.Graph().as_default():
        dataset = dataset_builder.build([os.getcwd() + FILEPATH_FROM + 'VOC2007/'], os.getcwd() + FILEPATH_TO, 'train')
        print('dataset is created')
        images, _, _, bndboxes, labels = load_batch(dataset)
        print('batches are ready')

        model = ssd.SSD()

        train_op, init_assign_op, init_feed_dict = \
            model.build(images, bndboxes, labels)

        print('model is builded')

        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)
            writer = tf.summary.FileWriter(logdir='summary/', graph=sess.graph)
            writer.flush()
            writer.close()

        images_with_bndboxes = model.get_images_with_boxes()

        init_assign_op_val, init_feed_dict_val = \
            slim.assign_from_checkpoint(os.path.join(os.getcwd()+CHECKPOINTS_DIR, 'model.ckpt'),
                                        slim.get_model_variables())

        with tf.Session() as sess:
            # InitAssignFn(sess)
            # sess.run(init_assign_op)
            # sess.run(tf.global_variables_initializer())
            # saver.restore(sess, '/checkpoints/model.ckpt')
            sess.run(init_assign_op_val, init_feed_dict_val)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Let's read off 3 batches just for example

            for i in range(3):
                imgs = sess.run(images_with_bndboxes)
                imgs = imgs / 256.0
                print(imgs[0, :, :, :].shape)

                print('current batch')
                print(imgs[0, :, :, :])
                # We selected the batch size of two
                # So we should get two image pairs in each batch
                # Let's make sure it is random

                io.imshow(imgs[0, :, :, :])
                io.show()

                io.imshow(imgs[1, :, :, :])
                io.show()

            coord.request_stop()
            coord.join(threads)

        # slim.learning.train(
        #     train_op=train_op,
        #     logdir=os.getcwd() + CHECKPOINTS_DIR,
        #     log_every_n_steps=1,
        #     init_fn=InitAssignFn,
        #     number_of_steps=1000,
        #     save_summaries_secs=300,
        #     save_interval_secs=600)


if __name__ == '__main__':
    main()
