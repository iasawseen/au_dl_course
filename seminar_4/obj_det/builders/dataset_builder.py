import xml.etree.ElementTree as ET
import tensorflow as tf
from tqdm import tqdm

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from .VOC_utils import VOC_LABELS, FILE_PATTERN, MAX_BOXES, get_split

slim = tf.contrib.slim


IMAGES_DIR = 'JPEGImages/'
ANNOTATIONS_DIR = 'Annotations/'


def build(filepaths_from, filepath_to, split_name):
    dataset_file = filepath_to + FILE_PATTERN

    # if not tf.gfile.Exists(filepath_to):
    # _build_tfrecords(filepaths_from, dataset_file.format(split_name))

    return get_split(filepath_to, split_name)


def _build_tfrecords(filepaths_from, filepath_to):

    writer = tf.python_io.TFRecordWriter(filepath_to)

    for filepath in filepaths_from:
        images_dir = filepath + IMAGES_DIR
        annotations_dir = filepath + ANNOTATIONS_DIR
        images = tf.gfile.ListDirectory(images_dir)
        annotations = tf.gfile.ListDirectory(annotations_dir)

        for image_name, annotation_name in tqdm(zip(sorted(images), sorted(annotations))):
            if not image_name[:-3] == annotation_name[:-3]:
                raise ValueError('images and annotations are not aligned')

            example = _convert_to_example(images_dir + image_name,
                                          annotations_dir + annotation_name)

            writer.write(example.SerializeToString())

    writer.close()


def _convert_to_example(image_path, annotations_path):

    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    root = ET.parse(annotations_path).getroot()

    image_sizes = root.find('size')
    shape = (int(image_sizes.find('height').text),
             int(image_sizes.find('width').text),
             int(image_sizes.find('depth').text))

    labels = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    count = 0

    for obj in root.findall('object'):
        count += 1
        labels.append(VOC_LABELS[obj.find('name').text][0])
        bnd_box = obj.find('bndbox')

        def get_norm_coor(name, scale):
            return float(bnd_box.find(name).text) / scale

        xmins.append(get_norm_coor(name='xmin', scale=shape[1]))
        ymins.append(get_norm_coor(name='ymin', scale=shape[0]))
        xmaxs.append(get_norm_coor(name='xmax', scale=shape[1]))
        ymaxs.append(get_norm_coor(name='ymax', scale=shape[0]))

    while count < MAX_BOXES:
        labels.append(0)
        xmins.append(0)
        ymins.append(0)
        xmaxs.append(0)
        ymaxs.append(0)
        count += 1

    assert count == MAX_BOXES

    image_format = b'JPEG'

    example = tf.train.Example(features=tf.train.Features(feature={
            'data': bytes_feature(image_data),
            'format': bytes_feature(image_format),
            'shape': int64_feature(shape),
            'qty': int64_feature(count),
            'bndbox/xmin': float_feature(xmins),
            'bndbox/ymin': float_feature(ymins),
            'bndbox/xmax': float_feature(xmaxs),
            'bndbox/ymax': float_feature(ymaxs),
            'bndbox/label': int64_feature(labels),

    }))

    return example

