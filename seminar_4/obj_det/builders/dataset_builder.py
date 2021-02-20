import xml.etree.ElementTree as ET
import tensorflow as tf
from tqdm import tqdm

from datasets.dataset_utils import int64_feature, float_feature, bytes_feature
from .VOC_utils import VOC_LABELS, FILE_PATTERN, MAX_BOXES, get_split

slim = tf.contrib.slim


IMAGES_DIR = 'JPEGImages/'
ANNOTATIONS_DIR = 'Annotations/'
VVG_DEFAULT_IMAGE_SIZE = 224


def build(filepaths_from, filepath_to, split_name):
    dataset_file = filepath_to + FILE_PATTERN

    print(dataset_file.format(split_name))
    if not tf.gfile.Exists(dataset_file.format(split_name)):
        _build_tfrecords(filepaths_from, dataset_file.format(split_name))

    return get_split(filepath_to, split_name)


def _build_tfrecords(filepaths_from, filepath_to):
    bad_count = 0

    writer = tf.python_io.TFRecordWriter(filepath_to)

    for filepath in filepaths_from:
        images_dir = filepath + IMAGES_DIR
        annotations_dir = filepath + ANNOTATIONS_DIR
        # print(images_dir)
        images = tf.gfile.ListDirectory(images_dir)
        annotations = tf.gfile.ListDirectory(annotations_dir)

        for image_name, annotation_name in tqdm(zip(sorted(images), sorted(annotations))):
            if not image_name[:-3] == annotation_name[:-3]:
                raise ValueError('images and annotations are not aligned')

            flag, example = _convert_to_example(images_dir + image_name,
                                          annotations_dir + annotation_name)
            if flag:
                writer.write(example.SerializeToString())
            else:
                bad_count += 1
    print('bad: {}'.format(bad_count))
    writer.close()


def _convert_to_example(image_path, annotations_path):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    root = ET.parse(annotations_path).getroot()

    image_sizes = root.find('size')
    shape = (int(image_sizes.find('height').text),
             int(image_sizes.find('width').text),
             int(image_sizes.find('depth').text))

    if shape[0] < VVG_DEFAULT_IMAGE_SIZE or shape[1] < VVG_DEFAULT_IMAGE_SIZE:
        print('achtung')

        return False, None

    labels = []
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    for obj in root.findall('object'):

        bnd_box = obj.find('bndbox')

        xmin = float(bnd_box.find('xmin').text)
        ymin = float(bnd_box.find('ymin').text)
        xmax = float(bnd_box.find('xmax').text)
        ymax = float(bnd_box.find('ymax').text)

        def validate_box(xmin, ymin, xmax, ymax):
            if not xmax > xmin or not ymax > ymin:

                return False

            area = (xmax - xmin) * (ymax - ymin)
            assert area > 0.0
            if area < 5000:
                print('area: {}'.format(area))
            return area > 5000

        def process_coors(xmin, ymin, xmax, ymax, shape):
            x_offset = (shape[1] - VVG_DEFAULT_IMAGE_SIZE) / 2
            y_offset = (shape[0] - VVG_DEFAULT_IMAGE_SIZE) / 2

            xmin -= x_offset
            ymin -= y_offset
            xmax -= x_offset
            ymax -= y_offset

            def restrict_coor(coor):
                return max(min(coor, VVG_DEFAULT_IMAGE_SIZE), 0)

            xmin = restrict_coor(xmin)
            ymin = restrict_coor(ymin)
            xmax = restrict_coor(xmax)
            ymax = restrict_coor(ymax)

            flag = validate_box(xmin, ymin, xmax, ymax)

            return flag, xmin, ymin, xmax, ymax

        label = int(VOC_LABELS[obj.find('name').text][0])

        flag, xmin, ymin, xmax, ymax = process_coors(xmin, ymin, xmax, ymax, shape)
        # print(xmin, ymin, xmax, ymax)

        if not flag:
            continue

        labels.append(label)

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)

    if len(xmins) == 0:
        print('waggghhhhh')
        return False, None

    count = 0
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

    return True, example

