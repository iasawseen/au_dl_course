import tensorflow as tf
slim = tf.contrib.slim


FILE_PATTERN = 'VOC2007_{}.tfrecords'

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'bndbox': 'A list of bounding boxes, one per each object.',
    'label': 'A list of labels, one per each object.',
}

SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
}

NUM_CLASSES = 20
MAX_BOXES = 42

VOC_LABELS = {
    'none': (0, 'Background'),
    'aeroplane': (1, 'Vehicle'),
    'bicycle': (2, 'Vehicle'),
    'bird': (3, 'Animal'),
    'boat': (4, 'Vehicle'),
    'bottle': (5, 'Indoor'),
    'bus': (6, 'Vehicle'),
    'car': (7, 'Vehicle'),
    'cat': (8, 'Animal'),
    'chair': (9, 'Indoor'),
    'cow': (10, 'Animal'),
    'diningtable': (11, 'Indoor'),
    'dog': (12, 'Animal'),
    'horse': (13, 'Animal'),
    'motorbike': (14, 'Vehicle'),
    'person': (15, 'Person'),
    'pottedplant': (16, 'Indoor'),
    'sheep': (17, 'Animal'),
    'sofa': (18, 'Indoor'),
    'train': (19, 'Vehicle'),
    'tvmonitor': (20, 'Indoor'),
}

LABELS_TO_NAMES = {VOC_LABELS[key][0]: key for key in VOC_LABELS}


def get_split(dataset_dir, split_name):

    keys_to_features = {
        'data': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
        'format': tf.FixedLenFeature((), tf.string, default_value=''),
        'shape': tf.FixedLenFeature([3], tf.int64),
        'qty': tf.FixedLenFeature([1], tf.int64),
        'bndbox/xmin': tf.FixedLenFeature([MAX_BOXES], dtype=tf.float32),
        'bndbox/ymin': tf.FixedLenFeature([MAX_BOXES], dtype=tf.float32),
        'bndbox/xmax': tf.FixedLenFeature([MAX_BOXES], dtype=tf.float32),
        'bndbox/ymax': tf.FixedLenFeature([MAX_BOXES], dtype=tf.float32),
        'bndbox/label': tf.FixedLenFeature([MAX_BOXES], dtype=tf.int64)

        # 'bndbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        # 'bndbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        # 'bndbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        # 'bndbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        # 'bndbox/label': tf.VarLenFeature(dtype=tf.int64)

    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('data', 'format'),
        'shape': slim.tfexample_decoder.Tensor('shape'),
        'qty': slim.tfexample_decoder.Tensor('qty'),

        'xmin': slim.tfexample_decoder.Tensor('bndbox/xmin'),
        'ymin': slim.tfexample_decoder.Tensor('bndbox/ymin'),
        'xmax': slim.tfexample_decoder.Tensor('bndbox/xmax'),
        'ymax': slim.tfexample_decoder.Tensor('bndbox/xmax'),

        # 'bndbox': slim.tfexample_decoder.BoundingBox(
        #     ['xmin', 'ymin', 'xmax', 'ymax'], 'bndbox/'),
        'label': slim.tfexample_decoder.Tensor('bndbox/label')
    }

    reader = tf.TFRecordReader
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    file_pattern = dataset_dir + FILE_PATTERN

    return slim.dataset.Dataset(
        data_sources=file_pattern.format(split_name),
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES,
        labels_to_names=LABELS_TO_NAMES)
