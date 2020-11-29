import os
import io
import hashlib
import logging
import PIL.Image

import tensorflow.compat.v1 as tf

from lxml import etree
from random import random
from shutil import copyfile
from object_detection.utils import dataset_util

from tqdm import tqdm
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

from enum import Enum

class BaseEnum(Enum):
    def __str__(self):
        return str(self.name)

class CardMapping(BaseEnum):
    Single = 1
    Seed = 2
    Numbers = 3
    Full = 4
class DatasetType(BaseEnum):
    training = 1
    evaluation = 2
    test = 3
class TFRecordCreator:

    def __init__(self,
                dataset_name,
                input_dir_path,
                card_mapping,
                n_images_shard):
        self.dataset_name = dataset_name
        self.input_dir_path = input_dir_path
        self.n_images_shard = n_images_shard

        self.img_ext = ".jpg"
        self.annotation_ext = ".xml"

        self.seeds = ["denari", "spade", "coppe", "bastoni"]
        self.max_card_number = 11
        self.numbers = range(1, self.max_card_number)

        self.mapping = card_mapping

        self.src_path_dir = os.path.join(self.input_dir_path, "source")

        self.tfrecords_path = "{}_{}_{}.records"

    ##### Main Methods #####
    def get_all_labels(self):
        annotation_path = os.path.join(self.src_path_dir, "annotated")

        all_labels = []
        for fff in os.listdir(annotation_path):
            if fff.endswith(self.annotation_ext):
                file_path = os.path.join(annotation_path, fff)
                with open(file_path, "r") as f:
                    xml_string = f.read()

                xml_doc = etree.fromstring(xml_string)
                xml_data = dataset_util.recursive_parse_xml_to_dict(xml_doc)['annotation']

                for obj in xml_data['object']:
                    all_labels.append(obj['name'])

        all_labels = list(set(all_labels))

        if self.mapping == CardMapping.Single:
            self.create_single_mapping(all_labels)
        elif self.mapping == CardMapping.Seed:
            self.create_seed_mapping(all_labels)
        elif self.mapping == CardMapping.Numbers:
            raise Exception("Unrecognised Card Mapping")
        elif self.mapping == CardMapping.Full:
            self.create_full_mapping(all_labels)
        else:
            raise Exception("Unrecognised Card Mapping")
    def separate_train_test(self, test_percentage=0.1):
        image_path = os.path.join(self.src_path_dir, "raw")
        annotation_path = os.path.join(self.src_path_dir, "annotated")

        self.train_path = os.path.join(self.input_dir_path, DatasetType.training.name)
        self.image_train_path =         os.path.join(self.train_path, "raw")
        self.annotation_train_path =    os.path.join(self.train_path, "annotated")

        os.makedirs(self.image_train_path, exist_ok=True)
        os.makedirs(self.annotation_train_path, exist_ok=True)

        self.eval_path = os.path.join(self.input_dir_path, DatasetType.evaluation.name)
        self.image_eval_path =      os.path.join(self.eval_path, "raw")
        self.annotation_eval_path = os.path.join(self.eval_path, "annotated")

        os.makedirs(self.image_eval_path, exist_ok=True)
        os.makedirs(self.annotation_eval_path, exist_ok=True)

        for fff in os.listdir(image_path):
            if fff.endswith(self.img_ext):
                annotation_file_name = fff.replace(self.img_ext, self.annotation_ext)
                src_img_path = os.path.join(image_path, fff)
                src_ann_path = os.path.join(annotation_path, annotation_file_name)
                if os.path.isfile(src_ann_path):
                    rnd_num = random()
                    if rnd_num <= test_percentage:
                        dst_img_path = os.path.join(self.image_eval_path, fff)
                        dst_ann_path = os.path.join(self.annotation_eval_path, annotation_file_name)
                    else:
                        dst_img_path = os.path.join(self.image_train_path, fff)
                        dst_ann_path = os.path.join(self.annotation_train_path, annotation_file_name)

                    copyfile(src_img_path, dst_img_path)
                    copyfile(src_ann_path, dst_ann_path)
    def create(self, label_map_path=None):
        self.save_label_map()
        self._create(self.image_train_path, self.annotation_train_path, self.train_path, DatasetType.training.name)
        self._create(self.image_eval_path, self.annotation_eval_path, self.eval_path, DatasetType.evaluation.name)
    ##### End #####

    ##### Mapping Helpers #####
    def create_single_mapping(self, input_label_list):
        class_name = "sicilian_card"
        self.intermediate_label_dict = {}
        for lll in input_label_list:
            self.intermediate_label_dict[lll] = class_name

        self.get_one_class_label_map_dict(class_name)
    def create_seed_mapping(self, input_label_list):
        self.intermediate_label_dict = {}
        for lll in input_label_list:
            l_arro = lll.split("_")
            current_seed = l_arro[0]
            if current_seed in self.seeds:
                self.intermediate_label_dict[lll] = current_seed
            else:
                raise Exception("Unrecognised seed for card: {0}".format(lll))

        self.get_seed_label_map_dict()
    def create_full_mapping(self, input_label_list):
        self.intermediate_label_dict = {}
        for lll in input_label_list:
            l_arro = lll.split("_")
            current_seed = l_arro[0]
            current_number = int(l_arro[1])
            if current_seed in self.seeds:
                if current_number > 0 and current_number < self.max_card_number:
                    self.intermediate_label_dict[lll] = "{0}_{1:02d}".format(current_seed, current_number)
                else:
                    raise Exception("Unrecognised number for card: {0}".format(lll))
            else:
                raise Exception("Unrecognised seed for card: {0}".format(lll))

        self.get_full_label_map_dict()

    def get_one_class_label_map_dict(self, class_name):
        categories = []
        categories.append([1, class_name])

        self.get_label_dict_from_categories(categories)
    def get_seed_label_map_dict(self):
        categories = []
        count = 0
        for sss in self.seeds:
            count += 1
            categories.append([count, sss])

        self.get_label_dict_from_categories(categories)
    def get_number_label_map_dict(self):
        categories = []
        for idx in self.numbers:
            categories.append([idx, "{}".format(idx)])

        self.get_label_dict_from_categories(categories)
    def get_full_label_map_dict(self):
        categories = []
        count = 0
        for sss in self.seeds:
            for idx in self.numbers:
                count += 1
                categories.append([count, "{0}_{1:02d}".format(sss, idx)])

        self.get_label_dict_from_categories(categories)

    def get_label_dict_from_categories(self, categories):
        self.label_map_dict = {}

        for idx, name in categories:
            self.label_map_dict[name] = {'id': idx, 'name': name}

        self.num_classes = len(categories)
        print("Found {0} categories.".format(self.num_classes))

        self.categories = categories
    ##### End #####

    ##### Helpers #####
    def _create(self, image_path, annotation_path, output_path, train_val_test):
        examples_list = self.read_examples_list(image_path)
        src_example_length = len(examples_list)
        n_shards = int(src_example_length / self.n_images_shard) + (1 if src_example_length % self.n_images_shard != 0 else 0)

        index = 0
        for shard in tqdm(range(n_shards)):
            tf_filename = self.tfrecords_path.format(self.dataset_name, train_val_test, '%.5d-of-%.5d' % (shard, n_shards - 1))
            tfrecords_shard_path = os.path.join(output_path, tf_filename)
            end = index + self.n_images_shard if src_example_length > (index + self.n_images_shard) else -1
            images_shard_list = examples_list[index: end]
            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for filename in images_shard_list:
                    path = os.path.join(annotation_path, filename + self.annotation_ext)

                    with tf.gfile.GFile(path, 'r') as fid:
                        xml_str = fid.read()
                    xml = etree.fromstring(xml_str)
                    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                    tf_example = self.dict_to_tf_example(data, image_path, False)
                    writer.write(tf_example.SerializeToString())
            index = end
    def read_examples_list(self, input_dir_path):
        train_images_names = []
        for root, dirs, files in os.walk(input_dir_path):
            for fff in files:
                if fff.endswith(self.img_ext):
                    fff_key = fff.replace(self.img_ext, "")
                    train_images_names.append(fff_key)

        return train_images_names
    def dict_to_tf_example(self, data,
                        dataset_directory,
                        ignore_difficult_instances=False,
                        image_subdirectory='JPEGImages'):
        full_path = os.path.join(dataset_directory, data['filename'])
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
            raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()

        width = int(data['size']['width'])
        height = int(data['size']['height'])

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        if 'object' in data:
            for obj in data['object']:
                difficult = bool(int(obj['difficult']))
                if ignore_difficult_instances and difficult:
                    continue

                difficult_obj.append(int(difficult))

                xmin.append(float(obj['bndbox']['xmin']) / width)
                ymin.append(float(obj['bndbox']['ymin']) / height)
                xmax.append(float(obj['bndbox']['xmax']) / width)
                ymax.append(float(obj['bndbox']['ymax']) / height)

                mapped_class = self.intermediate_label_dict[obj['name']]
                classes_text.append(mapped_class.encode('utf8'))
                classes.append(self.label_map_dict[mapped_class]["id"])

                truncated.append(int(obj['truncated']))
                poses.append(obj['pose'].encode('utf8'))

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/source_id': dataset_util.bytes_feature(
                data['filename'].encode('utf8')),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))
        return example
    def save_label_map(self):
        label_map = string_int_label_map_pb2.StringIntLabelMap()

        for kkk, vvv in self.label_map_dict.items():
            new_item = string_int_label_map_pb2.StringIntLabelMapItem()
            new_item.name = vvv['name']
            new_item.id = vvv['id']
            label_map.item.append(new_item)

        lb_string = text_format.MessageToString(label_map)
        label_map_path = os.path.join(self.input_dir_path, "label_map.pbtxt")
        with tf.gfile.Open(label_map_path, "wb") as f:                                                                                                                                                                                                                       
            f.write(lb_string)
        print("Label map saved: {0}".format(label_map_path))
    ##### End #####

if __name__ == "__main__":
    input_dir = ''

    create_tfrecord = TFRecordCreator("sicilian_cards", input_dir, CardMapping.Full, 200)
    create_tfrecord.get_all_labels()
    create_tfrecord.separate_train_test()
    create_tfrecord.create()
