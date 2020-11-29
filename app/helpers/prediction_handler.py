import os
import cv2
import json
import xmltodict

from collections import OrderedDict
from shutil import copyfile

class PredictionHandler:

    def __init__(self,
                input_dir,
                output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.annotation_key = "annotations"
        self.raw_key = "raw"

        self.json_ext = ".json"
        self.xml_ext = ".xml"
        self.img_ext = ".jpg"

    ##### Main methods #####
    @staticmethod
    def export_folder(input_dir, output_dir, game_name, game_user):
        new_files = 0
        for fff in os.listdir(input_dir):
            src_file_path = os.path.join(input_dir, fff)
            dst_file_path = os.path.join(output_dir, "-".join([game_user, game_name, fff]))
            if not os.path.isfile(dst_file_path):
                copyfile(src_file_path, dst_file_path)
                new_files += 1
        return new_files
    def get_obj_list(self, input_obj_list, img_width, img_height):
        obj_list = []
        for ooo in input_obj_list:
            box = ooo["box"]
            class_name = ooo["class"]
            ymin, xmin, ymax, xmax = box

            new_obj = OrderedDict()

            new_obj["name"] = class_name
            new_obj["pose"] = "Unspecified"
            new_obj["truncated"] = 0
            new_obj["difficult"] = 0

            new_obj["bndbox"] = {}
            new_obj["bndbox"]["xmin"] = int(xmin * img_width)
            new_obj["bndbox"]["ymin"] = int(ymin * img_height)
            new_obj["bndbox"]["xmax"] = int(xmax * img_width)
            new_obj["bndbox"]["ymax"] = int(ymax * img_height)

            obj_list.append(new_obj)
        return obj_list
    def annotation_xml_template(self,
                            file_name,
                            file_path,
                            img_width,
                            img_height,
                            img_depth,
                            input_obj_list):
        new_dict = OrderedDict()

        annotation_dict = OrderedDict()

        annotation_dict["folder"] = "raw"
        annotation_dict["filename"] = file_name
        annotation_dict["path"] = file_path
        annotation_dict["source"] = {"database": "Unknown"}
        annotation_dict["size"] = {"width": img_width, "height": img_height, "depth": img_depth}
        annotation_dict["segmented"] = 0
        annotation_dict["object"] = self.get_obj_list(input_obj_list, img_width, img_height)

        new_dict["annotation"] = annotation_dict

        return new_dict
    ##### End #####

    ##### Main methods #####
    def collect_input(self):
        dst_annot_path = os.path.join(self.output_dir, self.annotation_key + "_json")
        os.makedirs(dst_annot_path, exist_ok=True)

        dst_raw_path = os.path.join(self.output_dir, self.raw_key)
        os.makedirs(dst_raw_path, exist_ok=True)

        new_annotations = 0
        new_images = 0
        for root_dir, dirs, files in os.walk(self.input_dir):
            if self.annotation_key in dirs and self.raw_key in dirs:
                arro = root_dir.replace(self.input_dir, "").split(os.sep)
                game_name = arro[-1]
                game_user = arro[-2]

                src_annotations_path = os.path.join(root_dir, self.annotation_key)
                src_raw_path = os.path.join(root_dir, self.raw_key)

                tmp_ann = self.export_folder(src_annotations_path, dst_annot_path, game_name, game_user)
                tmp_img = self.export_folder(src_raw_path, dst_raw_path, game_name, game_user)

                new_annotations += tmp_ann
                new_images += tmp_img

                print("Found {0} new annotations and {1} new images: {2}".format(tmp_ann, tmp_img, root_dir))
        print("Found {0} new annotations and {1} new images".format(new_annotations, new_images))
    def create_xml(self):
        src_dir = os.path.join(self.output_dir, self.annotation_key + "_json")
        src_img_dir = os.path.join(self.output_dir, self.raw_key)

        dst_dir = os.path.join(self.output_dir, self.annotation_key)
        os.makedirs(dst_dir, exist_ok=True)
        count = 0
        for fff in os.listdir(src_dir):
            if fff.endswith(self.json_ext):
                base_file_name = fff.replace(self.json_ext, "")
                output_file_path = os.path.join(dst_dir, base_file_name + self.xml_ext)
                if not os.path.isfile(output_file_path):
                    src_file_path = os.path.join(src_dir, fff)
                    with open(src_file_path, "r") as f:
                        src_dict = json.load(f)

                    src_img_path = os.path.join(src_img_dir, base_file_name + self.img_ext)
                    src_img = cv2.imread(src_img_path)
                    im_height, im_width, im_channels = src_img.shape

                    output_dict = self.annotation_xml_template(fff, src_file_path, im_width, im_height, im_channels, src_dict)
                    xml_string = xmltodict.unparse(output_dict, pretty=True)
                    with open(output_file_path, "w") as f:
                        f.write(xml_string)
                    count += 1
                    print("{1:03d} annotations saved: {0}".format(output_file_path, count))
    ##### End #####

if __name__ == "__main__":
    input_dir = ""
    output_dir = ""
    pred_handler = PredictionHandler(input_dir, output_dir)

    pred_handler.collect_input()
    pred_handler.create_xml()
