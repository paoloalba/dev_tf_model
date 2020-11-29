import os
import logging

from object_detection.utils import config_util

from .create_tf_records import DatasetType

class MLPipelinePatcher:

    def __init__(self,
        path_perm_storage,
        src_train_path):
        self.path_perm_storage = path_perm_storage
        self.src_train_path = src_train_path

        self.batch_size = self.get_double_from_env("BATCH_SIZE", "40")
        self.num_classes = self.get_double_from_env("NUM_CLASSES", "1")

        self.record_ext = ".records"

    ##### Helpers #####
    @staticmethod
    def get_double_from_env(env_string, default_val=None):
        os_env_val = os.getenv(env_string, default_val)
        return int(os_env_val)
    @staticmethod
    def get_boolean_from_env(env_string, default_val=None):
        os_env_val = os.getenv(env_string, default_val)
        if 'y' in os_env_val.lower():
            return True
        elif 'n' in os_env_val.lower():
            return False
        else:
            raise Exception("Unparsable boolean from env variable: {0} -> {1}".format(env_string, os_env_val))
    def scan_dir_for_records(self, data_type):
        records_list = []
        records_dir = os.path.join(self.src_train_path, data_type)
        for fff in os.listdir(records_dir):
            if fff.endswith(self.record_ext):
                records_list.append(os.path.join(records_dir, fff))
        if len(records_list) > 0:
            return records_list
        else:
            raise Exception("No tf record files found for {1} in {0}".format(records_dir, data_type))
    ##### End #####

    ##### Main #####
    def patch_pipeline_config(self, model_base_name):
        self.label_map_path = os.path.join(self.src_train_path, "label_map.pbtxt")

        model_base_dir_path = os.path.join(self.path_perm_storage, "model_base_checkpoints", model_base_name)
        config_path = os.path.join(model_base_dir_path, "pipeline.config")

        cf_dict = config_util.get_configs_from_pipeline_file(config_path)

        cf_dict["model"].ssd.num_classes = self.num_classes

        cf_dict["train_config"].fine_tune_checkpoint = os.path.join(model_base_dir_path, "ckpt-0")
        cf_dict["train_config"].batch_size = self.batch_size
        cf_dict["train_config"].use_bfloat16 = False

        cf_dict["train_input_config"].label_map_path = self.label_map_path
        cf_dict["train_input_config"].tf_record_input_reader.input_path[:] = self.scan_dir_for_records(DatasetType.training.name)

        cf_dict["eval_input_config"].label_map_path = self.label_map_path
        cf_dict["eval_input_config"].tf_record_input_reader.input_path[:] = self.scan_dir_for_records(DatasetType.evaluation.name)

        cf_obj = config_util.create_pipeline_proto_from_configs(cf_dict)
        tmp_config_path = os.path.join(self.path_perm_storage, "patched_config")
        config_util.save_pipeline_config(cf_obj, tmp_config_path)
        self.patched_config_path = os.path.join(tmp_config_path, "pipeline.config")
        print("Source configuration was patched: {0}".format(self.patched_config_path))
    ##### End #####
