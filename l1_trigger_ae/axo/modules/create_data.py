import os
import json
import h5py
from .. import data_util


def run(config):
    processed_data_path = config["data_config"]["Processed_data_path"]

    if os.path.isfile(processed_data_path):
        print("File already exists, checking if the config match")
        f = h5py.File(processed_data_path, "r")
        existing_ser_config = f.attrs["config"]
        f.close()
        present_ser_config = json.dumps(config["data_config"])
        if existing_ser_config == present_ser_config:
            print("Configs match, skipping")
        else:
            print("[WARNING]: CONFIG DO NOT MATCH, OVERWRITING!!!")
            data_util.data.get_data(config_master=config["data_config"])
    else:
        print("File does not exist, creating data file")
        data_util.data.get_data(config_master=config["data_config"])
