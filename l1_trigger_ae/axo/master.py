import os
import yaml
from . import utilities


def tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor)


def main(slave={}, data_creation=False, train_model=False, inference=False):

    module_dir = os.path.dirname(__file__)
    master = yaml.load(open(os.path.join(module_dir, "utilities/config.yml"), "r"), Loader=yaml.Loader)

    if utilities.check_compartibility(master=master, slave=slave) != 1:
        print("Smoke test failed !!! check the dictionary and the documentations")
        return 0

    print("Configurations are compartible")
    print("Generating new config ....")
    config = utilities.merge_dict(master=master, slave=slave)

    _setup_reproducibility(config)

    if data_creation:
        from .modules.create_data import run as run_data_creation
        run_data_creation(config)

    if train_model:
        from .modules.train import run as run_training
        run_training(config)

    if inference:
        from .modules.inference import run as run_inference
        run_inference(config)

    print("Run completing exiting ...")


def _setup_reproducibility(config):
    import os
    import numpy as np
    import tensorflow as tf

    seed = config["determinism"]["global_seed"]
    if config["determinism"]["python_determinism"]:
        os.environ["PYTHONHASHSEED"] = str(seed)
    if config["determinism"]["numpy_determinism"]:
        np.random.seed(seed)
    if config["determinism"]["tf_op_determinism"]:
        tf.random.set_seed(seed)
        tf.config.experimental.enable_op_determinism()
