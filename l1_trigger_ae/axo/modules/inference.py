from keras.models import load_model
from .. import metric


def run(config):
    trimmed_encoder = load_model(config["trimmed_encoder"])

    print("[INFO] Model loaded successfully.")
    trimmed_encoder.summary()

    config_thres = config["threshold"]
    config_thres["data_path"] = config["data_config"]["Processed_data_path"]
    config_thres["inference"] = config["inference"]

    axo_man = metric.axo_threshold_manager(
        model=trimmed_encoder,
        config=config_thres,
        inference=True
    )
    raw_score_dict, threshold_dict = axo_man.get_raw_dict()
    print(raw_score_dict)
    print(threshold_dict)
