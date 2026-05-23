## This wrapper will be removed once the entire module becomes installable packagw

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
import axo
import argparse
import yaml

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Process some configurations.")
    parser.add_argument('--config_path',
        nargs='?',
        default=None,
        help="Path to the configuration file (optional, if not provided runs with default config)."
    )

    parser.add_argument('--data_creation', 
                        action='store_true', 
                        help="Flag to indicate whether to create data (if set, train_model is set to False).")
    
    parser.add_argument('--train_model', 
                        action='store_true', 
                        help="Flag to indicate whether to train the model (if set, data_creation and inference is set to False).") 

    parser.add_argument('--inference', 
                        action='store_true', 
                        help="Flag to indicate whether to take predictions (if set, data_creation and train_model is set to False).") 

    args = parser.parse_args()

    if not (args.data_creation or args.train_model or args.inference):
        print("ERROR: You must pass at least one of the flags: --data_creation, --train_model, or --inference.")
        sys.exit(1)

    if args.config_path is None:
        print("No config found, running with default")
        axo.master.main(data_creation=args.data_creation, train_model=args.train_model)
    else:
        slave = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
        axo.master.main(slave=slave, data_creation=args.data_creation, train_model=args.train_model, inference=args.inference)

print("Success")
