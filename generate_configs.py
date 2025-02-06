import argparse
from os import makedirs
from os.path import join
import itertools
import yaml

# parses line args
parser = argparse.ArgumentParser(prog="Noisy EEG",
                                 description='What the program does',
                                 epilog='Text at the bottom of help')
parser.add_argument("path", help="Where to save the configs")
parser.add_argument("--batch_size", default=128)
parser.add_argument("--max_epochs", default=30)
parser.add_argument("--seed", default=42)
parser.add_argument("--single_run", action=argparse.BooleanOptionalAction, default=False, help="Whether to stop after the first run in the set")
line_args = vars(parser.parse_args())

# creates the directory
makedirs(line_args["path"], exist_ok=True)

# defines the parameters
datasets = ["deap", "gal", "hg"]
validations = ["kfold", "loso"]
eegs_info = ["eeg", "noeeg"]
models = ["linear", "mlp", "dino", "eegnet", "edpnet", "sateer"]

# loops over each configuration
for dataset, validation, frequencies, model in itertools.product(datasets, validations, eegs_info, models):
    content = {
        "dataset": dataset,
        "dataset_path": f"../../datasets/{dataset}",
        "device": "auto",
        "model": model,
        "seed": line_args["seed"],
        "windows_size": 2,
        "windows_stride": 2,
        "min_freq": 0 if frequencies == "eeg" else 100,
        "max_freq": 100 if frequencies == "eeg" else 1000,
        "validation": validation,
        "k": 10,
        "train_perc": 0.8,
        "checkpoints_path": "./checkpoints",
        "batch_size": line_args["batch_size"],
        "max_epochs": line_args["max_epochs"],
        # "max_epochs": 50 if dataset in {"deap", "amigos"} else 100,
        "hidden_size": 512,
        "lr": 5e-5 if model in {"dino", "sateer"} else 5e-4,
        "predict_ids": True if validation not in {"loso"} else False,
        "single_run": line_args["single_run"],
    }
    filename = f"{dataset}_{validation}_{frequencies}_{model}.yaml"
    with open(join(line_args["path"], filename), 'w') as file:
        yaml.dump(content, file)