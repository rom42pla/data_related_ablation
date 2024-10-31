

if __name__ == '__main__':
    import colorama
    colorama.init()
    
    import argparse
    import os
    import gc
    import yaml
    from loguru import logger
    from datetime import datetime
    from os import makedirs
    from os.path import join, splitext, basename
    from pprint import pformat
    from typing import Union, Dict

    import torch
    torch.set_float32_matmul_precision("high")

    from torch.utils.data import Subset, DataLoader
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch import Trainer
    import wandb

    from utils import set_global_seed, get_k_fold_runs, get_loso_runs, get_simple_runs
    from models.linear import Linear4EEG
    from models.mlp import MLP4EEG
    from models.dino4eeg import DINO4EEG
    from datasets.base_class import EEGClassificationDataset
    from models.base_model import EEGClassificationModel
    from datasets.deap import DEAPDataset
    from datasets.amigos import AMIGOSDataset
    from datasets.grasp_and_lift import GraspAndLiftDataset
    from datasets.high_gamma import HighGammaDataset

    import torchaudio

    # parses line args
    parser = argparse.ArgumentParser(prog="Noisy EEG")
    parser.add_argument('cfg')
    line_args = vars(parser.parse_args())
    
    # torchaudio.set_audio_backend("sox_io")
    num_workers = os.cpu_count() // 2
    with open(line_args["cfg"], 'r') as fp:
        args = yaml.safe_load(fp)
    logger.info(f"args:\n{pformat(args)}")

    # sets the random seed
    set_global_seed(seed=args['seed'])

    # sets the logging folder
    datetime_str: str = datetime.now().strftime("%Y%m%d_%H:%M")
    experiment_name: str = f"{datetime_str}_{args['dataset']}_size={args['windows_size']}_stride={args['windows_stride']}"
    experiment_path: str = join(args['checkpoints_path'], experiment_name)
    makedirs(experiment_path, exist_ok=True)

    # sets up the dataset
    if args["dataset"] == "deap":
        dataset_class = DEAPDataset
    elif args["dataset"] == "amigos":
        dataset_class = AMIGOSDataset
    elif args["dataset"] in {"grasp_and_lift", "gal"}:
        dataset_class = GraspAndLiftDataset
    elif args["dataset"] in {"high_gamma", "hg"}:
        dataset_class = HighGammaDataset
    else:
        raise NotImplementedError(f"unknown dataset {args['dataset']}")
    dataset: EEGClassificationDataset = dataset_class(
        path=args['dataset_path'],
        window_size=args['windows_size'],
        window_stride=args['windows_stride'],
        min_freq=args["min_freq"],
        max_freq=args["max_freq"],
        drop_last=True,
        discretize_labels=True,
        normalize_eegs=True,
    )

    if args['validation'] in ["k_fold", "kfold"]:
        runs = get_k_fold_runs(k=args["k"], dataset=dataset)
    elif args['validation'] == "loso":
        runs = get_loso_runs(dataset=dataset)
    elif args['validation'] == "simple":
        runs = get_simple_runs(
            dataset=dataset, train_perc=args["train_perc"])
    else:
        raise NotImplementedError

    # instantiate the model
    if args["device"] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args["device"] in {"cuda", "gpu"}:
        device = "cuda"
    elif args["device"] == "cpu":
        device = "cpu"
    else:
        raise NotImplementedError(f"unrecognized device {args['device']}")
    if args["model"] == "linear":
        model_class = Linear4EEG
    elif args["model"] == "mlp":
        model_class = MLP4EEG
    elif args["model"] == "dino":
        model_class = DINO4EEG
    else:
        raise NotImplementedError(f"model {args['model']} not implemented")
    model: EEGClassificationModel = model_class(
        eeg_sampling_rate=dataset.sampling_rate,
        eeg_num_channels=len(dataset.electrodes),
        eeg_samples=dataset[0]["eegs"].shape[-1],
        num_labels=len(dataset.labels),
        min_freq=args["min_freq"],
        max_freq=args["max_freq"],
        predict_ids=args["predict_ids"],
        h_dim=args["hidden_size"],
        ids=dataset.subject_ids,
    )
    
    # saves the initial weights
    print(model)
    initial_state_dict_path = join(".", "_initial_state_dict.pth")
    torch.save({"model_state_dict": model.state_dict()},
               initial_state_dict_path)

    # metas
    date = datetime.now().strftime("%Y%m%d_%H%M")
    cfg_name = splitext(basename(line_args["cfg"]))[0]
    run_name = f"{date}_{cfg_name}"
    
    # loops over runs
    for i_run, run in enumerate(runs):
        logger.info(
            f"{run_name} with frequencies in [{dataset.min_freq}, {dataset.max_freq}]: doing run {i_run+1} of {len(runs)} ({((i_run+1)/len(runs)) * 100:.1f}%)")

        # splits the dataset
        dataloader_train = DataLoader(
            dataset=Subset(dataset, indices=run["train_idx"]),
            batch_size=args["batch_size"],
            shuffle=True, pin_memory=True, num_workers=num_workers)
        dataloader_val = DataLoader(
            dataset=Subset(dataset, indices=run["val_idx"]),
            batch_size=args["batch_size"],
            shuffle=False, pin_memory=True, num_workers=num_workers)

        # initialize the model
        model.to("cpu")
        model.load_state_dict(torch.load(
            initial_state_dict_path)['model_state_dict'])
        model.to(device)
        
        wandb_logger = WandbLogger(
            project="noisy_eeg", name=run_name, log_model=False, prefix=f"run_{i_run}")
        # dataset.plot_labels_distribution()
        trainer = Trainer(logger=wandb_logger, accelerator=device,
                          precision="16-mixed", max_epochs=args["max_epochs"],
                          enable_model_summary=True)
        trainer.fit(model, dataloader_train, dataloader_val)
        if args["single_run"]:
            break
    wandb.finish()

    # frees some memory
    del dataset
    gc.collect()
