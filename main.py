if __name__ == '__main__':
    import gc
    import yaml
    from loguru import logger
    from datetime import datetime
    from os import makedirs
    from os.path import join
    from pprint import pformat
    from typing import Union, Dict

    from torch.utils.data import Subset

    # from arg_parsers.train import get_args
    # from plots import plot_metrics, plot_cross_subject
    from utils import set_global_seed, parse_dataset_class, get_k_fold_runs, get_loso_runs, get_simple_runs
    from datasets.base_class import EEGClassificationDataset
    # from models.sateer import SATEER

    import torchaudio

    # torchaudio.set_audio_backend("sox_io")
    with open('configs/deap.yaml', 'r') as fp:
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
    dataset_class = parse_dataset_class(name=args["dataset"])
    dataset: EEGClassificationDataset = dataset_class(
        path=args['dataset_path'],
        window_size=args['windows_size'],
        window_stride=args['windows_stride'],
        drop_last=True,
        discretize_labels=not args['dont_discretize_labels'],
        normalize_eegs=not args['dont_normalize_eegs'],
    )

    if args['setting'] == "cross_subject":
        if args['validation'] == "k_fold":
            runs = get_k_fold_runs(k=args["k"], dataset=dataset)
        elif args['validation'] == "loso":
            runs = get_loso_runs(dataset=dataset)
        elif args['validation'] == "simple":
            runs = get_simple_runs(dataset=dataset, train_perc=args["train_perc"])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    
    for i_run, run in enumerate(runs):
        logger.info(f"doing run {i_run+1} of {len(runs)} ({((i_run+1)/len(runs)) * 100:.1f}%)")
        
    # frees some memory
    del dataset
    gc.collect()
